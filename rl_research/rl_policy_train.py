from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parents[1] / "out" / "rl_research"
LOOKBACK = 20
HIDDEN = 16
TAKER_FEE = 0.0001
MAKER_FEE = 0.0001
MIN_TP_PCT, MAX_TP_PCT = 0.05, 0.5
MIN_SL_PCT, MAX_SL_PCT = 0.1, 1.0


@dataclass(frozen=True)
class Tick:
    time_ms: int
    stock: float
    bid: float
    ask: float


STOCK_JUMP_LIMIT = 0.007
PERP_MOVE_CONFIRM = 0.003
PENDING_MATCH = 0.003
PENDING_CONFIRM_MS = 30_000
PREMIUM_CLAMP = 0.012


class StockOutlierFilter:
    def __init__(self) -> None:
        self.accepted: float | None = None
        self.pending: float | None = None
        self.pending_since_ms = 0
        self.previous_mid: float | None = None

    def accept(self, time_ms: int, stock: float, perp_mid: float) -> float:
        if self.accepted is None:
            self.accepted = stock
        else:
            perp_confirms = self.previous_mid is not None and abs(perp_mid / self.previous_mid - 1) > PERP_MOVE_CONFIRM
            if abs(stock / self.accepted - 1) <= STOCK_JUMP_LIMIT or perp_confirms:
                self.accepted = stock
                self.pending = None
            else:
                if self.pending is None or abs(stock / self.pending - 1) > PENDING_MATCH:
                    self.pending = stock
                    self.pending_since_ms = time_ms
                if time_ms - self.pending_since_ms >= PENDING_CONFIRM_MS:
                    self.accepted = stock
                    self.pending = None
        self.previous_mid = perp_mid
        lower = perp_mid * (1 - PREMIUM_CLAMP)
        upper = perp_mid * (1 + PREMIUM_CLAMP)
        return min(max(self.accepted, lower), upper)


def load_ticks(path: Path) -> list[Tick]:
    ticks: list[Tick] = []
    outlier_filter = StockOutlierFilter()
    for line in path.open():
        row = json.loads(line)
        if "basis_pct" not in row:
            continue
        mid = (row["perp_bid"] + row["perp_ask"]) / 2
        stock = outlier_filter.accept(row["time_ms"], row["stock_px"], mid)
        ticks.append(Tick(row["time_ms"], stock, row["perp_bid"], row["perp_ask"]))
    return ticks


PRICE_LAGS = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144)
PREMIUM_LAGS = (10, 30, 60, 120)
FEATURE_COUNT = 2 * len(PRICE_LAGS) + len(PREMIUM_LAGS) + 2


def build_features(ticks: list[Tick]) -> np.ndarray:
    stock = np.array([t.stock for t in ticks])
    bid = np.array([t.bid for t in ticks])
    ask = np.array([t.ask for t in ticks])
    mid = (bid + ask) / 2
    premium = (stock / mid - 1) * 100
    spread = (ask - bid) / mid * 100
    count = len(ticks)
    features = np.zeros((count, FEATURE_COUNT), dtype=np.float64)
    for column, lag in enumerate(PRICE_LAGS):
        features[lag:, column] = (stock[:-lag] / stock[lag:] - 1) * 100
        features[lag:, len(PRICE_LAGS) + column] = (mid[:-lag] / mid[lag:] - 1) * 100
    for column, lag in enumerate(PREMIUM_LAGS):
        features[lag:, 2 * len(PRICE_LAGS) + column] = premium[:-lag] - premium[lag:]
    features[:, -2] = premium
    features[:, -1] = spread
    return features


@dataclass
class PolicyShape:
    input_size: int = FEATURE_COUNT
    hidden: int = HIDDEN

    @property
    def param_count(self) -> int:
        return self.input_size * self.hidden + self.hidden + self.hidden * 3 + 3


def policy_forward(params: np.ndarray, shape: PolicyShape, feature_row: np.ndarray) -> tuple[bool, float, float]:
    offset = 0
    w1 = params[offset : offset + shape.input_size * shape.hidden].reshape(shape.input_size, shape.hidden)
    offset += shape.input_size * shape.hidden
    b1 = params[offset : offset + shape.hidden]
    offset += shape.hidden
    w2 = params[offset : offset + shape.hidden * 3].reshape(shape.hidden, 3)
    offset += shape.hidden * 3
    b2 = params[offset : offset + 3]
    hidden = np.tanh(feature_row @ w1 + b1)
    raw = hidden @ w2 + b2
    enter = raw[0] > 0
    tp_pct = MIN_TP_PCT + (MAX_TP_PCT - MIN_TP_PCT) / (1 + math.exp(-raw[1]))
    sl_pct = MIN_SL_PCT + (MAX_SL_PCT - MIN_SL_PCT) / (1 + math.exp(-raw[2]))
    return enter, tp_pct, sl_pct


def run_episode(params: np.ndarray, shape: PolicyShape, ticks: list[Tick], features: np.ndarray) -> tuple[float, int, int]:
    pnl = 0.0
    trades = 0
    wins = 0
    index = max(PRICE_LAGS[-1], PREMIUM_LAGS[-1])
    while index < len(ticks):
        enter, tp_pct, sl_pct = policy_forward(params, shape, features[index])
        if not enter:
            index += 1
            continue
        entry = ticks[index].bid
        take_profit = entry * (1 - tp_pct / 100)
        stop_loss = entry * (1 + sl_pct / 100)
        exit_return = None
        cursor = index + 1
        while cursor < len(ticks):
            ask = ticks[cursor].ask
            if ask >= stop_loss:
                exit_return = (entry - ask) / entry - TAKER_FEE - TAKER_FEE
                break
            if ask <= take_profit:
                exit_return = (entry - take_profit) / entry - TAKER_FEE - MAKER_FEE
                break
            cursor += 1
        if exit_return is None:
            exit_return = (entry - ticks[-1].ask) / entry - TAKER_FEE - TAKER_FEE
            cursor = len(ticks) - 1
        pnl += exit_return
        trades += 1
        wins += exit_return > 0
        index = cursor + 1
    return pnl * 100, trades, wins


def train(ticks: list[Tick], features: np.ndarray, generations: int, population: int, seed: int) -> np.ndarray:
    shape = PolicyShape()
    rng = np.random.default_rng(seed)
    mean = np.zeros(shape.param_count)
    std = np.ones(shape.param_count)
    elite_count = max(4, population // 5)
    best_params = mean.copy()
    best_score = -1e18
    for generation in range(generations):
        samples = rng.normal(mean, std, size=(population, shape.param_count))
        scored = []
        for candidate in samples:
            pnl, trades, _ = run_episode(candidate, shape, ticks, features)
            inactivity_penalty = 0.05 if trades == 0 else 0.0
            scored.append((pnl - inactivity_penalty, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        elites = np.array([candidate for _, candidate in scored[:elite_count]])
        mean = 0.3 * mean + 0.7 * elites.mean(axis=0)
        std = 0.3 * std + 0.7 * (elites.std(axis=0) + 0.02)
        if scored[0][0] > best_score:
            best_score, best_params = scored[0][0], scored[0][1].copy()
        print(f"gen {generation + 1}: best train pnl so far {best_score:+.3f}%", flush=True)
    return best_params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(OUT_DIR / "spcx_leadlag.jsonl"))
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    ticks = load_ticks(Path(args.data))
    features = build_features(ticks)
    third = len(ticks) // 3
    splits = {
        "train": (0, third),
        "validation": (third, 2 * third),
        "test": (2 * third, len(ticks)),
    }
    span_minutes = (ticks[-1].time_ms - ticks[0].time_ms) / 60000
    print(f"dataset: {len(ticks)} ticks, {span_minutes:.0f} min; split thirds of {third} ticks")
    shape = PolicyShape()
    train_slice = ticks[: splits["train"][1]]
    params = train(train_slice, features[: splits["train"][1]], args.generations, args.population, args.seed)
    for name, (start, end) in splits.items():
        pnl, trades, wins = run_episode(params, shape, ticks[start:end], features[start:end])
        winrate = wins / trades * 100 if trades else 0.0
        print(f"{name:>10}: pnl {pnl:+.3f}% trades {trades} winrate {winrate:.0f}%")
    weights_path = OUT_DIR / "rl_policy_weights.npy"
    np.save(weights_path, params)
    print(weights_path)


if __name__ == "__main__":
    main()
