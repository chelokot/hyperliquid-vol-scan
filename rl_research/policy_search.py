from __future__ import annotations

import argparse
import csv
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Literal

import requests


API_URL = "https://api.hyperliquid.xyz/info"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "out" / "rl_research"
CACHE_DIR = OUT_DIR / "cache"
MAKER_FEE = 0.00015
TAKER_FEE = 0.00045
BASE_SLIPPAGE = 0.0002
UNIT_NOTIONAL = 100.0
MS_PER_DAY = 24 * 60 * 60 * 1000

MODES: tuple[Literal["breakout", "fade"], ...] = ("breakout", "fade")
SIDES: tuple[Literal["both", "long", "short"], ...] = ("both", "long", "short")
ENTRY_PCTS = (0.0, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02)
TP_PCTS = (0.0, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05)
SL_PCTS = (0.0, 0.003, 0.005, 0.0075, 0.01, 0.02, 0.05)
TIME_STOP_BARS = (0, 4, 8, 16, 32, 96)
COOLDOWN_BARS = (0, 2, 4, 8, 16)
TREND_LOOKBACK_BARS = (0, 16, 48, 96)
PARAM_KEYS = ("mode", "side", "entry_pct", "tp_pct", "sl_pct", "time_stop_bars", "cooldown_bars", "trend_lookback_bars")
PARAM_VALUES: dict[str, tuple[Any, ...]] = {
    "mode": MODES,
    "side": SIDES,
    "entry_pct": ENTRY_PCTS,
    "tp_pct": TP_PCTS,
    "sl_pct": SL_PCTS,
    "time_stop_bars": TIME_STOP_BARS,
    "cooldown_bars": COOLDOWN_BARS,
    "trend_lookback_bars": TREND_LOOKBACK_BARS,
}


@dataclass(frozen=True)
class Candle:
    time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Market:
    coin: str
    max_leverage: int
    day_ntl_vlm: float
    funding: float
    spread_pct: float

    @property
    def taker_slippage(self) -> float:
        return self.spread_pct / 100 / 2 + BASE_SLIPPAGE


@dataclass(frozen=True)
class Policy:
    mode: Literal["breakout", "fade"]
    side: Literal["both", "long", "short"]
    entry_pct: float
    tp_pct: float
    sl_pct: float
    time_stop_bars: int
    cooldown_bars: int
    trend_lookback_bars: int


@dataclass(frozen=True)
class WindowResult:
    pnl: float
    trades: int
    wins: int
    losses: int
    trade_pnls: tuple[float, ...]

    @property
    def multiplier(self) -> float:
        return 1.0 + self.pnl / UNIT_NOTIONAL


@dataclass(frozen=True)
class Score:
    windows: int
    geo_pct: float
    median_pct: float
    mean_pct: float
    positive_window_pct: float
    active_window_pct: float
    worst_window_pct: float
    avg_trades: float


def post(payload: dict[str, Any]) -> Any:
    last_error: Exception | None = None
    for attempt in range(10):
        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            if response.status_code == 429:
                time.sleep(0.8 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError("Hyperliquid API request failed")


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def as_float(value: Any) -> float:
    return 0.0 if value is None else float(value)


def live_spread_pct(coin: str) -> float:
    book = post({"type": "l2Book", "coin": coin})
    bids, asks = book["levels"]
    if not bids or not asks:
        return 999.0
    bid = float(bids[0]["px"])
    ask = float(asks[0]["px"])
    return (ask - bid) / ((ask + bid) / 2) * 100


def load_markets(max_coins: int, explicit_coins: set[str] | None, dex: str = "", min_volume: float = 500_000) -> list[Market]:
    payload: dict[str, Any] = {"type": "metaAndAssetCtxs"}
    if dex:
        payload["dex"] = dex
    meta, contexts = post(payload)
    markets: list[Market] = []
    for universe_item, context in zip(meta["universe"], contexts, strict=True):
        if universe_item.get("isDelisted"):
            continue
        coin = universe_item["name"]
        day_ntl_vlm = as_float(context.get("dayNtlVlm"))
        if explicit_coins is not None and coin not in explicit_coins:
            continue
        if explicit_coins is None and day_ntl_vlm < min_volume:
            continue
        markets.append(
            Market(
                coin=coin,
                max_leverage=int(universe_item["maxLeverage"]),
                day_ntl_vlm=day_ntl_vlm,
                funding=as_float(context.get("funding")),
                spread_pct=live_spread_pct(coin),
            )
        )
    markets.sort(key=lambda market: market.day_ntl_vlm, reverse=True)
    return markets[:max_coins] if explicit_coins is None else markets


def load_candles(coin: str, interval: str, lookback_days: int) -> list[Candle]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{coin.replace(':', '_')}_{interval}_{lookback_days}d.csv"
    if cache_path.exists():
        with cache_path.open() as handle:
            return [
                Candle(
                    time_ms=int(row["time_ms"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
                for row in csv.DictReader(handle)
            ]
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - lookback_days * MS_PER_DAY
    raw = post(
        {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
            },
        }
    )
    candles = [
        Candle(
            time_ms=int(candle["t"]),
            open=float(candle["o"]),
            high=float(candle["h"]),
            low=float(candle["l"]),
            close=float(candle["c"]),
            volume=float(candle["v"]),
        )
        for candle in sorted(raw, key=lambda item: int(item["t"]))
    ]
    with cache_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_ms", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        for candle in candles:
            writer.writerow(
                {
                    "time_ms": candle.time_ms,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                }
            )
    return candles


def trade_pnl(side: Literal["long", "short"], entry_price: float, exit_price: float, entry_fee: float, exit_fee: float) -> float:
    quantity = UNIT_NOTIONAL / entry_price
    gross = quantity * (exit_price - entry_price) if side == "long" else quantity * (entry_price - exit_price)
    return gross - UNIT_NOTIONAL * entry_fee - quantity * exit_price * exit_fee


@dataclass
class TradeState:
    side: Literal["flat", "long", "short"] = "flat"
    entry_price: float = 0.0
    take_profit: float = 0.0
    stop_loss: float = 0.0
    entry_fee: float = TAKER_FEE
    entry_index: int = 0


def simulate_window(candles: list[Candle], market: Market, policy: Policy, stress_cost: float = 0.0) -> WindowResult:
    slippage = market.taker_slippage
    anchor = candles[0].close
    state = TradeState()
    cooldown_until = 0
    pnl = 0.0
    trades = 0
    wins = 0
    losses = 0
    trade_pnls: list[float] = []

    def close_position(exit_price: float, exit_fee: float, next_anchor: float, index: int) -> None:
        nonlocal pnl, trades, wins, losses, anchor, cooldown_until
        result = trade_pnl(state.side, state.entry_price, exit_price, state.entry_fee, exit_fee) - UNIT_NOTIONAL * stress_cost
        pnl += result
        trades += 1
        wins += int(result > 0)
        losses += int(result < 0)
        trade_pnls.append(result)
        anchor = next_anchor
        state.side = "flat"
        cooldown_until = index + policy.cooldown_bars

    def trend_direction(index: int) -> Literal["up", "down", "none"]:
        lookback = policy.trend_lookback_bars
        if lookback == 0 or index - 1 - lookback < 0:
            return "none"
        return "up" if candles[index - 1].close > candles[index - 1 - lookback].close else "down"

    def exit_long(candle: Candle, index: int) -> bool:
        if policy.sl_pct and candle.low <= state.stop_loss:
            fill = min(state.stop_loss, candle.open)
            close_position(fill * (1.0 - slippage), TAKER_FEE, fill, index)
            return True
        if policy.tp_pct and candle.high > state.take_profit:
            close_position(state.take_profit, MAKER_FEE, state.take_profit, index)
            return True
        if policy.time_stop_bars and index - state.entry_index >= policy.time_stop_bars:
            close_position(candle.close * (1.0 - slippage), TAKER_FEE, candle.close, index)
            return True
        if policy.trend_lookback_bars and trend_direction(index) == "down":
            close_position(candle.close * (1.0 - slippage), TAKER_FEE, candle.close, index)
            return True
        return False

    def exit_short(candle: Candle, index: int) -> bool:
        if policy.sl_pct and candle.high >= state.stop_loss:
            fill = max(state.stop_loss, candle.open)
            close_position(fill * (1.0 + slippage), TAKER_FEE, fill, index)
            return True
        if policy.tp_pct and candle.low < state.take_profit:
            close_position(state.take_profit, MAKER_FEE, state.take_profit, index)
            return True
        if policy.time_stop_bars and index - state.entry_index >= policy.time_stop_bars:
            close_position(candle.close * (1.0 + slippage), TAKER_FEE, candle.close, index)
            return True
        if policy.trend_lookback_bars and trend_direction(index) == "up":
            close_position(candle.close * (1.0 + slippage), TAKER_FEE, candle.close, index)
            return True
        return False

    def open_position(side: Literal["long", "short"], fill_price: float, entry_fee: float, index: int) -> None:
        state.side = side
        state.entry_price = fill_price
        state.take_profit = fill_price * (1.0 + policy.tp_pct) if side == "long" else fill_price * (1.0 - policy.tp_pct)
        state.stop_loss = fill_price * (1.0 - policy.sl_pct) if side == "long" else fill_price * (1.0 + policy.sl_pct)
        state.entry_fee = entry_fee
        state.entry_index = index

    for index, candle in enumerate(candles[1:], start=1):
        if state.side == "long":
            if exit_long(candle, index):
                continue
        elif state.side == "short":
            if exit_short(candle, index):
                continue
        elif index >= cooldown_until:
            allow_long = policy.side in {"both", "long"}
            allow_short = policy.side in {"both", "short"}
            trend = trend_direction(index)
            if policy.trend_lookback_bars:
                allow_long = allow_long and trend == "up"
                allow_short = allow_short and trend == "down"
            if policy.entry_pct == 0.0:
                if allow_long and not allow_short:
                    open_position("long", candle.open * (1.0 + slippage), TAKER_FEE, index)
                elif allow_short and not allow_long:
                    open_position("short", candle.open * (1.0 - slippage), TAKER_FEE, index)
                if state.side == "long" and policy.sl_pct and candle.low <= state.stop_loss:
                    close_position(state.stop_loss * (1.0 - slippage), TAKER_FEE, state.stop_loss, index)
                elif state.side == "short" and policy.sl_pct and candle.high >= state.stop_loss:
                    close_position(state.stop_loss * (1.0 + slippage), TAKER_FEE, state.stop_loss, index)
                continue
            upper = anchor * (1.0 + policy.entry_pct)
            lower = anchor * (1.0 - policy.entry_pct)
            touched_upper = candle.high >= upper
            touched_lower = candle.low <= lower
            if touched_upper and touched_lower:
                continue
            if policy.mode == "breakout":
                if allow_long and touched_upper:
                    open_position("long", max(upper, candle.open) * (1.0 + slippage), TAKER_FEE, index)
                elif allow_short and touched_lower:
                    open_position("short", min(lower, candle.open) * (1.0 - slippage), TAKER_FEE, index)
            elif policy.mode == "fade":
                if allow_long and candle.low < lower:
                    open_position("long", lower, MAKER_FEE, index)
                elif allow_short and candle.high > upper:
                    open_position("short", upper, MAKER_FEE, index)
            if state.side == "long" and policy.sl_pct and candle.low <= state.stop_loss:
                close_position(state.stop_loss * (1.0 - slippage), TAKER_FEE, state.stop_loss, index)
            elif state.side == "short" and policy.sl_pct and candle.high >= state.stop_loss:
                close_position(state.stop_loss * (1.0 + slippage), TAKER_FEE, state.stop_loss, index)

    final_close = candles[-1].close
    if state.side == "long":
        final_pnl = trade_pnl("long", state.entry_price, final_close * (1.0 - slippage), state.entry_fee, TAKER_FEE) - UNIT_NOTIONAL * stress_cost
        pnl += final_pnl
        trade_pnls.append(final_pnl)
    elif state.side == "short":
        final_pnl = trade_pnl("short", state.entry_price, final_close * (1.0 + slippage), state.entry_fee, TAKER_FEE) - UNIT_NOTIONAL * stress_cost
        pnl += final_pnl
        trade_pnls.append(final_pnl)

    return WindowResult(pnl=pnl, trades=trades, wins=wins, losses=losses, trade_pnls=tuple(trade_pnls))


def build_windows(candles: list[Candle], window_bars: int, shift_bars: int) -> list[list[Candle]]:
    return [candles[index : index + window_bars] for index in range(0, len(candles) - window_bars + 1, shift_bars)]


def split_holdout(windows: list[list[Candle]], boundary_ms: int) -> tuple[list[list[Candle]], list[list[Candle]]]:
    development = [window for window in windows if window[-1].time_ms < boundary_ms]
    holdout = [window for window in windows if window[0].time_ms >= boundary_ms]
    return development, holdout


def split_train_validation(windows: list[list[Candle]]) -> tuple[list[list[Candle]], list[list[Candle]]]:
    train_end = int(len(windows) * 0.7)
    return windows[:train_end], windows[train_end:]


def score_results(results: list[WindowResult]) -> Score:
    multipliers = [result.multiplier for result in results]
    profits = [(multiplier - 1.0) * 100 for multiplier in multipliers]
    return Score(
        windows=len(results),
        geo_pct=(geometric_mean(multipliers) - 1.0) * 100,
        median_pct=median(profits),
        mean_pct=mean(profits),
        positive_window_pct=sum(1 for profit in profits if profit > 0) / len(profits) * 100,
        active_window_pct=sum(1 for result in results if result.trades > 0) / len(results) * 100,
        worst_window_pct=min(profits),
        avg_trades=mean(result.trades for result in results),
    )


def evaluate_windows(windows: list[list[Candle]], market: Market, policy: Policy, stress_cost: float = 0.0) -> Score:
    return score_results([simulate_window(window, market, policy, stress_cost) for window in windows])


def policy_score(score: Score) -> float:
    return (
        score.geo_pct * 50.0
        + score.positive_window_pct * 0.05
        + score.active_window_pct * 0.02
        - max(0.0, -score.worst_window_pct - 2.0) * 5.0
    )


def selection_score(train: Score, validation: Score, stress: Score, market: Market) -> float:
    min_geo = min(train.geo_pct, validation.geo_pct, stress.geo_pct)
    min_positive = min(train.positive_window_pct, validation.positive_window_pct)
    worst = min(train.worst_window_pct, validation.worst_window_pct, stress.worst_window_pct)
    overfit_penalty = max(0.0, train.geo_pct - validation.geo_pct) * 30.0
    return (
        min_geo * 60.0
        + min_positive * 0.08
        - max(0.0, -worst - 3.0) * 4.0
        - overfit_penalty
        - market.spread_pct * 5.0
        + min(market.day_ntl_vlm / 20_000_000, 2.0)
    )


def sample_policy(distribution: dict[str, list[float]], rng: random.Random) -> Policy:
    values: dict[str, Any] = {}
    for key in PARAM_KEYS:
        choices = PARAM_VALUES[key]
        values[key] = rng.choices(choices, weights=distribution[key], k=1)[0]
    return Policy(**values)


def uniform_distribution() -> dict[str, list[float]]:
    return {key: [1.0 / len(PARAM_VALUES[key])] * len(PARAM_VALUES[key]) for key in PARAM_KEYS}


def update_distribution(elites: list[Policy], previous: dict[str, list[float]], smoothing: float) -> dict[str, list[float]]:
    updated: dict[str, list[float]] = {}
    for key in PARAM_KEYS:
        choices = PARAM_VALUES[key]
        counts = [0.5] * len(choices)
        for policy in elites:
            counts[choices.index(getattr(policy, key))] += 1.0
        total = sum(counts)
        learned = [count / total for count in counts]
        updated[key] = [
            smoothing * previous[key][index] + (1.0 - smoothing) * learned[index]
            for index in range(len(choices))
        ]
    return updated


def policy_key(policy: Policy) -> tuple[Any, ...]:
    return tuple(getattr(policy, key) for key in PARAM_KEYS)


def train_policy(
    train_windows: list[list[Candle]],
    market: Market,
    generations: int,
    population: int,
    elite_fraction: float,
    rng: random.Random,
) -> tuple[Policy, Score, float]:
    distribution = uniform_distribution()
    best_policy: Policy | None = None
    best_score: Score | None = None
    best_value = -1e18
    evaluated: dict[tuple[Any, ...], tuple[Policy, Score, float]] = {}
    elite_count = max(4, int(population * elite_fraction))

    for _ in range(generations):
        generation: list[tuple[Policy, Score, float]] = []
        while len(generation) < population:
            policy = sample_policy(distribution, rng)
            key = policy_key(policy)
            if key in evaluated:
                item = evaluated[key]
            else:
                score = evaluate_windows(train_windows, market, policy)
                value = policy_score(score)
                item = (policy, score, value)
                evaluated[key] = item
            generation.append(item)
        generation.sort(key=lambda item: item[2], reverse=True)
        elites = [item[0] for item in generation[:elite_count]]
        distribution = update_distribution(elites, distribution, smoothing=0.35)
        if generation[0][2] > best_value:
            best_policy, best_score, best_value = generation[0]

    if best_policy is None or best_score is None:
        raise RuntimeError("No policy was trained")
    return best_policy, best_score, best_value


def null_calibration(
    holdout_windows: list[list[Candle]],
    market: Market,
    candidate_geo_pct: float,
    samples: int,
    seed_token: str,
) -> tuple[float, float]:
    rng = random.Random(seed_token)
    seen: set[tuple[Any, ...]] = set()
    null_geos: list[float] = []
    while len(null_geos) < samples:
        policy = sample_policy(uniform_distribution(), rng)
        key = policy_key(policy)
        if key in seen:
            continue
        seen.add(key)
        null_geos.append(evaluate_windows(holdout_windows, market, policy).geo_pct)
    null_geos.sort()
    beaten = sum(1 for geo in null_geos if geo < candidate_geo_pct)
    percentile = beaten / len(null_geos) * 100
    p95 = null_geos[int(len(null_geos) * 0.95)]
    return percentile, p95


def as_row(prefix: str, score: Score) -> dict[str, Any]:
    return {
        f"{prefix}_geo_pct": score.geo_pct,
        f"{prefix}_median_pct": score.median_pct,
        f"{prefix}_mean_pct": score.mean_pct,
        f"{prefix}_positive_pct": score.positive_window_pct,
        f"{prefix}_active_pct": score.active_window_pct,
        f"{prefix}_worst_pct": score.worst_window_pct,
        f"{prefix}_avg_trades": score.avg_trades,
        f"{prefix}_windows": score.windows,
    }


def policy_dict(policy: Policy) -> dict[str, Any]:
    return {
        "mode": policy.mode,
        "side": policy.side,
        "entry_pct": policy.entry_pct * 100,
        "tp_pct": policy.tp_pct * 100,
        "sl_pct": policy.sl_pct * 100,
        "time_stop_bars": policy.time_stop_bars,
        "cooldown_bars": policy.cooldown_bars,
        "trend_lookback_bars": policy.trend_lookback_bars,
    }


def evaluate_market(market: Market, args: argparse.Namespace) -> dict[str, Any] | None:
    rng = random.Random(f"{args.seed}:{market.coin}")
    candles = load_candles(market.coin, args.interval, args.lookback_days)
    if len(candles) < args.window_bars * 3:
        return None
    windows = build_windows(candles, args.window_bars, args.shift_bars)
    boundary_ms = candles[-1].time_ms - args.holdout_days * MS_PER_DAY
    development_windows, holdout_windows = split_holdout(windows, boundary_ms)
    if len(development_windows) < 60 or len(holdout_windows) < 24:
        return None
    train_windows, validation_windows = split_train_validation(development_windows)
    policy, train_score, train_objective = train_policy(
        train_windows,
        market,
        args.generations,
        args.population,
        args.elite_fraction,
        rng,
    )
    validation_score = evaluate_windows(validation_windows, market, policy)
    stress_score = evaluate_windows(development_windows, market, policy, args.stress_cost)
    holdout_score = evaluate_windows(holdout_windows, market, policy)
    holdout_stress_score = evaluate_windows(holdout_windows, market, policy, args.stress_cost)
    baseline = Policy("breakout", "both", 0.005, 0.015, 0.005, 0, 0, 0)
    baseline_holdout = evaluate_windows(holdout_windows, market, baseline)
    hold_long = Policy("breakout", "long", 0.0, 0.0, 0.0, 0, 0, 0)
    hold_holdout = evaluate_windows(holdout_windows, market, hold_long)
    null_percentile, null_p95 = null_calibration(
        holdout_windows,
        market,
        holdout_score.geo_pct,
        args.null_samples,
        f"{args.seed}:{market.coin}",
    )
    row: dict[str, Any] = {
        "coin": market.coin,
        "interval": args.interval,
        "max_leverage": market.max_leverage,
        "day_ntl_vlm": market.day_ntl_vlm,
        "funding": market.funding,
        "spread_pct": market.spread_pct,
        "candles": len(candles),
        "windows": len(windows),
        "train_objective": train_objective,
        "selection_score": selection_score(train_score, validation_score, stress_score, market),
        "baseline_holdout_geo_pct": baseline_holdout.geo_pct,
        "baseline_holdout_positive_pct": baseline_holdout.positive_window_pct,
        "baseline_holdout_worst_pct": baseline_holdout.worst_window_pct,
        "hold_holdout_geo_pct": hold_holdout.geo_pct,
        "holdout_null_percentile": null_percentile,
        "holdout_null_p95_geo_pct": null_p95,
        **policy_dict(policy),
        **as_row("train", train_score),
        **as_row("validation", validation_score),
        **as_row("stress", stress_score),
        **as_row("holdout", holdout_score),
        **as_row("holdout_stress", holdout_stress_score),
    }
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", default="")
    parser.add_argument("--dex", default="")
    parser.add_argument("--min-volume", type=float, default=500_000)
    parser.add_argument("--max-coins", type=int, default=40)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--holdout-days", type=int, default=7)
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--window-bars", type=int, default=24 * 4)
    parser.add_argument("--shift-bars", type=int, default=4)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--population", type=int, default=160)
    parser.add_argument("--elite-fraction", type=float, default=0.20)
    parser.add_argument("--stress-cost", type=float, default=0.001)
    parser.add_argument("--null-samples", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    explicit_coins = {coin.strip() for coin in args.coins.split(",") if coin.strip()} or None
    markets = load_markets(args.max_coins, explicit_coins, args.dex, args.min_volume)
    for market in markets:
        load_candles(market.coin, args.interval, args.lookback_days)
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_market, market, args): market for market in markets}
        for future, market in futures.items():
            try:
                row = future.result()
            except Exception as exc:
                print(f"skip {market.coin}: {exc}", flush=True)
                continue
            if row is None:
                print(f"skip {market.coin}: insufficient data", flush=True)
                continue
            rows.append(row)
            print(
                f"{market.coin:>8} {row['mode']} {row['side']} "
                f"entry={row['entry_pct']:.2f}% tp={row['tp_pct']:.2f}% sl={row['sl_pct']:.2f}% "
                f"train={row['train_geo_pct']:.3f}% val={row['validation_geo_pct']:.3f}% "
                f"holdout={row['holdout_geo_pct']:.3f}% null_pct={row['holdout_null_percentile']:.0f} "
                f"select={row['selection_score']:.1f}",
                flush=True,
            )

    if not rows:
        raise RuntimeError("No markets were evaluated")
    rows.sort(key=lambda item: item["selection_score"], reverse=True)
    output_path = OUT_DIR / f"policy_search_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(output_path)
    print("top by selection score (holdout shown for report only)")
    for row in rows[:20]:
        print(
            f"{row['coin']:>8} {row['mode']} {row['side']} "
            f"entry={row['entry_pct']:.2f}% tp={row['tp_pct']:.2f}% sl={row['sl_pct']:.2f}% "
            f"time={row['time_stop_bars']} cool={row['cooldown_bars']} "
            f"train={row['train_geo_pct']:.3f}% val={row['validation_geo_pct']:.3f}% "
            f"holdout={row['holdout_geo_pct']:.3f}% stress={row['holdout_stress_geo_pct']:.3f}% "
            f"baseline={row['baseline_holdout_geo_pct']:.3f}% null_pct={row['holdout_null_percentile']:.0f} "
            f"null_p95={row['holdout_null_p95_geo_pct']:.3f}%"
        )


if __name__ == "__main__":
    main()
