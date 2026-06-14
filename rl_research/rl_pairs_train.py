from __future__ import annotations

import argparse
import csv
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

TRAIN_EPISODES: list[dict] = []


def score_candidate(candidate: np.ndarray) -> tuple[float, int]:
    pnl, trades, _ = evaluate(candidate, TRAIN_EPISODES)
    return pnl, trades

OUT_DIR = Path(__file__).resolve().parents[1] / "out" / "rl_research"
ARCHIVE_DIR = OUT_DIR / "hl_archive"

PAIRS = {
    "NVDA": "cash:NVDA",
    "TSLA": "cash:TSLA",
    "META": "cash:META",
    "GOOGL": "cash:GOOGL",
    "AMZN": "cash:AMZN",
    "MSFT": "cash:MSFT",
    "HOOD": "cash:HOOD",
    "EWY": "cash:EWY",
    "INTC": "xyz:INTC",
    "CRCL": "xyz:CRCL",
    "MSTR": "xyz:MSTR",
    "CRWV": "xyz:CRWV",
    "MU": "xyz:MU",
    "SNDK": "xyz:SNDK",
}
TRAIN_PAIRS = ["NVDA", "GOOGL", "AMZN", "HOOD", "INTC", "MSTR", "MU", "EWY", "CRCL"]
VALIDATION_PAIRS = ["TSLA", "MSFT", "CRWV"]
TEST_PAIRS = ["META", "SNDK"]

RTH_START_UTC = 13 * 3600 + 30 * 60
RTH_END_UTC = 20 * 3600
PRICE_LAGS = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144)
PREMIUM_LAGS = (10, 30, 60, 120)
WARMUP = 300
FEATURE_COUNT = 2 * len(PRICE_LAGS) + len(PREMIUM_LAGS) + 1 + 2 + 2 + 1

TAKER_FEE = 0.0001
MAKER_FEE = 0.00003
SLIPPAGE = 0.0003
MIN_TP_PCT, MAX_TP_PCT = 0.05, 0.5
MIN_SL_PCT, MAX_SL_PCT = 0.1, 1.0

HIDDEN_1 = 32
HIDDEN_2 = 16
PARAM_COUNT = FEATURE_COUNT * HIDDEN_1 + HIDDEN_1 + HIDDEN_1 * HIDDEN_2 + HIDDEN_2 + HIDDEN_2 * 4 + 4


def load_second_series(path: Path, day_start_ts: int) -> tuple[np.ndarray, np.ndarray]:
    seconds_count = RTH_END_UTC - RTH_START_UTC
    last_price = np.full(seconds_count, np.nan)
    counts = np.zeros(seconds_count)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            second = int(row["time_ms"]) // 1000 - day_start_ts - RTH_START_UTC
            if 0 <= second < seconds_count:
                last_price[second] = float(row["px"])
                counts[second] += 1
    return last_price, counts


def forward_fill(values: np.ndarray) -> np.ndarray:
    filled = values.copy()
    mask = np.isnan(filled)
    indices = np.where(~mask, np.arange(len(filled)), 0)
    np.maximum.accumulate(indices, out=indices)
    filled = filled[indices]
    filled[np.isnan(filled)] = filled[~np.isnan(filled)][0] if (~np.isnan(filled)).any() else 0.0
    return filled


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.cumsum(values)
    result = cumulative.copy()
    result[window:] = cumulative[window:] - cumulative[:-window]
    return result


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    mean = rolling_sum(values, window) / window
    mean_sq = rolling_sum(values**2, window) / window
    return np.sqrt(np.maximum(mean_sq - mean**2, 0.0))


def build_pair(symbol: str, date: str) -> dict[str, np.ndarray] | None:
    iex_path = ARCHIVE_DIR / "iex" / date / f"iex_{symbol}_{date}.csv"
    hl_coin = PAIRS[symbol]
    hl_path = ARCHIVE_DIR / "trades" / date / f"{hl_coin.replace(':', '_')}.csv"
    if not iex_path.exists() or not hl_path.exists():
        return None
    day = time.strptime(date, "%Y%m%d")
    day_start_ts = int(time.mktime((day.tm_year, day.tm_mon, day.tm_mday, 0, 0, 0, 0, 0, 0)) - time.timezone)
    stock_px, stock_counts = load_second_series(iex_path, day_start_ts)
    perp_px, perp_counts = load_second_series(hl_path, day_start_ts)
    if stock_counts.sum() < 200 or perp_counts.sum() < 200:
        return None
    stock = forward_fill(stock_px)
    perp = forward_fill(perp_px)
    premium = (stock / perp - 1) * 100
    seconds_count = len(stock)
    features = np.zeros((seconds_count, FEATURE_COUNT))
    column = 0
    for lag in PRICE_LAGS:
        features[lag:, column] = (stock[:-lag] / stock[lag:] - 1) * 100
        features[lag:, column + len(PRICE_LAGS)] = (perp[:-lag] / perp[lag:] - 1) * 100
        column += 1
    column = 2 * len(PRICE_LAGS)
    for lag in PREMIUM_LAGS:
        features[lag:, column] = premium[:-lag] - premium[lag:]
        column += 1
    features[:, column] = premium
    column += 1
    perp_returns = np.zeros(seconds_count)
    perp_returns[1:] = (perp[1:] / perp[:-1] - 1) * 100
    features[:, column] = rolling_std(perp_returns, 60)
    features[:, column + 1] = rolling_std(perp_returns, 300)
    column += 2
    features[:, column] = np.log1p(rolling_sum(stock_counts, 10))
    features[:, column + 1] = np.log1p(rolling_sum(perp_counts, 10))
    column += 2
    features[:, column] = np.linspace(0, 1, seconds_count)
    return {"perp": perp, "features": features}


def unpack(params: np.ndarray) -> tuple:
    offset = 0
    w1 = params[offset : offset + FEATURE_COUNT * HIDDEN_1].reshape(FEATURE_COUNT, HIDDEN_1)
    offset += FEATURE_COUNT * HIDDEN_1
    b1 = params[offset : offset + HIDDEN_1]
    offset += HIDDEN_1
    w2 = params[offset : offset + HIDDEN_1 * HIDDEN_2].reshape(HIDDEN_1, HIDDEN_2)
    offset += HIDDEN_1 * HIDDEN_2
    b2 = params[offset : offset + HIDDEN_2]
    offset += HIDDEN_2
    w3 = params[offset : offset + HIDDEN_2 * 4].reshape(HIDDEN_2, 4)
    offset += HIDDEN_2 * 4
    b3 = params[offset : offset + 4]
    return w1, b1, w2, b2, w3, b3


def run_pair_episode(params: np.ndarray, pair: dict[str, np.ndarray]) -> tuple[float, int, int]:
    w1, b1, w2, b2, w3, b3 = unpack(params)
    raw = np.tanh(np.tanh(pair["features"] @ w1 + b1) @ w2 + b2) @ w3 + b3
    long_signal = (raw[:, 0] > 0) & (raw[:, 0] >= raw[:, 1])
    short_signal = (raw[:, 1] > 0) & (raw[:, 1] > raw[:, 0])
    any_signal = long_signal | short_signal
    tp_pct = (MIN_TP_PCT + (MAX_TP_PCT - MIN_TP_PCT) / (1 + np.exp(-raw[:, 2]))) / 100
    sl_pct = (MIN_SL_PCT + (MAX_SL_PCT - MIN_SL_PCT) / (1 + np.exp(-raw[:, 3]))) / 100
    perp = pair["perp"]
    seconds_count = len(perp)
    pnl = 0.0
    trades = 0
    wins = 0
    cursor = WARMUP
    while cursor < seconds_count - 2:
        offsets = np.flatnonzero(any_signal[cursor:])
        if len(offsets) == 0:
            break
        entry_index = cursor + offsets[0]
        if entry_index >= seconds_count - 2:
            break
        is_long = long_signal[entry_index]
        direction = 1.0 if is_long else -1.0
        entry_price = perp[entry_index + 1] * (1 + direction * SLIPPAGE)
        take_profit = entry_price * (1 + direction * tp_pct[entry_index])
        stop_loss = entry_price * (1 - direction * sl_pct[entry_index])
        path = perp[entry_index + 2 :]
        if is_long:
            tp_hits = path >= take_profit
            sl_hits = path <= stop_loss
        else:
            tp_hits = path <= take_profit
            sl_hits = path >= stop_loss
        tp_first = np.argmax(tp_hits) if tp_hits.any() else seconds_count
        sl_first = np.argmax(sl_hits) if sl_hits.any() else seconds_count
        if sl_first <= tp_first and sl_first < seconds_count:
            exit_offset = sl_first
            exit_price = min(stop_loss, path[exit_offset]) if is_long else max(stop_loss, path[exit_offset])
            exit_price *= 1 - direction * SLIPPAGE
            exit_fee = TAKER_FEE
        elif tp_first < seconds_count:
            exit_offset = tp_first
            exit_price = take_profit
            exit_fee = MAKER_FEE
        else:
            exit_offset = len(path) - 1
            exit_price = path[-1] * (1 - direction * SLIPPAGE)
            exit_fee = TAKER_FEE
        trade_return = direction * (exit_price - entry_price) / entry_price - TAKER_FEE - exit_fee
        pnl += trade_return
        trades += 1
        wins += trade_return > 0
        cursor = entry_index + 2 + exit_offset + 1
    return pnl * 100, trades, wins


def evaluate(params: np.ndarray, episodes: list[dict]) -> tuple[float, int, int]:
    total_pnl = 0.0
    total_trades = 0
    total_wins = 0
    for episode in episodes:
        pnl, trades, wins = run_pair_episode(params, episode)
        total_pnl += pnl
        total_trades += trades
        total_wins += wins
    return total_pnl / len(episodes), total_trades, total_wins


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", default="20260611")
    parser.add_argument("--holdout-days", type=int, default=3)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    print(f"params: {PARAM_COUNT}, features: {FEATURE_COUNT}")
    dates = sorted(date.strip() for date in args.dates.split(",") if date.strip())
    early_dates = dates[: len(dates) - args.holdout_days] if len(dates) > args.holdout_days else dates
    late_dates = dates[len(early_dates) :]
    episodes: dict[tuple[str, str], dict] = {}
    for date in dates:
        for symbol in PAIRS:
            episode = build_pair(symbol, date)
            if episode is not None:
                episodes[(date, symbol)] = episode
    train_data = [ep for (date, symbol), ep in episodes.items() if date in early_dates and symbol in TRAIN_PAIRS]
    validation_data = [
        ep
        for (date, symbol), ep in episodes.items()
        if (date in early_dates and symbol in VALIDATION_PAIRS) or (date in late_dates and symbol in TRAIN_PAIRS)
    ]
    test_data = [
        ep for (date, symbol), ep in episodes.items() if date in late_dates and symbol in (VALIDATION_PAIRS + TEST_PAIRS)
    ]
    test_pure = [ep for (date, symbol), ep in episodes.items() if date in late_dates and symbol in TEST_PAIRS]
    print(
        f"days: {len(early_dates)} train + {len(late_dates)} holdout | episodes: train {len(train_data)}, "
        f"val {len(validation_data)}, test {len(test_data)} (pure {len(test_pure)})"
    )

    global TRAIN_EPISODES
    TRAIN_EPISODES = train_data
    pool = ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context("fork"))
    mean = np.zeros(PARAM_COUNT)
    std = np.ones(PARAM_COUNT) * 0.5
    elite_count = max(6, args.population // 6)
    best_val = -1e18
    best_params = mean.copy()
    started = time.time()
    for generation in range(1, args.generations + 1):
        samples = rng.normal(mean, std, size=(args.population, PARAM_COUNT))
        results = list(pool.map(score_candidate, samples, chunksize=4))
        scored = [
            (pnl - (0.1 if trades == 0 else 0.0), candidate)
            for (pnl, trades), candidate in zip(results, samples)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        elites = np.array([candidate for _, candidate in scored[: elite_count]])
        mean = 0.3 * mean + 0.7 * elites.mean(axis=0)
        std = np.maximum(0.3 * std + 0.7 * elites.std(axis=0), 0.03)
        for _, candidate in scored[:3]:
            val_pnl, val_trades, _ = evaluate(candidate, validation_data)
            if val_pnl > best_val and val_trades > 0:
                best_val = val_pnl
                best_params = candidate.copy()
        if generation % 5 == 0 or generation == 1:
            train_best = scored[0][0]
            elapsed = time.time() - started
            print(
                f"gen {generation}: train_best {train_best:+.3f}%/pair, best_val {best_val:+.3f}%/pair, {elapsed:.0f}s",
                flush=True,
            )

    np.save(OUT_DIR / "rl_pairs_weights.npy", best_params)
    print("\nfinal evaluation of best-by-validation model:")
    for name, subset in (
        ("train", train_data),
        ("validation", validation_data),
        ("test", test_data),
        ("test_pure", test_pure),
    ):
        if not subset:
            continue
        pnl, trades, wins = evaluate(best_params, subset)
        winrate = wins / trades * 100 if trades else 0.0
        print(f"{name:>10}: {pnl:+.3f}%/эпизод (пара-день), trades {trades}, winrate {winrate:.0f}%")
    for (date, symbol), episode in episodes.items():
        if date in late_dates and symbol in TEST_PAIRS:
            pnl, trades, _ = run_pair_episode(best_params, episode)
            print(f"  {date} {symbol}: {pnl:+.3f}%, {trades} trades")


if __name__ == "__main__":
    main()
