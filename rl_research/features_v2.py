from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[1] / "out" / "rl_research"
ARCHIVE_DIR = OUT_DIR / "hl_archive"
CACHE_DIR = OUT_DIR / "cache_features_v4"

# Each pair is identified by its fully-qualified HL coin (dex:stock). Stocks that
# list on both cash and xyz are kept as two distinct pairs (own one-hot) so the
# model learns each book's microstructure; the stock (IEX) leg is shared.
CASH_COINS = [
    "cash:NVDA", "cash:TSLA", "cash:META", "cash:GOOGL", "cash:AMZN",
    "cash:MSFT", "cash:HOOD", "cash:EWY", "cash:INTC",
]
XYZ_COINS = [
    "xyz:INTC", "xyz:CRCL", "xyz:MSTR", "xyz:CRWV", "xyz:MU", "xyz:SNDK",
    "xyz:MRVL", "xyz:AMD", "xyz:PLTR", "xyz:COIN", "xyz:AAPL", "xyz:TSM",
    "xyz:ARM", "xyz:AVGO", "xyz:NVDA", "xyz:TSLA", "xyz:META", "xyz:GOOGL",
    "xyz:AMZN", "xyz:MSFT", "xyz:HOOD", "xyz:EWY",
]
SYMBOLS = CASH_COINS + XYZ_COINS
PAIRS = {coin: coin for coin in SYMBOLS}

RTH_START_UTC = 13 * 3600 + 30 * 60
RTH_END_UTC = 20 * 3600
SECONDS = RTH_END_UTC - RTH_START_UTC
WARMUP = 600
PRICE_LAGS = (3, 8, 21, 55, 144, 377)
PREMIUM_LAGS = (10, 30, 90, 300)
PREMIUM_Z_WINDOWS = (300, 900, 1800)
PERP_VOL_WINDOWS = (60, 300, 900)
PREMIUM_STRETCH_WINDOWS = (120, 600)
PREMIUM_VOL_WINDOWS = (60, 300)
PREMIUM_RANGE_WINDOW = 600

MARKET_FEATURES = (
    [f"perp_ret_{lag}s" for lag in PRICE_LAGS]
    + [f"stock_ret_{lag}s" for lag in PRICE_LAGS]
    + [f"premium_delta_{lag}s" for lag in PREMIUM_LAGS]
    + ["premium"]
    + [f"premium_z_{window}s" for window in PREMIUM_Z_WINDOWS]
    + [f"perp_vol_{window}s" for window in PERP_VOL_WINDOWS]
    + [f"premium_stretch_{window}s" for window in PREMIUM_STRETCH_WINDOWS]
    + [f"premium_vol_{window}s" for window in PREMIUM_VOL_WINDOWS]
    + [f"premium_range_{PREMIUM_RANGE_WINDOW}s"]
    + [
        "flow_imbalance_30s",
        "flow_imbalance_120s",
        "perp_prints_10s",
        "stock_prints_10s",
        "stock_staleness",
        "perp_vwap_dev_300s",
        "session_frac",
    ]
)
FEATURE_NAMES = MARKET_FEATURES + [f"is_{symbol}" for symbol in SYMBOLS]


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.cumsum(values)
    result = cumulative.copy()
    result[window:] = cumulative[window:] - cumulative[:-window]
    return result


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    mean = rolling_sum(values, window) / window
    mean_sq = rolling_sum(values**2, window) / window
    return np.sqrt(np.maximum(mean_sq - mean**2, 0.0))


def forward_fill(values: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(values)
    if not mask.any():
        return np.zeros_like(values)
    indices = np.where(mask, np.arange(len(values)), 0)
    np.maximum.accumulate(indices, out=indices)
    filled = values[indices]
    filled[np.isnan(filled)] = values[mask][0]
    return filled


def day_start_timestamp(date: str) -> int:
    day = time.strptime(date, "%Y%m%d")
    return int(time.mktime((day.tm_year, day.tm_mon, day.tm_mday, 0, 0, 0, 0, 0, 0)) - time.timezone)


def load_perp_seconds(path: Path, day_start: int) -> dict[str, np.ndarray]:
    last_price = np.full(SECONDS, np.nan)
    counts = np.zeros(SECONDS)
    buy_volume = np.zeros(SECONDS)
    sell_volume = np.zeros(SECONDS)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            second = int(row["time_ms"]) // 1000 - day_start - RTH_START_UTC
            if not 0 <= second < SECONDS:
                continue
            price = float(row["px"])
            notional = price * float(row["sz"])
            last_price[second] = price
            counts[second] += 1
            if row["crossed"] == "False":
                buy_volume[second] += notional
            else:
                sell_volume[second] += notional
    return {"px": last_price, "counts": counts, "buy": buy_volume, "sell": sell_volume}


def load_stock_seconds(path: Path, day_start: int) -> dict[str, np.ndarray]:
    last_price = np.full(SECONDS, np.nan)
    counts = np.zeros(SECONDS)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            second = int(row["time_ms"]) // 1000 - day_start - RTH_START_UTC
            if 0 <= second < SECONDS:
                last_price[second] = float(row["px"])
                counts[second] += 1
    return {"px": last_price, "counts": counts}


def staleness_counter(stock_counts: np.ndarray) -> np.ndarray:
    result = np.zeros(len(stock_counts))
    counter = 60.0
    for index in range(len(stock_counts)):
        counter = 0.0 if stock_counts[index] > 0 else min(counter + 1.0, 60.0)
        result[index] = counter
    return result


def compute_features(
    symbol: str,
    perp: np.ndarray,
    stock: np.ndarray,
    perp_buy: np.ndarray,
    perp_sell: np.ndarray,
    perp_counts: np.ndarray,
    stock_counts: np.ndarray,
    session_seconds: np.ndarray,
) -> np.ndarray:
    """Single source of truth for the feature matrix, shared by the offline
    backtest (build_episode) and the live engine. Inputs are forward-filled
    per-second price series plus raw per-second flow/print counts; everything is
    causal so the last row is valid for a real-time decision."""
    length = len(perp)
    premium = (stock / perp - 1) * 100

    columns: list[np.ndarray] = []
    for lag in PRICE_LAGS:
        shifted = np.zeros(length)
        shifted[lag:] = (perp[lag:] / perp[:-lag] - 1) * 100
        columns.append(shifted)
    for lag in PRICE_LAGS:
        shifted = np.zeros(length)
        shifted[lag:] = (stock[lag:] / stock[:-lag] - 1) * 100
        columns.append(shifted)
    for lag in PREMIUM_LAGS:
        shifted = np.zeros(length)
        shifted[lag:] = premium[lag:] - premium[:-lag]
        columns.append(shifted)
    columns.append(premium)
    for window in PREMIUM_Z_WINDOWS:
        mean = rolling_sum(premium, window) / window
        std = rolling_std(premium, window)
        columns.append((premium - mean) / (std + 1e-4))
    perp_returns = np.zeros(length)
    perp_returns[1:] = (perp[1:] / perp[:-1] - 1) * 100
    for window in PERP_VOL_WINDOWS:
        columns.append(rolling_std(perp_returns, window))
    for window in PREMIUM_STRETCH_WINDOWS:
        columns.append(premium - rolling_sum(premium, window) / window)
    premium_diff = np.zeros(length)
    premium_diff[1:] = premium[1:] - premium[:-1]
    for window in PREMIUM_VOL_WINDOWS:
        columns.append(rolling_std(premium_diff, window))
    premium_series = pd.Series(premium)
    low = premium_series.rolling(PREMIUM_RANGE_WINDOW, min_periods=1).min().to_numpy()
    high = premium_series.rolling(PREMIUM_RANGE_WINDOW, min_periods=1).max().to_numpy()
    columns.append((premium - low) / (high - low + 1e-4))

    for window in (30, 120):
        buys = rolling_sum(perp_buy, window)
        sells = rolling_sum(perp_sell, window)
        columns.append((buys - sells) / (buys + sells + 1.0))
    columns.append(np.log1p(rolling_sum(perp_counts, 10)))
    columns.append(np.log1p(rolling_sum(stock_counts, 10)))
    columns.append(staleness_counter(stock_counts))
    notional = perp_buy + perp_sell
    vwap_numerator = rolling_sum(perp * notional, 300)
    vwap_denominator = rolling_sum(notional, 300)
    vwap = np.where(vwap_denominator > 0, vwap_numerator / (vwap_denominator + 1e-9), perp)
    columns.append((perp / vwap - 1) * 100)
    columns.append(np.clip(session_seconds / SECONDS, 0.0, 1.0))

    one_hot_block = np.zeros((length, len(SYMBOLS)))
    one_hot_block[:, SYMBOLS.index(symbol)] = 1.0
    return np.concatenate([np.stack(columns, axis=1), one_hot_block], axis=1).astype(np.float32)


def build_episode(symbol: str, date: str) -> dict[str, np.ndarray] | None:
    stock = symbol.split(":")[-1]
    cache_path = CACHE_DIR / f"{symbol.replace(':', '_')}_{date}.npz"
    if cache_path.exists():
        loaded = np.load(cache_path)
        return {"features": loaded["features"], "perp": loaded["perp"]}
    iex_path = ARCHIVE_DIR / "iex" / date / f"iex_{stock}_{date}.csv"
    hl_path = ARCHIVE_DIR / "trades" / date / f"{PAIRS[symbol].replace(':', '_')}.csv"
    if not iex_path.exists() or not hl_path.exists():
        return None
    day_start = day_start_timestamp(date)
    perp_raw = load_perp_seconds(hl_path, day_start)
    stock_raw = load_stock_seconds(iex_path, day_start)
    if perp_raw["counts"].sum() < 500 or stock_raw["counts"].sum() < 500:
        return None
    perp = forward_fill(perp_raw["px"])
    stock = forward_fill(stock_raw["px"])
    features = compute_features(
        symbol, perp, stock,
        perp_raw["buy"], perp_raw["sell"], perp_raw["counts"], stock_raw["counts"],
        np.arange(SECONDS, dtype=np.float64),
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=features, perp=perp.astype(np.float64))
    return {"features": features, "perp": perp}


def effective_spread_bps(symbol: str, dates: list[str]) -> float:
    estimates = []
    for date in dates:
        hl_path = ARCHIVE_DIR / "trades" / date / f"{PAIRS[symbol].replace(':', '_')}.csv"
        if not hl_path.exists():
            continue
        prices, signs = [], []
        with hl_path.open() as handle:
            for row in csv.DictReader(handle):
                prices.append(float(row["px"]))
                signs.append(1 if row["crossed"] == "False" else -1)
        if len(prices) < 500:
            continue
        prices = np.array(prices)
        signs = np.array(signs)
        flips = np.where(signs[1:] != signs[:-1])[0]
        if len(flips) < 50:
            continue
        jumps = np.abs(np.diff(prices)[flips])
        estimates.append(float(np.median(jumps) / np.median(prices) * 1e4))
    return float(np.median(estimates)) if estimates else 4.0
