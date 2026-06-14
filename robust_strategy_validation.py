from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd

from advanced_strategy_search import Candle, Market, Strategy, load_candles, load_markets, simulate_window


OUT_DIR = Path(__file__).resolve().parent / "out"
SOURCE_CSV = OUT_DIR / "hyperliquid_advanced_search_20260611_141910.csv"
TOP_CANDIDATES = 750
WINDOW_15M = 24 * 4
SHIFT_15M = 4
WINDOW_1M = 24 * 60
SHIFT_1M = 60
EXTRA_STRESS_COST = 0.001
UNIT_NOTIONAL = 100.0


@dataclass(frozen=True)
class Score:
    windows: int
    geo_profit_pct: float
    median_profit_pct: float
    mean_profit_pct: float
    positive_window_pct: float
    active_window_pct: float
    worst_window_pct: float
    avg_trades: float


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def score_results(results: list[Any], stress_cost: float = 0.0) -> Score:
    multipliers: list[float] = []
    profits: list[float] = []
    for result in results:
        stressed_pnl = result.pnl - result.trades * UNIT_NOTIONAL * stress_cost
        multiplier = 1.0 + stressed_pnl / result.capital_at_risk if result.capital_at_risk > 0 else 1.0
        multipliers.append(multiplier)
        profits.append((multiplier - 1.0) * 100)
    return Score(
        windows=len(results),
        geo_profit_pct=(geometric_mean(multipliers) - 1.0) * 100,
        median_profit_pct=median(profits),
        mean_profit_pct=mean(profits),
        positive_window_pct=sum(1 for profit in profits if profit > 0) / len(profits) * 100,
        active_window_pct=sum(1 for result in results if result.trades > 0) / len(results) * 100,
        worst_window_pct=min(profits),
        avg_trades=mean(result.trades for result in results),
    )


def window_results(candles: list[Candle], market: Market, strategy: Strategy, window_points: int, shift_points: int) -> list[Any]:
    return [
        simulate_window(candles[index : index + window_points], market, strategy)
        for index in range(0, len(candles) - window_points + 1, shift_points)
    ]


def strategy_from_row(row: pd.Series) -> Strategy:
    return Strategy(
        mode=row["mode"],
        side_filter=row["side"],
        entry_pct=float(row["entry_pct"]) / 100,
        tp_pct=float(row["tp_pct"]) / 100,
        sl_pct=float(row["sl_pct"]) / 100,
    )


def main() -> None:
    candidates = pd.read_csv(SOURCE_CSV).sort_values("combined_score", ascending=False).head(TOP_CANDIDATES)
    markets = load_markets()
    candle_cache_15m: dict[str, list[Candle]] = {}
    candle_cache_1m: dict[str, list[Candle]] = {}
    rows: list[dict[str, Any]] = []

    for index, row in candidates.iterrows():
        coin = row["coin"]
        market = markets[coin]
        if coin not in candle_cache_15m:
            candle_cache_15m[coin] = load_candles(coin, "15m", 30 * 24 * 60 * 60 * 1000)
            candle_cache_1m[coin] = load_candles(coin, "1m", 5000 * 60 * 1000)
            print(f"loaded candles {coin}", flush=True)
        strategy = strategy_from_row(row)
        results_15m = window_results(candle_cache_15m[coin], market, strategy, WINDOW_15M, SHIFT_15M)
        split = int(len(results_15m) * 0.7)
        train = score_results(results_15m[:split])
        test = score_results(results_15m[split:])
        full = score_results(results_15m)
        stressed = score_results(results_15m, EXTRA_STRESS_COST)
        recent_1m = score_results(window_results(candle_cache_1m[coin], market, strategy, WINDOW_1M, SHIFT_1M))
        min_geo = min(train.geo_profit_pct, test.geo_profit_pct, recent_1m.geo_profit_pct, stressed.geo_profit_pct)
        min_win = min(train.positive_window_pct, test.positive_window_pct, recent_1m.positive_window_pct)
        worst = min(train.worst_window_pct, test.worst_window_pct, recent_1m.worst_window_pct)
        robust_score = (
            min_geo * 2.2
            + min_win * 0.12
            + min(train.active_window_pct, test.active_window_pct, recent_1m.active_window_pct) * 0.06
            - max(0.0, -worst - 20.0) * 1.3
            - market.spread_pct * 80.0
            + min(market.day_ntl_vlm / 10_000_000, 2.0)
        )
        rows.append(
            {
                "coin": coin,
                "mode": row["mode"],
                "side": row["side"],
                "entry_pct": row["entry_pct"],
                "tp_pct": row["tp_pct"],
                "sl_pct": row["sl_pct"],
                "max_leverage": market.max_leverage,
                "day_ntl_vlm": market.day_ntl_vlm,
                "spread_pct": market.spread_pct,
                "train_geo_pct": train.geo_profit_pct,
                "test_geo_pct": test.geo_profit_pct,
                "full_geo_pct": full.geo_profit_pct,
                "recent_1m_geo_pct": recent_1m.geo_profit_pct,
                "stressed_geo_pct": stressed.geo_profit_pct,
                "min_geo_pct": min_geo,
                "train_win_pct": train.positive_window_pct,
                "test_win_pct": test.positive_window_pct,
                "recent_1m_win_pct": recent_1m.positive_window_pct,
                "min_win_pct": min_win,
                "worst_window_pct": worst,
                "train_trades": train.avg_trades,
                "test_trades": test.avg_trades,
                "recent_1m_trades": recent_1m.avg_trades,
                "source_combined_score": row["combined_score"],
                "robust_score": robust_score,
            }
        )
        if len(rows) % 50 == 0:
            print(f"validated {len(rows)} / {len(candidates)}", flush=True)

    rows.sort(key=lambda item: item["robust_score"], reverse=True)
    output_path = OUT_DIR / "hyperliquid_robust_validation.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(output_path)
    for row in rows[:30]:
        print(
            f"{row['coin']:>7} {row['mode']} {row['side']} "
            f"entry={row['entry_pct']:.2f}% tp={row['tp_pct']:.2f}% sl={row['sl_pct']:.2f}% "
            f"train={row['train_geo_pct']:.2f}% test={row['test_geo_pct']:.2f}% "
            f"1m={row['recent_1m_geo_pct']:.2f}% stress={row['stressed_geo_pct']:.2f}% "
            f"worst={row['worst_window_pct']:.1f}% score={row['robust_score']:.1f}"
        )


if __name__ == "__main__":
    main()
