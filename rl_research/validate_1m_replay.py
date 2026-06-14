from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

from policy_search import (
    OUT_DIR,
    Market,
    Policy,
    as_row,
    build_windows,
    evaluate_windows,
    load_candles,
    load_markets,
    null_calibration,
)


def latest_policy_csv() -> Path:
    candidates = sorted(OUT_DIR.glob("policy_search_*.csv"))
    if not candidates:
        raise RuntimeError("No policy_search CSV found")
    return candidates[-1]


def load_rows(path: Path, top: int) -> list[dict[str, Any]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: float(row["selection_score"]), reverse=True)
    return rows[:top]


def row_policy(row: dict[str, Any]) -> Policy:
    minutes_per_bar = int(row["interval"].rstrip("m"))
    return Policy(
        mode=row["mode"],
        side=row["side"],
        entry_pct=float(row["entry_pct"]) / 100,
        tp_pct=float(row["tp_pct"]) / 100,
        sl_pct=float(row["sl_pct"]) / 100,
        time_stop_bars=int(row["time_stop_bars"]) * minutes_per_bar,
        cooldown_bars=int(row["cooldown_bars"]) * minutes_per_bar,
        trend_lookback_bars=int(row["trend_lookback_bars"]) * minutes_per_bar,
    )


def market_by_coin(max_coins: int, dex: str = "", min_volume: float = 50_000) -> dict[str, Market]:
    return {market.coin: market for market in load_markets(max_coins, None, dex, min_volume)}


def validate_row(row: dict[str, Any], market: Market, args: argparse.Namespace) -> dict[str, Any]:
    candles = load_candles(row["coin"], "1m", args.lookback_days)
    windows = build_windows(candles, 24 * 60, 60)
    if len(windows) < 24:
        raise RuntimeError("insufficient 1m windows")
    policy = row_policy(row)
    replay = evaluate_windows(windows, market, policy)
    replay_stress = evaluate_windows(windows, market, policy, args.stress_cost)
    null_percentile, null_p95 = null_calibration(
        windows,
        market,
        replay.geo_pct,
        args.null_samples,
        f"1m:{row['coin']}",
    )
    return {
        "coin": row["coin"],
        "selection_score": row["selection_score"],
        "holdout_15m_geo_pct": row["holdout_geo_pct"],
        "mode": row["mode"],
        "side": row["side"],
        "entry_pct": row["entry_pct"],
        "tp_pct": row["tp_pct"],
        "sl_pct": row["sl_pct"],
        "time_stop_1m_bars": policy.time_stop_bars,
        "cooldown_1m_bars": policy.cooldown_bars,
        "day_ntl_vlm": market.day_ntl_vlm,
        "spread_pct": market.spread_pct,
        "candles_1m": len(candles),
        "windows_1m": len(windows),
        "replay_null_percentile": null_percentile,
        "replay_null_p95_geo_pct": null_p95,
        **as_row("replay", replay),
        **as_row("replay_stress", replay_stress),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="")
    parser.add_argument("--dex", default="")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--stress-cost", type=float, default=0.0015)
    parser.add_argument("--null-samples", type=int, default=100)
    parser.add_argument("--max-coins", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.input) if args.input else latest_policy_csv()
    markets = market_by_coin(args.max_coins, args.dex)
    output_rows: list[dict[str, Any]] = []
    for row in load_rows(source_path, args.top):
        coin = row["coin"]
        if coin not in markets:
            print(f"skip {coin}: market not loaded", flush=True)
            continue
        try:
            output = validate_row(row, markets[coin], args)
        except Exception as exc:
            print(f"skip {coin}: {exc}", flush=True)
            continue
        output_rows.append(output)
        print(
            f"{coin:>8} 1m replay={output['replay_geo_pct']:.3f}% "
            f"stress={output['replay_stress_geo_pct']:.3f}% "
            f"worst={min(output['replay_worst_pct'], output['replay_stress_worst_pct']):.2f}% "
            f"null_pct={output['replay_null_percentile']:.0f}",
            flush=True,
        )
    if not output_rows:
        raise RuntimeError("No candidates were validated")
    output_rows.sort(key=lambda row: row["replay_geo_pct"], reverse=True)
    output_path = OUT_DIR / f"policy_1m_replay_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    print(output_path)
    print("top by 1m replay (report only, selection already frozen)")
    for row in output_rows[:10]:
        print(
            f"{row['coin']:>8} entry={float(row['entry_pct']):.2f}% tp={float(row['tp_pct']):.2f}% "
            f"sl={float(row['sl_pct']):.2f}% replay={row['replay_geo_pct']:.3f}% "
            f"stress={row['replay_stress_geo_pct']:.3f}% null_pct={row['replay_null_percentile']:.0f}"
        )


if __name__ == "__main__":
    main()
