from __future__ import annotations

import argparse
import csv
import io
import time
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import requests

from policy_search import (
    OUT_DIR,
    Candle,
    Market,
    as_row,
    build_windows,
    evaluate_windows,
    load_markets,
    null_calibration,
)
from validate_1m_replay import latest_policy_csv, load_rows, row_policy

BINANCE_BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
CACHE_BINANCE = OUT_DIR / "cache_binance"


def month_starts(months_back: int) -> list[str]:
    today = date.today()
    year, month = today.year, today.month
    result: list[str] = []
    for _ in range(months_back):
        month -= 1
        if month == 0:
            year, month = year - 1, 12
        result.append(f"{year:04d}-{month:02d}")
    return sorted(result)


def fetch_month(symbol: str, month: str) -> list[Candle] | None:
    CACHE_BINANCE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_BINANCE / f"{symbol}-1m-{month}.csv"
    if not cache_path.exists():
        url = f"{BINANCE_BASE}/{symbol}/1m/{symbol}-1m-{month}.zip"
        response = requests.get(url, timeout=120)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        raw = archive.read(archive.namelist()[0]).decode()
        cache_path.write_text(raw)
    candles: list[Candle] = []
    for line in cache_path.read_text().splitlines():
        fields = line.split(",")
        if not fields[0].isdigit():
            continue
        open_time = int(fields[0])
        if open_time > 10**14:
            open_time //= 1000
        candles.append(
            Candle(
                time_ms=open_time,
                open=float(fields[1]),
                high=float(fields[2]),
                low=float(fields[3]),
                close=float(fields[4]),
                volume=float(fields[5]),
            )
        )
    return candles


def validate_row(row: dict[str, Any], market: Market, args: argparse.Namespace) -> dict[str, Any] | None:
    symbol = f"{row['coin']}USDT"
    policy = row_policy(row)
    monthly_geo: list[str] = []
    all_windows: list[list[Candle]] = []
    for month in month_starts(args.months):
        candles = fetch_month(symbol, month)
        if candles is None:
            monthly_geo.append(f"{month}:absent")
            continue
        windows = build_windows(candles, 24 * 60, args.shift_minutes)
        if len(windows) < 24:
            monthly_geo.append(f"{month}:short")
            continue
        score = evaluate_windows(windows, market, policy, args.stress_cost)
        monthly_geo.append(f"{month}:{score.geo_pct:.2f}%")
        all_windows.extend(windows)
    if len(all_windows) < 100:
        return None
    overall = evaluate_windows(all_windows, market, policy, args.stress_cost)
    null_windows = all_windows[:: args.null_window_stride]
    null_reference = evaluate_windows(null_windows, market, policy, args.stress_cost)
    null_percentile, null_p95 = null_calibration(
        null_windows,
        market,
        null_reference.geo_pct,
        args.null_samples,
        f"binance:{row['coin']}",
    )
    return {
        "coin": row["coin"],
        "symbol": symbol,
        "selection_score": row["selection_score"],
        "mode": row["mode"],
        "side": row["side"],
        "entry_pct": row["entry_pct"],
        "tp_pct": row["tp_pct"],
        "sl_pct": row["sl_pct"],
        "months": args.months,
        "monthly_geo": ";".join(monthly_geo),
        "binance_null_percentile": null_percentile,
        "binance_null_p95_geo_pct": null_p95,
        **as_row("binance", overall),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--months", type=int, default=3)
    parser.add_argument("--shift-minutes", type=int, default=60)
    parser.add_argument("--stress-cost", type=float, default=0.0)
    parser.add_argument("--null-samples", type=int, default=50)
    parser.add_argument("--null-window-stride", type=int, default=6)
    parser.add_argument("--max-coins", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.input) if args.input else latest_policy_csv()
    markets = {market.coin: market for market in load_markets(args.max_coins, None)}
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
        if output is None:
            print(f"skip {coin}: no binance history", flush=True)
            continue
        output_rows.append(output)
        print(
            f"{coin:>8} binance geo={output['binance_geo_pct']:.3f}% "
            f"positive={output['binance_positive_pct']:.0f}% worst={output['binance_worst_pct']:.2f}% "
            f"null_pct={output['binance_null_percentile']:.0f} [{output['monthly_geo']}]",
            flush=True,
        )
    if not output_rows:
        raise RuntimeError("No candidates were validated")
    output_rows.sort(key=lambda row: row["binance_geo_pct"], reverse=True)
    output_path = OUT_DIR / f"binance_history_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    print(output_path)


if __name__ == "__main__":
    main()
