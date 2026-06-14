from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lz4.frame

ARCHIVE_DIR = Path(__file__).resolve().parents[1] / "out" / "rl_research" / "hl_archive"
BUCKET = "s3://hl-mainnet-node-data/node_fills_by_block/hourly"
FIELDNAMES = ["time_ms", "px", "sz", "side", "tid", "crossed"]
HOUR_WORKERS = 6


def download_hour(date: str, hour: int) -> Path | None:
    target = ARCHIVE_DIR / date / f"{hour}.lz4"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["aws", "s3", "cp", "--request-payer", "requester", f"{BUCKET}/{date}/{hour}.lz4", str(target), "--quiet"],
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return target


def process_hour(args: tuple[str, int, set[str] | None]) -> dict[str, dict[int, tuple]]:
    date, hour, coins = args
    path = download_hour(date, hour)
    if path is None:
        return {}
    hour_trades: dict[str, dict[int, tuple]] = defaultdict(dict)
    with lz4.frame.open(path, "rt") as handle:
        for line in handle:
            block = json.loads(line)
            for _user, fill in block["events"]:
                if coins is not None and fill["coin"] not in coins:
                    continue
                hour_trades[fill["coin"]][fill["tid"]] = (
                    fill["time"],
                    fill["px"],
                    fill["sz"],
                    fill["side"],
                    fill["tid"],
                    fill["crossed"],
                )
    return hour_trades


def extract_day(date: str, hours: list[int], coins: set[str] | None) -> None:
    trades: dict[str, dict[int, tuple]] = defaultdict(dict)
    with ProcessPoolExecutor(max_workers=HOUR_WORKERS) as pool:
        for hour_trades in pool.map(process_hour, [(date, hour, coins) for hour in hours]):
            for coin, by_tid in hour_trades.items():
                trades[coin].update(by_tid)
    print(f"{date}: parsed {len(hours)} hours, {len(trades)} coins", flush=True)
    out_dir = ARCHIVE_DIR / "trades" / date
    out_dir.mkdir(parents=True, exist_ok=True)
    for coin, by_tid in trades.items():
        rows = sorted(by_tid.values())
        path = out_dir / f"{coin.replace(':', '_').replace('/', '_')}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(FIELDNAMES)
            writer.writerows(rows)
    totals = sorted(((len(by_tid), coin) for coin, by_tid in trades.items()), reverse=True)
    print(f"\n{len(trades)} markets -> {out_dir}")
    for count, coin in totals[:15]:
        print(f"  {coin:>16}: {count} trades")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--hours", default="0-23")
    parser.add_argument("--coins", default="")
    args = parser.parse_args()
    if "-" in args.hours:
        start, end = args.hours.split("-")
        hours = list(range(int(start), int(end) + 1))
    else:
        hours = [int(part) for part in args.hours.split(",")]
    coins = {coin.strip() for coin in args.coins.split(",") if coin.strip()} or None
    extract_day(args.date, hours, coins)


if __name__ == "__main__":
    main()
