from __future__ import annotations

import csv
import time
from pathlib import Path

from policy_search import Candle, load_markets, post

STORE_DIR = Path(__file__).resolve().parents[1] / "out" / "rl_research" / "candles_1m"
FETCH_INTERVAL_SECONDS = 30 * 60
MAX_COINS = 80
FIELDNAMES = ["time_ms", "open", "high", "low", "close", "volume"]


def read_store(path: Path) -> dict[int, Candle]:
    if not path.exists():
        return {}
    with path.open() as handle:
        return {
            int(row["time_ms"]): Candle(
                time_ms=int(row["time_ms"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            for row in csv.DictReader(handle)
        }


def write_store(path: Path, candles: dict[int, Candle]) -> None:
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for time_ms in sorted(candles):
            candle = candles[time_ms]
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
    temp_path.replace(path)


def fetch_recent(coin: str) -> list[Candle]:
    end_ms = int(time.time() * 1000)
    raw = post(
        {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": "1m",
                "startTime": end_ms - 4 * 24 * 60 * 60 * 1000,
                "endTime": end_ms,
            },
        }
    )
    return [
        Candle(
            time_ms=int(candle["t"]),
            open=float(candle["o"]),
            high=float(candle["h"]),
            low=float(candle["l"]),
            close=float(candle["c"]),
            volume=float(candle["v"]),
        )
        for candle in raw
    ]


def run_cycle() -> None:
    cutoff_ms = int(time.time() * 1000) - 2 * 60 * 1000
    for market in load_markets(MAX_COINS, None):
        path = STORE_DIR / f"{market.coin}.csv"
        stored = read_store(path)
        fetched = {
            candle.time_ms: candle
            for candle in fetch_recent(market.coin)
            if candle.time_ms < cutoff_ms
        }
        if not fetched:
            continue
        stored.update(fetched)
        write_store(path, stored)
        print(f"{market.coin}: {len(stored)} candles stored", flush=True)


def main() -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        cycle_started = time.time()
        try:
            run_cycle()
        except Exception as exc:
            print(f"cycle failed: {exc}", flush=True)
        elapsed = time.time() - cycle_started
        time.sleep(max(60.0, FETCH_INTERVAL_SECONDS - elapsed))


if __name__ == "__main__":
    main()
