from __future__ import annotations

import json
import time
from pathlib import Path

import requests

OUT_PATH = Path(__file__).resolve().parents[1] / "out" / "rl_research" / "spcx_leadlag.jsonl"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/SPCX?interval=1m&range=1d&includePrePost=true"
HL_URL = "https://api.hyperliquid.xyz/info"
YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0"}
POLL_SECONDS = 1.0


def yahoo_last() -> tuple[float, int]:
    response = requests.get(YAHOO_URL, headers=YAHOO_HEADERS, timeout=5)
    response.raise_for_status()
    result = response.json()["chart"]["result"][0]
    timestamps = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]
    for index in range(len(closes) - 1, -1, -1):
        if closes[index] is not None:
            return closes[index], timestamps[index]
    raise RuntimeError("no yahoo price")


def perp_book() -> tuple[float, float]:
    response = requests.post(HL_URL, json={"type": "l2Book", "coin": "xyz:SPCX"}, timeout=5)
    response.raise_for_status()
    bids, asks = response.json()["levels"]
    return float(bids[0]["px"]), float(asks[0]["px"])


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    yahoo_backoff_until = 0.0
    last_yahoo: tuple[float, int] | None = None
    while True:
        loop_started = time.time()
        record: dict = {"time_ms": int(loop_started * 1000)}
        if loop_started >= yahoo_backoff_until:
            try:
                last_yahoo = yahoo_last()
            except requests.HTTPError as exc:
                if exc.response.status_code == 429:
                    yahoo_backoff_until = loop_started + 30.0
                    record["yahoo_throttled"] = True
            except requests.RequestException:
                pass
        if last_yahoo is not None:
            record["stock_px"] = last_yahoo[0]
            record["stock_bar_ts"] = last_yahoo[1]
        try:
            bid, ask = perp_book()
            record["perp_bid"] = bid
            record["perp_ask"] = ask
        except requests.RequestException:
            pass
        if "stock_px" in record and "perp_bid" in record:
            record["basis_pct"] = ((record["perp_bid"] + record["perp_ask"]) / 2 / record["stock_px"] - 1) * 100
        with OUT_PATH.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        time.sleep(max(0.0, POLL_SECONDS - (time.time() - loop_started)))


if __name__ == "__main__":
    main()
