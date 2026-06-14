from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HL_COINS = (
    "cash:NVDA,cash:TSLA,cash:META,cash:GOOGL,cash:AMZN,cash:MSFT,cash:HOOD,cash:EWY,"
    "xyz:INTC,xyz:CRCL,xyz:MSTR,xyz:CRWV,xyz:MU,xyz:SNDK,xyz:MRVL,xyz:AMD,xyz:PLTR,"
    "xyz:COIN,xyz:AAPL,xyz:TSM,xyz:ARM,xyz:AVGO"
)
SENTINEL_COIN = "cash_NVDA"


def iex_published(day: str) -> bool:
    try:
        with urllib.request.urlopen(f"https://iextrading.com/api/1.0/hist?date={day}", timeout=15) as response:
            return response.status == 200 and response.read(10) != b""
    except urllib.error.HTTPError:
        return False
    except Exception:
        return False


def trading_days(start: str, end: str) -> list[str]:
    current = date(int(start[:4]), int(start[4:6]), int(start[6:]))
    last = date(int(end[:4]), int(end[4:6]), int(end[6:]))
    days = []
    while current <= last:
        if current.weekday() < 5:
            days.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return days


def run(day: str) -> None:
    iex_dir = ROOT / "out" / "rl_research" / "hl_archive" / "iex" / day
    hl_file = ROOT / "out" / "rl_research" / "hl_archive" / "trades" / day / f"{SENTINEL_COIN}.csv"
    if not iex_dir.exists():
        if not iex_published(day):
            print(f"{day}: IEX not published (holiday/weekend), skip", flush=True)
            return
        print(f"=== IEX {day}", flush=True)
        subprocess.run(
            [sys.executable, "rl_research/iex_daily_extract.py", "--date", day, "--output-dir", str(iex_dir)],
            cwd=ROOT, check=False,
        )
    if not hl_file.exists():
        print(f"=== HL {day}", flush=True)
        subprocess.run(
            [sys.executable, "rl_research/backfill_hl_trades.py", "--date", day, "--hours", "11-22", "--coins", HL_COINS],
            cwd=ROOT, check=False,
        )
        raw = ROOT / "out" / "rl_research" / "hl_archive" / day
        if raw.exists():
            subprocess.run(["rm", "-rf", str(raw)], check=False)


def main() -> None:
    start, end = sys.argv[1], sys.argv[2]
    days = trading_days(start, end)
    started = time.time()
    for index, day in enumerate(days, 1):
        run(day)
        print(f"--- {index}/{len(days)} done, {(time.time() - started) / 60:.0f} min elapsed", flush=True)
    print("range fetch complete", flush=True)


if __name__ == "__main__":
    main()
