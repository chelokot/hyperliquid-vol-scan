from __future__ import annotations

import argparse
import csv
import queue
import re
import struct
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from iex_tops_stream import hist_link

TRADE_REPORT = 0x54
SYMBOL_OFFSET = 10
RANGE_SIZE = 1 << 27
FETCH_THREADS = 12
SCAN_WORKERS = 4
SCAN_CHUNK = 1 << 25
OVERLAP = 64

WATCHLIST = (
    "SPCX,SKHX,SNDK,MU,INTC,MRVL,CRCL,DRAM,MSTR,CRWV,CBRS,NVDA,TSLA,META,GOOGL,AMZN,MSFT,HOOD,EWY,"
    "SPY,QQQ,GLD,SLV,USO,BNO,UNG,AAPL,AMD,AVGO,COIN,PLTR,SMCI,ARM,TSM,QCOM,IBIT"
)


def content_length(link: str) -> int:
    request = urllib.request.Request(link, method="HEAD")
    with urllib.request.urlopen(request) as response:
        return int(response.headers["Content-Length"])


def parallel_fetch(link: str, total: int, sink) -> None:
    ranges = [(start, min(start + RANGE_SIZE, total) - 1) for start in range(0, total, RANGE_SIZE)]
    results: dict[int, bytes] = {}
    lock = threading.Lock()
    ready = threading.Condition(lock)
    next_index = 0

    def fetch(index: int, start: int, end: int) -> None:
        nonlocal next_index
        data = subprocess.run(
            ["curl", "-s", "--retry", "3", "--range", f"{start}-{end}", link],
            capture_output=True,
        ).stdout
        with ready:
            results[index] = data
            ready.notify_all()

    def feeder() -> None:
        nonlocal next_index
        with ready:
            while next_index < len(ranges):
                while next_index not in results:
                    ready.wait()
                data = results.pop(next_index)
                next_index += 1
                ready.release()
                try:
                    sink.write(data)
                finally:
                    ready.acquire()
        sink.close()

    feeder_thread = threading.Thread(target=feeder)
    feeder_thread.start()
    active: list[threading.Thread] = []
    for index, (start, end) in enumerate(ranges):
        while True:
            with lock:
                pending = len(results) + sum(1 for t in active if t.is_alive())
            if pending < FETCH_THREADS + 2:
                break
            time.sleep(0.2)
        thread = threading.Thread(target=fetch, args=(index, start, end))
        thread.start()
        active.append(thread)
    for thread in active:
        thread.join()
    feeder_thread.join()


def scan_chunk(args: tuple[bytes, int, int]) -> list[tuple[bytes, int, float, int, int]]:
    data, day_start_ns, day_end_ns = args
    pattern = SCAN_PATTERN
    found = []
    for match in pattern.finditer(data):
        hit = match.start()
        message_start = hit - SYMBOL_OFFSET
        if message_start < 0 or message_start + 38 > len(data):
            continue
        if data[message_start] != TRADE_REPORT:
            continue
        timestamp_ns = struct.unpack("<q", data[message_start + 2 : message_start + 10])[0]
        if not day_start_ns <= timestamp_ns <= day_end_ns:
            continue
        size = struct.unpack("<I", data[message_start + 18 : message_start + 22])[0]
        price_raw = struct.unpack("<q", data[message_start + 22 : message_start + 30])[0]
        if size == 0 or not 0 < price_raw < 10_000_000_000:
            continue
        trade_id = struct.unpack("<q", data[message_start + 30 : message_start + 38])[0]
        found.append((match.group(), timestamp_ns // 1_000_000, price_raw / 1e4, size, trade_id))
    return found


SCAN_PATTERN: re.Pattern | None = None


def worker_init(symbols: list[bytes]) -> None:
    global SCAN_PATTERN
    SCAN_PATTERN = re.compile(b"|".join(re.escape(symbol) for symbol in symbols))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--symbols", default=WATCHLIST)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    started = time.time()
    link = hist_link(args.date, "TOPS")
    total = content_length(link)
    print(f"file size: {total / 1e9:.1f} GB compressed", flush=True)
    day = time.strptime(args.date, "%Y%m%d")
    day_start_ns = int((time.mktime(day) - 86400) * 1e9)
    day_end_ns = day_start_ns + 3 * 86400 * int(1e9)
    symbols = [symbol.strip().upper().encode().ljust(8) for symbol in args.symbols.split(",") if symbol.strip()]

    decompress = subprocess.Popen(["pigz", "-dc"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    fetcher = threading.Thread(target=parallel_fetch, args=(link, total, decompress.stdin))
    fetcher.start()

    trades: dict[bytes, dict[int, tuple]] = {symbol: {} for symbol in symbols}
    scanned = 0
    with ProcessPoolExecutor(max_workers=SCAN_WORKERS, initializer=worker_init, initargs=(symbols,)) as pool:
        pending = []
        tail = b""
        while True:
            chunk = decompress.stdout.read(SCAN_CHUNK)
            if not chunk:
                break
            data = tail + chunk
            tail = data[-OVERLAP:]
            scanned += len(chunk)
            pending.append(pool.submit(scan_chunk, (data, day_start_ns, day_end_ns)))
            if len(pending) >= SCAN_WORKERS * 3:
                for row in pending.pop(0).result():
                    trades[row[0]][row[4]] = row[1:4]
            if scanned % (10 << 30) < SCAN_CHUNK:
                print(f"scanned {scanned / 1e9:.0f} GB raw, {time.time() - started:.0f}s elapsed", flush=True)
        for future in pending:
            for row in future.result():
                trades[row[0]][row[4]] = row[1:4]
    fetcher.join()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for symbol_key, by_tid in trades.items():
        symbol = symbol_key.decode().strip()
        rows = sorted((time_ms, price, size, tid) for tid, (time_ms, price, size) in by_tid.items())
        path = out_dir / f"iex_{symbol}_{args.date}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_ms", "px", "sz", "tid"])
            writer.writerows(rows)
        if rows:
            print(f"{symbol}: {len(rows)} trades")
    print(f"total {scanned / 1e9:.1f} GB raw in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
