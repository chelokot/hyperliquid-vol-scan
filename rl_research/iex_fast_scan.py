from __future__ import annotations

import argparse
import csv
import re
import struct
import subprocess
import time
from pathlib import Path

from iex_tops_stream import hist_link

TRADE_REPORT = 0x54
SYMBOL_OFFSET_IN_MESSAGE = 10
CHUNK_SIZE = 1 << 25
OVERLAP = 64


def scan_stream(raw, wanted: dict[bytes, list], day_start_ns: int, day_end_ns: int) -> int:
    pattern = re.compile(b"|".join(re.escape(symbol) for symbol in wanted))
    scanned = 0
    started = time.time()
    tail = b""
    while True:
        chunk = raw.read(CHUNK_SIZE)
        if not chunk:
            break
        data = tail + chunk
        for match in pattern.finditer(data):
            hit = match.start()
            message_start = hit - SYMBOL_OFFSET_IN_MESSAGE
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
            wanted[match.group()].append((timestamp_ns // 1_000_000, price_raw / 1e4, size, trade_id))
        tail = data[-OVERLAP:]
        scanned += len(chunk)
        if scanned % (5 << 30) < CHUNK_SIZE:
            rate = scanned / (time.time() - started) / 1e6
            print(f"scanned {scanned / 1e9:.1f} GB raw ({rate:.0f} MB/s)", flush=True)
    return scanned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-bytes", type=int, default=0)
    args = parser.parse_args()
    link = hist_link(args.date, "TOPS")
    curl = ["curl", "-s", link]
    if args.max_bytes:
        curl += ["--range", f"0-{args.max_bytes}"]
    download = subprocess.Popen(curl, stdout=subprocess.PIPE)
    decompress = subprocess.Popen(["pigz", "-dc"], stdin=download.stdout, stdout=subprocess.PIPE)
    day = time.strptime(args.date, "%Y%m%d")
    day_start_ns = int((time.mktime(day) - 86400) * 1e9)
    day_end_ns = day_start_ns + 3 * 86400 * int(1e9)
    symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
    wanted: dict[bytes, list] = {symbol.encode().ljust(8): [] for symbol in symbols}
    started = time.time()
    scanned = scan_stream(decompress.stdout, wanted, day_start_ns, day_end_ns)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for symbol_key, rows in wanted.items():
        symbol = symbol_key.decode().strip()
        unique = sorted({row[3]: row for row in rows}.values())
        path = out_dir / f"iex_{symbol}_{args.date}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_ms", "px", "sz", "tid"])
            writer.writerows(unique)
        print(f"{symbol}: {len(unique)} trades -> {path}")
    print(f"total: {scanned / 1e9:.1f} GB raw in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
