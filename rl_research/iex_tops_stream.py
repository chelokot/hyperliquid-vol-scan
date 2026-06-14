from __future__ import annotations

import argparse
import csv
import json
import struct
import subprocess
import sys
import urllib.request

ENHANCED_PACKET_BLOCK = 0x00000006
TRADE_REPORT = 0x54


def hist_link(date: str, feed: str) -> str:
    with urllib.request.urlopen(f"https://iextrading.com/api/1.0/hist?date={date}") as response:
        files = json.load(response)
    for entry in files:
        if entry["feed"] == feed:
            return entry["link"]
    raise RuntimeError(f"feed {feed} not found for {date}")


class StreamReader:
    def __init__(self, raw) -> None:
        self.raw = raw
        self.buffer = b""

    def read(self, count: int) -> bytes:
        while len(self.buffer) < count:
            chunk = self.raw.read(1 << 20)
            if not chunk:
                return b""
            self.buffer += chunk
        result, self.buffer = self.buffer[:count], self.buffer[count:]
        return result


def iter_trade_reports(reader: StreamReader, wanted_symbols: set[bytes]):
    while True:
        block_header = reader.read(8)
        if len(block_header) < 8:
            return
        block_type, block_length = struct.unpack("<II", block_header)
        body = reader.read(block_length - 8)
        if len(body) < block_length - 8:
            return
        if block_type != ENHANCED_PACKET_BLOCK:
            continue
        captured_length = struct.unpack("<I", body[12:16])[0]
        packet = body[20 : 20 + captured_length]
        ethertype = struct.unpack(">H", packet[12:14])[0]
        offset = 18 if ethertype == 0x8100 else 14
        offset += 20 + 8
        payload = packet[offset + 40 :]
        cursor = 0
        while cursor + 2 <= len(payload):
            message_length = struct.unpack("<H", payload[cursor : cursor + 2])[0]
            message = payload[cursor + 2 : cursor + 2 + message_length]
            cursor += 2 + message_length
            if len(message) < 38 or message[0] != TRADE_REPORT:
                continue
            symbol = message[10:18]
            if symbol not in wanted_symbols:
                continue
            timestamp_ns = struct.unpack("<q", message[2:10])[0]
            size = struct.unpack("<I", message[18:22])[0]
            price = struct.unpack("<q", message[22:30])[0] / 1e4
            yield symbol, timestamp_ns // 1_000_000, price, size


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
    gunzip = subprocess.Popen(["gunzip", "-c"], stdin=download.stdout, stdout=subprocess.PIPE)
    reader = StreamReader(gunzip.stdout)
    symbols = [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
    wanted = {symbol.encode().ljust(8) for symbol in symbols}
    from pathlib import Path

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    handles = []
    for symbol in symbols:
        handle = (out_dir / f"iex_{symbol}_{args.date}.csv").open("w", newline="")
        handles.append(handle)
        writer = csv.writer(handle)
        writer.writerow(["time_ms", "px", "sz"])
        writers[symbol.encode().ljust(8)] = writer
    counts = {key: 0 for key in wanted}
    total = 0
    for symbol_key, time_ms, price, size in iter_trade_reports(reader, wanted):
        writers[symbol_key].writerow([time_ms, price, size])
        counts[symbol_key] += 1
        total += 1
        if total % 20000 == 0:
            print(f"{total} trades extracted", flush=True)
    for handle in handles:
        handle.close()
    for symbol_key, count in counts.items():
        print(f"{symbol_key.decode().strip()}: {count} trades")
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
