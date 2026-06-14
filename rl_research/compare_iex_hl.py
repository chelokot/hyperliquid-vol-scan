from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parents[1] / "out" / "rl_research"
ARCHIVE_DIR = OUT_DIR / "hl_archive"


def load_trades(path: Path) -> list[tuple[int, float]]:
    with path.open() as handle:
        return [(int(row["time_ms"]), float(row["px"])) for row in csv.DictReader(handle)]


def per_second_series(trades: list[tuple[int, float]]) -> dict[int, float]:
    series: dict[int, float] = {}
    for time_ms, price in trades:
        series[time_ms // 1000] = price
    return series


def build_chart(symbol: str, hl_coin: str, date: str) -> Path:
    iex_trades = load_trades(ARCHIVE_DIR / "iex" / date / f"iex_{symbol}_{date}.csv")
    hl_trades = load_trades(ARCHIVE_DIR / "trades" / date / f"{hl_coin.replace(':', '_')}.csv")
    iex_series = per_second_series(iex_trades)
    hl_series = per_second_series(hl_trades)
    common = sorted(set(iex_series) & set(hl_series))
    premium_times = [datetime.fromtimestamp(second) for second in common]
    premium = [(iex_series[second] / hl_series[second] - 1) * 100 for second in common]

    figure, (price_axis, premium_axis) = plt.subplots(
        2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    price_axis.plot(
        [datetime.fromtimestamp(t // 1000) for t, _ in hl_trades],
        [price for _, price in hl_trades],
        color="#e4572e", linewidth=0.7, label=f"Hyperliquid {hl_coin}",
    )
    price_axis.plot(
        [datetime.fromtimestamp(t // 1000) for t, _ in iex_trades],
        [price for _, price in iex_trades],
        color="#17bebb", linewidth=0.7, label=f"IEX {symbol}",
    )
    price_axis.set_ylabel("цена, $")
    price_axis.legend(loc="upper left")
    price_axis.grid(alpha=0.3)
    price_axis.set_title(
        f"{symbol} {date}: IEX {len(iex_trades)} сделок vs HL {len(hl_trades)} сделок | "
        f"общих секунд: {len(common)}"
    )
    premium_axis.plot(premium_times, premium, color="#5b5f97", linewidth=0.6)
    premium_axis.axhline(0, color="gray", linewidth=0.8)
    premium_axis.set_ylabel("премия IEX к HL, %")
    premium_axis.set_xlabel("время (Киев)")
    premium_axis.grid(alpha=0.3)
    premium_axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.tight_layout()
    output = OUT_DIR / f"compare_{symbol}_{date}.png"
    plt.savefig(output, dpi=110)
    plt.close(figure)
    print(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--pairs", required=True)
    args = parser.parse_args()
    for pair in args.pairs.split(","):
        symbol, hl_coin = pair.split("=")
        build_chart(symbol.strip(), hl_coin.strip(), args.date)


if __name__ == "__main__":
    main()
