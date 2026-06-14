from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("GTK4Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import requests
from matplotlib.animation import FuncAnimation

from rl_policy_train import load_ticks

DATA_PATH = Path(__file__).resolve().parents[1] / "out" / "rl_research" / "spcx_leadlag.jsonl"
ACCOUNT = os.environ.get("HL_ACCOUNT_ADDRESS", "")

figure, (price_axis, premium_axis) = plt.subplots(
    2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
)
figure.canvas.manager.set_window_title("SPCX: Nasdaq vs Hyperliquid (live)")


def load_fills(start_ms: int) -> tuple[list, list]:
    try:
        fills = requests.post(
            "https://api.hyperliquid.xyz/info",
            json={"type": "userFills", "user": ACCOUNT},
            timeout=5,
        ).json()
    except (requests.RequestException, ValueError):
        return [], []
    if not isinstance(fills, list):
        return [], []
    buys, sells = [], []
    for fill in fills:
        if fill["coin"] != "xyz:SPCX" or fill["time"] < start_ms or float(fill["sz"]) < 0.5:
            continue
        point = (datetime.fromtimestamp(fill["time"] / 1000), float(fill["px"]))
        (buys if fill["side"] == "B" else sells).append(point)
    return buys, sells


def refresh(_frame: int) -> None:
    ticks = load_ticks(DATA_PATH)
    if not ticks:
        return
    times = [datetime.fromtimestamp(tick.time_ms / 1000) for tick in ticks]
    stock = [tick.stock for tick in ticks]
    mid = [(tick.bid + tick.ask) / 2 for tick in ticks]
    premium = [(s / m - 1) * 100 for s, m in zip(stock, mid)]
    buys, sells = load_fills(ticks[0].time_ms)

    price_axis.clear()
    price_axis.plot(times, mid, color="#e4572e", linewidth=1.1, label="Hyperliquid perp (mid)")
    price_axis.plot(times, stock, color="#17bebb", linewidth=1.1, label="Nasdaq SPCX (Yahoo)")
    if buys:
        price_axis.scatter(*zip(*buys), marker="^", color="#2ecc71", s=80, zorder=5, label="твои покупки")
    if sells:
        price_axis.scatter(*zip(*sells), marker="v", color="#e74c3c", s=80, zorder=5, label="твои продажи")
    price_axis.set_ylabel("цена, $")
    price_axis.legend(loc="upper left")
    price_axis.grid(alpha=0.3)
    price_axis.set_title(
        f"SPCX live | тиков: {len(ticks)} | премия сейчас: {premium[-1]:+.3f}% | "
        f"акция {stock[-1]:.2f} / перп {mid[-1]:.2f}"
    )

    premium_axis.clear()
    premium_axis.plot(times, premium, color="#5b5f97", linewidth=0.9)
    premium_axis.axhline(0, color="gray", linewidth=0.8)
    for level in (0.3, -0.3):
        premium_axis.axhline(level, color="orange", linestyle="--", linewidth=0.8)
    premium_axis.set_ylabel("премия акции к перпу, %")
    premium_axis.set_xlabel("время (Киев)")
    premium_axis.grid(alpha=0.3)
    premium_axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))


animation = FuncAnimation(figure, refresh, interval=3000, cache_frame_data=False)
plt.show()
