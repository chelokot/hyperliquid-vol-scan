from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("GTK4Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

PROGRESS_PATH = Path(__file__).resolve().parents[1] / "out" / "rl_research" / "rl_train_progress.jsonl"

figure, axis = plt.subplots(figsize=(11, 6))
figure.canvas.manager.set_window_title("RL training — SPCX lead-lag")


def refresh(_frame: int) -> None:
    if not PROGRESS_PATH.exists():
        return
    rows = [json.loads(line) for line in PROGRESS_PATH.open()]
    if not rows:
        return
    times = [datetime.fromtimestamp(row["time_ms"] / 1000) for row in rows]
    axis.clear()
    axis.plot(times, [row["best_train_pnl"] for row in rows], color="#1f77b4", label="best train pnl")
    axis.plot(times, [row["val_of_best_train"] for row in rows], color="#ff7f0e", label="val pnl того же best-train")
    axis.plot(times, [row["best_val_pnl"] for row in rows], color="#2ca02c", linestyle="--", label="best val pnl")
    axis.axhline(0, color="gray", linewidth=0.8)
    last = rows[-1]
    axis.set_title(
        f"gen {last['generation']} | train {last['train_ticks']} ticks / val {last['val_ticks']} ticks | "
        f"best train {last['best_train_pnl']:+.2f}% | best val {last['best_val_pnl']:+.2f}%"
    )
    axis.set_xlabel("время")
    axis.set_ylabel("pnl, % on notional")
    axis.legend(loc="upper left")
    axis.grid(alpha=0.3)
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))


animation = FuncAnimation(figure, refresh, interval=2000, cache_frame_data=False)
plt.show()
