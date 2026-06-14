from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("GTK4Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

PROGRESS_PATH = Path(__file__).resolve().parents[1] / "out" / "rl_research" / "rl_ppo_progress.jsonl"
EMA_ALPHA = 0.05

figure, axis = plt.subplots(figsize=(11, 6))
figure.canvas.manager.set_window_title("PPO training — pairs")


def refresh(_frame: int) -> None:
    rows = [json.loads(line) for line in PROGRESS_PATH.open()] if PROGRESS_PATH.exists() else []
    if not rows:
        axis.clear()
        axis.text(0.5, 0.5, "ждём первые апдейты...", ha="center", va="center", fontsize=16, color="gray")
        axis.set_xticks([])
        axis.set_yticks([])
        return
    updates = [row["update"] for row in rows]
    train_raw = [row["train_sampled_pnl"] for row in rows]
    ema = []
    value = 0.0
    weight = 0.0
    for sample in train_raw:
        value = EMA_ALPHA * sample + (1 - EMA_ALPHA) * value
        weight = EMA_ALPHA + (1 - EMA_ALPHA) * weight
        ema.append(value / weight)
    val_points = [(row["update"], row["val_pnl"]) for row in rows if "val_pnl" in row]
    axis.clear()
    axis.plot(updates, train_raw, color="#1f77b4", alpha=0.25, linewidth=0.8, label="train pnl (сэмплированная)")
    axis.plot(updates, ema, color="#1f77b4", linewidth=1.8, label="train pnl EMA")
    if val_points:
        axis.plot(*zip(*val_points), color="#2ca02c", marker="o", markersize=3, linewidth=1.2, label="val pnl (argmax)")
    axis.axhline(0, color="gray", linewidth=0.8)
    recent = train_raw[-max(len(train_raw) // 2, 100) :] + [p for _, p in val_points[-100:]] + [0.0]
    low = float(np.percentile(recent, 5))
    high = float(np.percentile(recent, 95))
    pad = max((high - low) * 0.25, 0.5)
    axis.set_ylim(low - pad, high + pad)
    last = rows[-1]
    best_val = max((row["val_pnl"] for row in rows if "val_pnl" in row), default=float("nan"))
    axis.set_title(
        f"update {last['update']} | train EMA {ema[-1]:+.2f}% | best val {best_val:+.2f}% | "
        f"entropy {last.get('entropy', 0):.3f} | activity {last.get('val_activity', float('nan'))}"
    )
    axis.set_xlabel("PPO update")
    axis.set_ylabel("pnl, %/эпизод (пара-день)")
    axis.legend(loc="upper left")
    axis.grid(alpha=0.3)


animation = FuncAnimation(figure, refresh, interval=3000, cache_frame_data=False)
plt.show()
