from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("GTK4Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

LOG_PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "out/rl_research/rl_pairs_train_10d.log")
LINE_PATTERN = re.compile(
    r"gen (\d+): train_best ([+-][\d.]+)%/pair, best_val ([+-][\d.]+)%/pair, (\d+)s"
)

figure, axis = plt.subplots(figsize=(11, 6))
figure.canvas.manager.set_window_title(f"RL pairs training — {LOG_PATH.name}")


def refresh(_frame: int) -> None:
    if not LOG_PATH.exists():
        return
    generations, train_best, best_val = [], [], []
    elapsed = 0
    for line in LOG_PATH.open():
        match = LINE_PATTERN.search(line)
        if match:
            generations.append(int(match.group(1)))
            train_best.append(float(match.group(2)))
            best_val.append(float(match.group(3)))
            elapsed = int(match.group(4))
    if not generations:
        return
    axis.clear()
    axis.plot(generations, train_best, color="#1f77b4", marker="o", label="best train pnl %/эпизод")
    axis.plot(generations, best_val, color="#2ca02c", marker="o", linestyle="--", label="best val pnl %/эпизод")
    axis.axhline(0, color="gray", linewidth=0.8)
    axis.set_title(
        f"gen {generations[-1]} | train {train_best[-1]:+.2f}% | val {best_val[-1]:+.2f}% | {elapsed}s"
    )
    axis.set_xlabel("поколение")
    axis.set_ylabel("pnl, %/эпизод (пара-день)")
    axis.legend(loc="upper left")
    axis.grid(alpha=0.3)


animation = FuncAnimation(figure, refresh, interval=3000, cache_frame_data=False)
plt.show()
