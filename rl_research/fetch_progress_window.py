from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("GTK4Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

LOG_PATH = Path(__file__).resolve().parents[1] / "out" / "rl_research" / "fetch_range.log"
DONE_RE = re.compile(r"--- (\d+)/(\d+) done, (\d+) min elapsed")
DAY_RE = re.compile(r"=== (IEX|HL) (\d{8})")
GB_RE = re.compile(r"scanned (\d+) GB raw, (\d+)s elapsed")

figure, axis = plt.subplots(figsize=(11, 3.2))
figure.canvas.manager.set_window_title("Качка истории — прогресс")


def refresh(_frame: int) -> None:
    axis.clear()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    if not LOG_PATH.exists():
        axis.text(0.5, 0.5, "лог ещё не создан...", ha="center", va="center", fontsize=14, color="gray")
        return
    text = LOG_PATH.read_text()
    done_matches = DONE_RE.findall(text)
    day_matches = DAY_RE.findall(text)
    gb_matches = GB_RE.findall(text)

    done, total, elapsed_min = (0, 50, 0)
    if done_matches:
        done, total, elapsed_min = (int(done_matches[-1][0]), int(done_matches[-1][1]), int(done_matches[-1][2]))
    fraction = done / total if total else 0.0
    rate = done / elapsed_min if elapsed_min and done else 0.0
    eta_min = (total - done) / rate if rate else 0.0
    current = f"{day_matches[-1][0]} {day_matches[-1][1]}" if day_matches else "—"
    current_gb = f" · {gb_matches[-1][0]}GB" if gb_matches and not done_matches or (gb_matches and day_matches and day_matches[-1][0] == "IEX") else ""

    axis.barh(0.62, 1.0, height=0.18, color="#e9e9e9")
    axis.barh(0.62, fraction, height=0.18, color="#2ca02c")
    axis.text(0.5, 0.62, f"{done}/{total} дней ({fraction * 100:.0f}%)", ha="center", va="center", fontsize=12, weight="bold")
    eta_text = f"ETA ~{int(eta_min // 60)}ч {int(eta_min % 60)}мин" if eta_min >= 60 else f"ETA ~{int(eta_min)}мин"
    if done >= total:
        eta_text = "ГОТОВО ✓"
    axis.text(0.0, 0.30, f"прошло {int(elapsed_min // 60)}ч {int(elapsed_min % 60)}мин", ha="left", va="center", fontsize=12)
    axis.text(1.0, 0.30, eta_text, ha="right", va="center", fontsize=12, weight="bold", color="#1f6f1f")
    axis.text(0.5, 0.06, f"сейчас: {current}{current_gb} · {rate * 60:.1f} дней/час", ha="center", va="center", fontsize=10, color="#555")


animation = FuncAnimation(figure, refresh, interval=3000, cache_frame_data=False)
plt.show()
