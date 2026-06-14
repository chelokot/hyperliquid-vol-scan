"""Measure a maker-execution variant on the existing model's outputs (no retrain).

Taker (baseline): enter by crossing the spread, pay spread/2 + fee + slippage.
Maker: post a limit half-a-spread away; fill only if a print trades through it
within W seconds (else miss the entry), earning the spread instead of paying it.
TP exits are maker either way; SL exits are taker.
"""

from __future__ import annotations

import json
import os

import numpy as np
import xgboost as xgb

from features_v2 import WARMUP, build_episode, effective_spread_bps
from xgb2_train import TAKER_FEE_BPS, EXTRA_BPS, quantile_positions

MAKER_FEE = 0.00003
TAKER_FEE = TAKER_FEE_BPS / 1e4
FILL_WINDOW = 30  # seconds a resting limit waits before we give up on the entry


def desired_positions(model, episode, enter_bps: float) -> np.ndarray:
    preds = model.inplace_predict(episode["features"][WARMUP:])
    return quantile_positions(preds, enter_bps)


def taker_pnl(episode, positions, half_spread, tp=0.005, sl=0.005, time_stop=120) -> float:
    perp = episode["perp"][WARMUP:]
    cost = half_spread + TAKER_FEE + EXTRA_BPS / 1e4
    pnl, i = 0.0, 1
    while i < len(positions) - 1:
        side = positions[i]
        if side == 0:
            i += 1
            continue
        entry = perp[i] * (1 + side * (half_spread))  # cross the spread
        take = entry * (1 + side * tp)
        stop = entry * (1 - side * sl)
        j = i + 1
        exit_px, fee = None, MAKER_FEE
        while j < len(perp):
            if side * (perp[j] - stop) <= 0:
                exit_px, fee = stop * (1 - side * half_spread), TAKER_FEE
                break
            if side * (perp[j] - take) >= 0:
                exit_px, fee = take, MAKER_FEE
                break
            if j - i >= time_stop:
                exit_px, fee = perp[j] * (1 - side * half_spread), TAKER_FEE
                break
            j += 1
        if exit_px is None:
            exit_px, fee, j = perp[-1] * (1 - side * half_spread), TAKER_FEE, len(perp) - 1
        pnl += side * (exit_px - entry) / entry - TAKER_FEE - fee - EXTRA_BPS / 1e4
        i = j + 1
    return pnl * 100


def maker_pnl(episode, positions, half_spread, tp=0.005, sl=0.005, time_stop=120) -> tuple[float, int, int]:
    perp = episode["perp"][WARMUP:]
    pnl, i, fills, misses = 0.0, 1, 0, 0
    while i < len(positions) - 1:
        side = positions[i]
        if side == 0:
            i += 1
            continue
        # post a limit half-a-spread on our side; fill if price trades through it
        limit = perp[i] * (1 - side * half_spread)
        fill_at = None
        for k in range(1, FILL_WINDOW + 1):
            if i + k >= len(perp):
                break
            if side * (perp[i + k] - limit) <= 0:
                fill_at = i + k
                break
        if fill_at is None:
            misses += 1
            i += FILL_WINDOW  # missed the entry, move on
            continue
        fills += 1
        entry = limit
        take = entry * (1 + side * tp)
        stop = entry * (1 - side * sl)
        j = fill_at + 1
        exit_px, fee = None, MAKER_FEE
        while j < len(perp):
            if side * (perp[j] - stop) <= 0:
                exit_px, fee = stop * (1 - side * half_spread), TAKER_FEE
                break
            if side * (perp[j] - take) >= 0:
                exit_px, fee = take, MAKER_FEE
                break
            if j - fill_at >= time_stop:
                exit_px, fee = perp[j] * (1 - side * half_spread), TAKER_FEE
                break
            j += 1
        if exit_px is None:
            exit_px, fee, j = perp[-1] * (1 - side * half_spread), TAKER_FEE, len(perp) - 1
        pnl += side * (exit_px - entry) / entry - MAKER_FEE - fee
        i = j + 1
    return pnl * 100, fills, misses


def main() -> None:
    model = xgb.Booster()
    model.load_model("out/rl_research/xgb2_quantile_model.json")
    cfg = json.load(open("out/rl_research/xgb2_config.json"))
    scale = cfg["enter_scale"]
    dates = sorted(os.listdir("out/rl_research/hl_archive/trades"))
    test = dates[-12:]
    train = dates[:-19]
    pairs = ["HOOD", "NVDA", "INTC", "MSTR", "CRWV", "TSM"]
    print(f"{'пара':>6} {'спред bps':>9} {'taker/день':>11} {'maker/день':>11} {'fills':>6} {'misses':>6}")
    tk_tot, mk_tot = [], []
    for s in pairs:
        spread = effective_spread_bps(s, train) / 1e4
        half = spread / 2
        enter = scale * (TAKER_FEE_BPS + spread / 2 * 1e4 + EXTRA_BPS) / 1e4 * 1e4
        tks, mks, F, M = [], [], 0, 0
        for d in test:
            e = build_episode(s, d)
            if e is None:
                continue
            pos = desired_positions(model, e, enter)
            tks.append(taker_pnl(e, pos, half))
            m, f, miss = maker_pnl(e, pos, half)
            mks.append(m)
            F += f
            M += miss
        tk, mk = np.mean(tks), np.mean(mks)
        tk_tot.append(tk)
        mk_tot.append(mk)
        print(f"{s:>6} {spread * 1e4:>9.1f} {tk:>+10.3f}% {mk:>+10.3f}% {F:>6} {M:>6}")
    print(f"\nПОРТФЕЛЬ (поровну): taker {np.mean(tk_tot):+.3f}%/день | maker {np.mean(mk_tot):+.3f}%/день")


if __name__ == "__main__":
    main()
