"""Did the sample contain a crash, and was the strategy robust to it?
Per day: market move (basket of cash majors), HOOD/INTC intraday drawdown, the
strategy's per-notional PnL on INTC+HOOD, and the WORST single-trade adverse
excursion that day (closeness to the 2.5% liquidation line at 20x)."""

from __future__ import annotations

import os
import numpy as np
import torch

from features_v2 import WARMUP, build_episode
from nn_train import DEVICE, QuantileMLP, TorchQuantiles
from xgb2_train import DELAY, step_position

TRADE = ["cash:INTC", "cash:HOOD"]
MARKET = ["cash:NVDA", "cash:TSLA", "cash:META", "cash:GOOGL", "cash:AMZN", "cash:MSFT", "cash:HOOD"]


def session_ret(perp):
    p = perp[WARMUP:]
    return p[-1] / p[0] - 1


def max_drawdown(perp):
    p = perp[WARMUP:]
    peak = np.maximum.accumulate(p)
    return float((p / peak - 1).min())


def ensemble_dir(members, episode, enter_bps):
    feats = episode["features"][WARMUP:]
    per = []
    for m in members:
        q = m.inplace_predict(feats)
        q25, q50, q75 = q[:, 1], q[:, 2], q[:, 3]
        pos = np.zeros(len(q50)); cur = 0.0
        for t in range(len(q50)):
            cur = step_position(cur, q25[t], q50[t], q75[t], enter_bps); pos[t] = cur
        per.append(pos)
    avg = np.mean(per, axis=0)
    d = np.where(avg >= 0.5, 1.0, np.where(avg <= -0.5, -1.0, 0.0))
    return np.concatenate([np.zeros(DELAY), d[:-DELAY]])


def trade_stats(direction, perp, cost):
    pnl, i, n = 0.0, 1, len(perp)
    worst_adverse = 0.0
    while i < n:
        d = direction[i]
        if d == 0:
            i += 1; continue
        entry = perp[i]; j = i
        while j + 1 < n and direction[j + 1] == d:
            j += 1
        path = perp[i : j + 1]
        adverse = (path.min() / entry - 1) if d > 0 else -(path.max() / entry - 1)
        worst_adverse = min(worst_adverse, adverse)
        pnl += d * (perp[j] / entry - 1) - 2 * cost
        i = j + 1
    return pnl * 100, worst_adverse * 100


def main() -> None:
    bundle = torch.load("out/rl_research/production_ensemble.pt", weights_only=False)
    members = []
    for st in bundle["states"]:
        m = QuantileMLP(bundle["n_features"]); m.load_state_dict(st); m.to(DEVICE).eval()
        members.append(TorchQuantiles(m, bundle["mean"], bundle["std"], bundle["y_mean"], bundle["y_std"]))
    enter = {p: bundle["enter_scale"] * bundle["costs_bps"][p] for p in set(TRADE)}
    cost = {p: bundle["costs_bps"][p] / 1e4 for p in set(TRADE)}

    rows = []
    for d in sorted(os.listdir("out/rl_research/hl_archive/trades")):
        mkt = []
        for p in MARKET:
            e = build_episode(p, d)
            if e is not None:
                mkt.append(session_ret(e["perp"]))
        if not mkt:
            continue
        market = float(np.mean(mkt)) * 100
        strat_pnls, adverses, dds = [], [], []
        for p in TRADE:
            e = build_episode(p, d)
            if e is None:
                continue
            perp = e["perp"][WARMUP:]
            dd = max_drawdown(e["perp"]) * 100
            direction = ensemble_dir(members, e, enter[p])
            pnl, wa = trade_stats(direction, perp, cost[p])
            strat_pnls.append(pnl); adverses.append(wa); dds.append(dd)
        if not strat_pnls:
            continue
        rows.append((d, market, np.mean(dds), np.mean(strat_pnls), min(adverses)))

    rows.sort(key=lambda r: r[1])  # worst market day first
    print(f"{'день':>9} {'рынок%':>7} {'avg DD%':>8} {'страт PnL%':>11} {'худший адверс% (сделка)':>22}")
    print("--- 12 ХУДШИХ рыночных дней ---")
    for r in rows[:12]:
        print(f"{r[0]:>9} {r[1]:>+7.2f} {r[2]:>+8.2f} {r[3]:>+11.3f} {r[4]:>+22.2f}")
    print("--- 3 ЛУЧШИХ рыночных дня ---")
    for r in rows[-3:]:
        print(f"{r[0]:>9} {r[1]:>+7.2f} {r[2]:>+8.2f} {r[3]:>+11.3f} {r[4]:>+22.2f}")
    worst_adverse_all = min(r[4] for r in rows)
    print(f"\nсамый глубокий адверс-экскурс по ЛЮБОЙ сделке за все дни: {worst_adverse_all:+.2f}%  (ликвидация при -2.50%)")


if __name__ == "__main__":
    main()
