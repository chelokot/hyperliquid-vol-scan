"""20x with REAL liquidation on the top-2 pairs (cash:INTC, cash:HOOD), using
the production ensemble. Trade-level sim: while a position is held, if the
adverse excursion reaches the liquidation threshold (2.5% on notional at 20x ->
-50% of margin) the trade is force-closed at -50% and cannot recover. Otherwise
it exits normally at 20x. Compares to the naive 'mean x20' (no-liquidation)
fantasy."""

from __future__ import annotations

import os
import numpy as np
import torch

from features_v2 import WARMUP, build_episode
from nn_train import DEVICE, QuantileMLP, TorchQuantiles
from xgb2_train import DELAY, EXTRA_BPS, TAKER_FEE_BPS, step_position

PAIRS = ["cash:INTC", "cash:HOOD"]
LEVERAGE = 20.0
LIQ_ADVERSE = 1.0 / LEVERAGE / 2.0   # 2.5% adverse notional -> 50% margin loss -> force close
ENTER_DEADBAND = 0.5                 # majority of members agree


def ensemble_positions(members, episode, enter_bps):
    feats = episode["features"][WARMUP:]
    per_member = []
    for m in members:
        q = m.inplace_predict(feats)
        q25, q50, q75 = q[:, 1], q[:, 2], q[:, 3]
        pos = np.zeros(len(q50)); cur = 0.0
        for t in range(len(q50)):
            cur = step_position(cur, q25[t], q50[t], q75[t], enter_bps); pos[t] = cur
        per_member.append(pos)
    avg = np.mean(per_member, axis=0)
    direction = np.where(avg >= ENTER_DEADBAND, 1.0, np.where(avg <= -ENTER_DEADBAND, -1.0, 0.0))
    return np.concatenate([np.zeros(DELAY), direction[:-DELAY]])


def day_equity_20x(direction, perp, cost_notional, with_liq):
    equity = 1.0
    n = len(perp)
    i = 1
    trades = liqs = 0
    while i < n:
        d = direction[i]
        if d == 0:
            i += 1
            continue
        entry = perp[i]
        j = i
        while j + 1 < n and direction[j + 1] == d:
            j += 1
        path = perp[i : j + 1]
        adverse = (path.min() / entry - 1) if d > 0 else -(path.max() / entry - 1)
        trades += 1
        if with_liq and adverse <= -LIQ_ADVERSE:
            equity *= 0.5  # liquidated: -50% of this position's margin
            liqs += 1
        else:
            notional_ret = d * (perp[j] / entry - 1) - 2 * cost_notional  # round-trip cost
            equity *= max(0.0, 1.0 + LEVERAGE * notional_ret)
        i = j + 1
    return equity - 1.0, trades, liqs


def main() -> None:
    bundle = torch.load("out/rl_research/production_ensemble.pt", weights_only=False)
    members = []
    for st in bundle["states"]:
        m = QuantileMLP(bundle["n_features"]); m.load_state_dict(st); m.to(DEVICE).eval()
        members.append(TorchQuantiles(m, bundle["mean"], bundle["std"], bundle["y_mean"], bundle["y_std"]))
    enter = {p: bundle["enter_scale"] * bundle["costs_bps"][p] for p in PAIRS}
    cost_notional = {p: bundle["costs_bps"][p] / 1e4 for p in PAIRS}

    import sys
    all_days = "--all" in sys.argv
    dates = sorted(os.listdir("out/rl_research/hl_archive/trades"))
    if not all_days:
        dates = dates[-12:]
    by_day_liq, by_day_noliq, by_day_liqcount = {}, {}, {}
    tot_trades = tot_liqs = 0
    for d in dates:
        for p in PAIRS:
            e = build_episode(p, d)
            if e is None:
                continue
            perp = e["perp"][WARMUP:]
            direction = ensemble_positions(members, e, enter[p])
            r_liq, tr, lq = day_equity_20x(direction, perp, cost_notional[p], True)
            r_no, _, _ = day_equity_20x(direction, perp, cost_notional[p], False)
            by_day_liq.setdefault(d, []).append(r_liq)
            by_day_noliq.setdefault(d, []).append(r_no)
            by_day_liqcount[d] = by_day_liqcount.get(d, 0) + lq
            tot_trades += tr; tot_liqs += lq

    days_sorted = sorted(by_day_liq)
    liq_daily = np.array([np.mean(by_day_liq[d]) for d in days_sorted]) * 100
    no_daily = np.array([np.mean(by_day_noliq[d]) for d in days_sorted]) * 100
    liq_days = sorted(by_day_liqcount, key=lambda d: -by_day_liqcount[d])
    print(f"INTC+HOOD, 20x, {len(liq_daily)} дней | trades {tot_trades}, ликвидаций {tot_liqs} ({tot_liqs/tot_trades*100:.1f}%)\n")
    print(f"дней с >=1 ликвидацией: {sum(1 for d in by_day_liqcount if by_day_liqcount[d]>0)}/{len(days_sorted)}")
    print(f"\nТОП-10 худших дней (с ликвидацией):")
    for d in sorted(days_sorted, key=lambda d: liq_daily[days_sorted.index(d)])[:10]:
        i = days_sorted.index(d)
        print(f"  {d}:  с ликв {liq_daily[i]:+8.1f}%  | наивный {no_daily[i]:+8.1f}%  | liqs {by_day_liqcount[d]}")
    print(f"\nСРЕДНЕЕ/день:  с ликвидацией {liq_daily.mean():+.1f}%  | наивный x20 {no_daily.mean():+.1f}%")
    print(f"ХУДШИЙ день:   с ликвидацией {liq_daily.min():+.1f}%  | наивный {no_daily.min():+.1f}%")
    eq = np.prod(1 + liq_daily / 100)
    print(f"\nкомпаунд за {len(liq_daily)} дней (с ликв., реинвест): x{eq:.3g}")


if __name__ == "__main__":
    main()
