"""Does USING the full quantile output help? Compare on one model (60s, 5
quantiles), real fees, subset:
  baseline  — current hysteresis, fixed size +-1 from q25/q50/q75
  sized     — same timing, but size ∝ q50/(q75-q25) (confidence) + tail gate on
              q10/q90 (skip if the against-tail is too fat vs cost)
Selection by validation Sharpe (leverage-invariant); report test mean/std/Sharpe
and average exposure so the comparison is fair, not just 'more leverage'.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from features_v2 import WARMUP, build_episode, effective_spread_bps
from nn_train import DEVICE, QuantileMLP, TorchQuantiles, pinball_loss
from xgb2_train import DELAY, EXTRA_BPS, FORWARD_SECONDS, TAKER_FEE_BPS, TRAIN_STRIDE, step_position
from target_ab import PAIRS, episode_targets


def train_avg(packed, train_dates, epochs, seeds):
    xs, ys = [], []
    for (date, _s), (x, yf, _ym) in packed.items():
        if date in train_dates:
            xs.append(x); ys.append(yf)
    train_x = np.concatenate(xs); train_y = np.concatenate(ys)
    mean, std = train_x.mean(0), train_x.std(0) + 1e-6
    y_mean, y_std = float(train_y.mean()), float(train_y.std() + 1e-6)
    tx = torch.from_numpy(((train_x - mean) / std).astype(np.float32)).to(DEVICE)
    ty = torch.from_numpy(((train_y - y_mean) / y_std).astype(np.float32)).to(DEVICE)
    boosters = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        model = QuantileMLP(train_x.shape[1]).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        n = len(tx)
        for _ in range(epochs):
            model.train()
            perm = torch.randperm(n, device=DEVICE)
            for s in range(0, n, 16384):
                idx = perm[s : s + 16384]
                opt.zero_grad(); pinball_loss(model(tx[idx]), ty[idx]).backward(); opt.step()
            sched.step()
        boosters.append(TorchQuantiles(model.eval(), mean, std, y_mean, y_std))
    return boosters


def preds_for(boosters, episode):
    return np.mean([b.inplace_predict(episode["features"][WARMUP:]) for b in boosters], axis=0)


def direction_series(q, enter_bps):
    q25, q50, q75 = q[:, 1], q[:, 2], q[:, 3]
    out = np.zeros(len(q50)); cur = 0.0
    for t in range(len(q50)):
        cur = step_position(cur, q25[t], q50[t], q75[t], enter_bps); out[t] = cur
    return out


def size_series(direction, q, conf_k, conf_cap, tail_mult, cost_bps):
    q10, q25, q50, q75, q90 = q[:, 0], q[:, 1], q[:, 2], q[:, 3], q[:, 4]
    out = np.zeros(len(direction)); cur_dir = 0.0; cur_size = 0.0
    for t in range(len(direction)):
        if direction[t] != cur_dir:
            cur_dir = direction[t]
            if cur_dir == 0:
                cur_size = 0.0
            else:
                conf = min(conf_k * abs(q50[t]) / (q75[t] - q25[t] + 1e-6), conf_cap)
                gate = (q10[t] > -tail_mult * cost_bps) if cur_dir > 0 else (q90[t] < tail_mult * cost_bps)
                cur_size = cur_dir * conf * (1.0 if gate else 0.0)
        out[t] = cur_size
    return out


def pnl(positions, perp, cost):
    positions = np.concatenate([np.zeros(DELAY), positions[:-DELAY]])
    returns = np.zeros(len(perp)); returns[:-1] = perp[1:] / perp[:-1] - 1
    changes = np.abs(np.diff(positions, prepend=0.0))
    expo = float(np.abs(positions[positions != 0]).mean()) if (positions != 0).any() else 0.0
    return float((positions * returns - cost * changes).sum() * 100), expo


def stats(daily):
    arr = np.array(daily)
    return arr.mean(), arr.std(), (arr.mean() / (arr.std() + 1e-9))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", required=True)
    ap.add_argument("--days", type=int, default=35)
    ap.add_argument("--val-days", type=int, default=5)
    ap.add_argument("--test-days", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    dates = sorted(d.strip() for d in args.dates.split(",") if d.strip())[-args.days :]
    test_dates = set(dates[-args.test_days :])
    val_dates = set(dates[-args.test_days - args.val_days : -args.test_days])
    train_dates = set(dates[: -args.test_days - args.val_days])

    packed, report = {}, {}
    spreads = {p: effective_spread_bps(p, sorted(train_dates)) for p in PAIRS}
    for date in dates:
        for pair in PAIRS:
            e = build_episode(pair, date)
            if e is None:
                continue
            t = episode_targets(e, TRAIN_STRIDE)
            if t is not None:
                packed[(date, pair)] = t; report[(date, pair)] = e
    cost = {p: (TAKER_FEE_BPS + spreads[p] / 2 + EXTRA_BPS) / 1e4 for p in PAIRS}
    cost_bps = {p: cost[p] * 1e4 for p in PAIRS}
    started = time.time()
    boosters = train_avg(packed, train_dates, args.epochs, args.seeds)
    qcache = {k: preds_for(boosters, e) for k, e in report.items()}
    print(f"pairs {len(PAIRS)} | fee {TAKER_FEE_BPS}bps | trained {args.seeds} seeds [{time.time()-started:.0f}s]", flush=True)

    def portfolio(dates_set, build_pos):
        by_day = {}
        expos = []
        for (date, sym), e in report.items():
            if date not in dates_set:
                continue
            pos = build_pos(qcache[(date, sym)], sym)
            p, ex = pnl(pos, e["perp"][WARMUP:], cost[sym])
            by_day.setdefault(date, []).append(p)
            if ex:
                expos.append(ex)
        daily = [np.mean(v) for _, v in sorted(by_day.items())]
        return daily, (np.mean(expos) if expos else 0.0)

    # baseline: pick enter_scale by val Sharpe
    best = (-1e18, None)
    dir_cache = {}
    for scale in (0.5, 1.0, 1.5, 2.5):
        def bp(q, sym, sc=scale):
            key = (id(q), sc)
            if key not in dir_cache:
                dir_cache[key] = direction_series(q, sc * cost_bps[sym])
            return dir_cache[key]
        d, _ = portfolio(val_dates, bp)
        sh = stats(d)[2]
        if sh > best[0]:
            best = (sh, scale)
    base_scale = best[1]

    def base_pos(q, sym):
        return direction_series(q, base_scale * cost_bps[sym])

    bd, bexpo = portfolio(test_dates, base_pos)
    bm, bs, bsh = stats(bd)

    # sized: pick (scale, conf_k, conf_cap, tail_mult) by val Sharpe
    best = (-1e18, None)
    for scale in (0.5, 1.5):
        for conf_k in (1.0, 3.0):
            for conf_cap in (2.0, 4.0):
                for tail_mult in (1e9, 1.5):
                    def sp(q, sym, sc=scale, ck=conf_k, cap=conf_cap, tm=tail_mult):
                        direction = direction_series(q, sc * cost_bps[sym])
                        return size_series(direction, q, ck, cap, tm, cost_bps[sym])
                    d, _ = portfolio(val_dates, sp)
                    sh = stats(d)[2]
                    if sh > best[0]:
                        best = (sh, (scale, conf_k, conf_cap, tail_mult))
    sc, ck, cap, tm = best[1]

    def sized_pos(q, sym):
        direction = direction_series(q, sc * cost_bps[sym])
        return size_series(direction, q, ck, cap, tm, cost_bps[sym])

    sd, sexpo = portfolio(test_dates, sized_pos)
    sm, ss, ssh = stats(sd)

    print(f"\n=== TEST (real fees) ===")
    print(f"baseline (q25/q75 fixed±1):   mean {bm:+.3f}% | std {bs:.3f} | Sharpe {bsh:+.2f} | exposure {bexpo:.2f} | scale {base_scale}")
    print(f"sized+tailgate (uses q10..q90): mean {sm:+.3f}% | std {ss:.3f} | Sharpe {ssh:+.2f} | exposure {sexpo:.2f} | cfg scale={sc} k={ck} cap={cap} tail={'off' if tm>1e6 else tm}")
    print(f"\nΔ mean {sm-bm:+.3f}% | Δ Sharpe {ssh-bsh:+.2f} | exposure ratio {sexpo/(bexpo+1e-9):.2f}x")


if __name__ == "__main__":
    main()
