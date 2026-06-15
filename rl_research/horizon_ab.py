"""Fast subset sweep over the prediction horizon (point return at +H seconds).
Same features (horizon-independent), same hysteresis trading, REAL fees. Which
H gives the best test PnL?"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from features_v2 import WARMUP, build_episode, effective_spread_bps
from nn_train import DEVICE, QuantileMLP, TorchQuantiles, pinball_loss
from xgb2_train import EXTRA_BPS, TAKER_FEE_BPS, TRAIN_STRIDE, portfolio_report
from target_ab import PAIRS

HORIZONS = [15, 20, 30, 40, 60, 90, 120]


def horizon_target(episode, horizon, stride):
    perp = episode["perp"][WARMUP:]
    feats = episode["features"][WARMUP:]
    rows = len(perp) - horizon - 1
    if rows <= 0:
        return None
    target = (perp[horizon + 1 : horizon + 1 + rows] / perp[1 : rows + 1] - 1) * 1e4
    return feats[:rows][::stride], target[::stride].astype(np.float32)


def train_eval(horizon, report_eps, val_dates, test_dates, costs, epochs, seeds):
    xs, ys = [], []
    for (date, _s), ep in report_eps.items():
        if date in val_dates or date in test_dates:
            continue
        t = horizon_target(ep, horizon, TRAIN_STRIDE)
        if t is not None:
            xs.append(t[0]); ys.append(t[1])
    train_x = np.concatenate(xs); train_y = np.concatenate(ys)
    mean, std = train_x.mean(0), train_x.std(0) + 1e-6
    y_mean, y_std = float(train_y.mean()), float(train_y.std() + 1e-6)
    tx = torch.from_numpy(((train_x - mean) / std).astype(np.float32)).to(DEVICE)
    ty = torch.from_numpy(((train_y - y_mean) / y_std).astype(np.float32)).to(DEVICE)
    val_eps = {k: e for k, e in report_eps.items() if k[0] in val_dates}
    test_eps = {k: e for k, e in report_eps.items() if k[0] in test_dates}
    alls = []
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
                opt.zero_grad()
                pinball_loss(model(tx[idx]), ty[idx]).backward()
                opt.step()
            sched.step()
        booster = TorchQuantiles(model.eval(), mean, std, y_mean, y_std)
        best_s, best = 0.5, -1e18
        for sc in (0.5, 1.0, 1.5, 2.5, 4.0):
            r = portfolio_report(booster, val_eps, costs, sc, "v")["mean_day_pct"]
            if r > best:
                best, best_s = r, sc
        alls.append(portfolio_report(booster, test_eps, costs, best_s, "t")["mean_day_pct"])
    return float(np.mean(alls)), float(np.std(alls))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", required=True)
    ap.add_argument("--days", type=int, default=35)
    ap.add_argument("--val-days", type=int, default=5)
    ap.add_argument("--test-days", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    dates = sorted(d.strip() for d in args.dates.split(",") if d.strip())[-args.days :]
    test_dates = set(dates[-args.test_days :])
    val_dates = set(dates[-args.test_days - args.val_days : -args.test_days])
    train_dates = set(dates[: -args.test_days - args.val_days])

    report = {}
    for date in dates:
        for pair in PAIRS:
            e = build_episode(pair, date)
            if e is not None:
                report[(date, pair)] = e
    costs = {p: (TAKER_FEE_BPS + effective_spread_bps(p, sorted(train_dates)) / 2 + EXTRA_BPS) / 1e4 for p in PAIRS}
    print(f"pairs {len(PAIRS)} | fee {TAKER_FEE_BPS}bps | seeds {args.seeds}", flush=True)
    results = []
    for h in HORIZONS:
        started = time.time()
        m, s = train_eval(h, report, val_dates, test_dates, costs, args.epochs, args.seeds)
        results.append((h, m, s))
        print(f"  H={h:>3}s: test_all {m:+.3f}±{s:.3f}%  [{time.time()-started:.0f}s]", flush=True)
    print("\n=== ranked ===")
    for h, m, s in sorted(results, key=lambda r: -r[1]):
        print(f"  H={h:>3}s: {m:+.3f}±{s:.3f}%")


if __name__ == "__main__":
    main()
