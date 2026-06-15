"""Targeted robustness test: inject the SPECIFIC consolidated-vs-IEX stock skew
(stock_prints_10s shifted up by +log(k) for k× denser prints; stock_staleness
scaled down) into the test features and measure PnL degradation. Real fees."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from features_v2 import FEATURE_NAMES, WARMUP, build_episode, effective_spread_bps
from nn_train import DEVICE, QuantileMLP, TorchQuantiles, pinball_loss
from xgb2_train import EXTRA_BPS, TAKER_FEE_BPS, TRAIN_STRIDE, portfolio_report
from target_ab import PAIRS, episode_targets

PRINTS_IDX = FEATURE_NAMES.index("stock_prints_10s")
STALE_IDX = FEATURE_NAMES.index("stock_staleness")


def train(packed, train_dates, epochs, seed):
    xs, ys = [], []
    for (date, _s), (x, yf, _ym) in packed.items():
        if date in train_dates:
            xs.append(x); ys.append(yf)
    train_x = np.concatenate(xs); train_y = np.concatenate(ys)
    mean, std = train_x.mean(0), train_x.std(0) + 1e-6
    y_mean, y_std = float(train_y.mean()), float(train_y.std() + 1e-6)
    tx = torch.from_numpy(((train_x - mean) / std).astype(np.float32)).to(DEVICE)
    ty = torch.from_numpy(((train_y - y_mean) / y_std).astype(np.float32)).to(DEVICE)
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
    return TorchQuantiles(model.eval(), mean, std, y_mean, y_std)


def shift_eps(episodes, prints_add, stale_mult):
    out = {}
    for k, e in episodes.items():
        f = e["features"].copy()
        f[:, PRINTS_IDX] = f[:, PRINTS_IDX] + prints_add  # log-feature: +log(k) for k× denser
        f[:, STALE_IDX] = f[:, STALE_IDX] * stale_mult
        out[k] = {"features": f, "perp": e["perp"]}
    return out


def best_test(booster, val_eps, test_eps, costs):
    best_s, best = 0.5, -1e18
    for sc in (0.5, 1.0, 1.5, 2.5, 4.0):
        r = portfolio_report(booster, val_eps, costs, sc, "v")["mean_day_pct"]
        if r > best:
            best, best_s = r, sc
    return portfolio_report(booster, test_eps, costs, best_s, "t")["mean_day_pct"]


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
    costs = {p: (TAKER_FEE_BPS + spreads[p] / 2 + EXTRA_BPS) / 1e4 for p in PAIRS}
    val_eps = {k: e for k, e in report.items() if k[0] in val_dates}
    test_eps = {k: e for k, e in report.items() if k[0] in test_dates}
    print(f"prints_idx {PRINTS_IDX} stale_idx {STALE_IDX} | fee {TAKER_FEE_BPS} | seeds {args.seeds}", flush=True)

    scenarios = [
        ("clean", 0.0, 1.0),
        ("2x prints, stale x0.5", np.log(2), 0.5),
        ("3x prints, stale x0.25", np.log(3), 0.25),
        ("5x prints, stale ~0", np.log(5), 0.05),
    ]
    agg = {name: [] for name, _, _ in scenarios}
    started = time.time()
    for seed in range(args.seeds):
        booster = train(packed, train_dates, args.epochs, seed)
        for name, padd, smult in scenarios:
            te = test_eps if name == "clean" else shift_eps(test_eps, padd, smult)
            ve = val_eps if name == "clean" else shift_eps(val_eps, padd, smult)
            agg[name].append(best_test(booster, ve, te, costs))
    print(f"trained {args.seeds} seeds [{time.time()-started:.0f}s]\n")
    clean = np.mean(agg["clean"])
    for name, _, _ in scenarios:
        m = np.mean(agg[name])
        print(f"  {name:>24}: test_all {m:+.3f}%   drop {clean-m:+.3f}% ({(m/clean-1)*100:+.0f}%)")


if __name__ == "__main__":
    main()
