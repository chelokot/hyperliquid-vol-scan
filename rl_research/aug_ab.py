"""Does affine feature augmentation during training buy robustness to a
train/serve domain shift? Train baseline vs augmented (per-step random
scale+shift on market features), then evaluate BOTH on clean test AND on test
with an injected affine skew (mimicking the IEX->consolidated source change).
Real fees. Win = augmented degrades less under skew without losing clean PnL."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from features_v2 import MARKET_FEATURES, WARMUP, build_episode, effective_spread_bps
from nn_train import DEVICE, QuantileMLP, TorchQuantiles, pinball_loss
from xgb2_train import EXTRA_BPS, TAKER_FEE_BPS, TRAIN_STRIDE, portfolio_report
from target_ab import PAIRS, episode_targets

N_MARKET = len(MARKET_FEATURES)  # first N_MARKET cols are source-skew-prone; rest is one-hot
AUG_SCALE_SD = 0.08
AUG_SHIFT_SD = 0.15  # in normalized-feature units


def train_model(packed, train_dates, epochs, seed, augment):
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
            xb = tx[idx]
            if augment:
                # one affine per feature per step (session-level shift proxy), market cols only
                scale = 1.0 + AUG_SCALE_SD * torch.randn(N_MARKET, device=DEVICE)
                shift = AUG_SHIFT_SD * torch.randn(N_MARKET, device=DEVICE)
                xb = xb.clone()
                xb[:, :N_MARKET] = xb[:, :N_MARKET] * scale + shift
            opt.zero_grad(); pinball_loss(model(xb), ty[idx]).backward(); opt.step()
        sched.step()
    return TorchQuantiles(model.eval(), mean, std, y_mean, y_std), mean, std


def skew_episodes(episodes, std, rng):
    # fixed affine skew on raw market features: x' = x*(1+a) + b*std, mimics source change
    a = AUG_SCALE_SD * rng.standard_normal(N_MARKET).astype(np.float32)
    b = AUG_SHIFT_SD * rng.standard_normal(N_MARKET).astype(np.float32) * std[:N_MARKET]
    out = {}
    for k, e in episodes.items():
        f = e["features"].copy()
        f[:, :N_MARKET] = f[:, :N_MARKET] * (1 + a) + b
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
    print(f"pairs {len(PAIRS)} | market feats {N_MARKET} | fee {TAKER_FEE_BPS} | seeds {args.seeds}", flush=True)

    for label, aug in (("baseline", False), ("augmented", True)):
        clean, skewed = [], []
        for seed in range(args.seeds):
            started = time.time()
            booster, mean, std = train_model(packed, train_dates, args.epochs, seed, aug)
            clean.append(best_test(booster, val_eps, test_eps, costs))
            rng = np.random.default_rng(123)  # SAME injected skew for all models
            sk_test = skew_episodes(test_eps, std, rng)
            sk_val = skew_episodes(val_eps, std, np.random.default_rng(123))
            skewed.append(best_test(booster, sk_val, sk_test, costs))
        print(f"  {label:>9}: clean {np.mean(clean):+.3f}%  |  skewed {np.mean(skewed):+.3f}%  |  drop {np.mean(clean)-np.mean(skewed):+.3f}%  [{time.time()-started:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
