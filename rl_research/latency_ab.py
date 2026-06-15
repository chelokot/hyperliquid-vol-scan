"""Targeted test of the latency-fake-premium skew: live perp (HL ws) is fresher
than live stock (Finnhub), so premium = stock/perp is computed with a STALE
stock. Model it physically: reconstruct stock, lag it by delta seconds, recompute
every stock-dependent feature, measure PnL drop. Real fees."""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from features_v2 import (
    FEATURE_NAMES, PREMIUM_LAGS, PREMIUM_RANGE_WINDOW, PREMIUM_STRETCH_WINDOWS,
    PREMIUM_VOL_WINDOWS, PREMIUM_Z_WINDOWS, PRICE_LAGS, WARMUP,
    build_episode, effective_spread_bps, rolling_std, rolling_sum,
)
from nn_train import DEVICE, QuantileMLP, TorchQuantiles, pinball_loss
from xgb2_train import EXTRA_BPS, TAKER_FEE_BPS, TRAIN_STRIDE, portfolio_report
from target_ab import PAIRS, episode_targets

IDX = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
PREMIUM_I = IDX["premium"]


def lag_series(x, f):
    k = int(np.floor(f)); frac = f - k
    def shift(n):
        return np.concatenate([np.full(n, x[0]), x[:-n]]) if n > 0 else x
    return (1 - frac) * shift(k) + frac * shift(k + 1)


def perturb(episode, delta):
    f = episode["features"].copy()
    perp = episode["perp"]
    stock = perp * (1 + f[:, PREMIUM_I] / 100)
    stock = lag_series(stock, delta)  # stale stock (perp fresher)
    premium = (stock / perp - 1) * 100
    length = len(perp)
    for i, lag in enumerate(PRICE_LAGS):
        col = IDX[f"stock_ret_{lag}s"]
        shifted = np.zeros(length); shifted[lag:] = (stock[lag:] / stock[:-lag] - 1) * 100
        f[:, col] = shifted
    for lag in PREMIUM_LAGS:
        shifted = np.zeros(length); shifted[lag:] = premium[lag:] - premium[:-lag]
        f[:, IDX[f"premium_delta_{lag}s"]] = shifted
    f[:, PREMIUM_I] = premium
    for w in PREMIUM_Z_WINDOWS:
        m = rolling_sum(premium, w) / w; s = rolling_std(premium, w)
        f[:, IDX[f"premium_z_{w}s"]] = (premium - m) / (s + 1e-4)
    for w in PREMIUM_STRETCH_WINDOWS:
        f[:, IDX[f"premium_stretch_{w}s"]] = premium - rolling_sum(premium, w) / w
    pdiff = np.zeros(length); pdiff[1:] = premium[1:] - premium[:-1]
    for w in PREMIUM_VOL_WINDOWS:
        f[:, IDX[f"premium_vol_{w}s"]] = rolling_std(pdiff, w)
    ps = pd.Series(premium)
    low = ps.rolling(PREMIUM_RANGE_WINDOW, min_periods=1).min().to_numpy()
    high = ps.rolling(PREMIUM_RANGE_WINDOW, min_periods=1).max().to_numpy()
    f[:, IDX[f"premium_range_{PREMIUM_RANGE_WINDOW}s"]] = (premium - low) / (high - low + 1e-4)
    return {"features": f, "perp": perp}


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


def best_test(b, val_eps, test_eps, costs):
    best_s, best = 0.5, -1e18
    for sc in (0.5, 1.0, 1.5, 2.5, 4.0):
        r = portfolio_report(b, val_eps, costs, sc, "v")["mean_day_pct"]
        if r > best:
            best, best_s = r, sc
    return portfolio_report(b, test_eps, costs, best_s, "t")["mean_day_pct"]


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
    deltas = [0.0, 0.3, 0.5, 1.0, 2.0]
    print(f"fee {TAKER_FEE_BPS} | seeds {args.seeds} | stock-lag deltas {deltas}", flush=True)
    agg = {d: [] for d in deltas}
    started = time.time()
    for seed in range(args.seeds):
        b = train(packed, train_dates, args.epochs, seed)
        for d in deltas:
            te = test_eps if d == 0 else {k: perturb(e, d) for k, e in test_eps.items()}
            ve = val_eps if d == 0 else {k: perturb(e, d) for k, e in val_eps.items()}
            agg[d].append(best_test(b, ve, te, costs))
    print(f"trained {args.seeds} seeds [{time.time()-started:.0f}s]\n")
    clean = np.mean(agg[0.0])
    for d in deltas:
        m = np.mean(agg[d])
        tag = "  (clean)" if d == 0 else f"  drop {clean-m:+.3f}% ({(m/clean-1)*100:+.0f}%)"
        print(f"  stock stale {d:>3}s: test_all {m:+.3f}%{tag}")


if __name__ == "__main__":
    main()
