"""Fast subset A/B: does training the quantile model on the MAX favorable
excursion over the next 60s (signed max-abs path move) beat the current FINAL
(point return at +60s)? Same features, same hysteresis trading, REAL fees.
PnL is always measured on the realized path, so this isolates the target's
effect on entry/exit timing."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

from features_v2 import WARMUP, build_episode, effective_spread_bps
from nn_train import DEVICE, QuantileMLP, TorchQuantiles, pinball_loss
from xgb2_train import EXTRA_BPS, FORWARD_SECONDS, TAKER_FEE_BPS, TRAIN_STRIDE, portfolio_report

PAIRS = ["cash:INTC", "cash:HOOD", "xyz:ARM", "xyz:MSTR", "cash:NVDA", "cash:TSLA"]


def episode_targets(episode, stride):
    feats = episode["features"][WARMUP:]
    perp = episode["perp"][WARMUP:]
    horizon = len(perp) - FORWARD_SECONDS - 1
    if horizon <= 0:
        return None
    entry = perp[1 : horizon + 1]
    windows = sliding_window_view(perp[2:], FORWARD_SECONDS)[:horizon]
    rets = windows / entry[:, None] - 1.0
    final = perp[FORWARD_SECONDS + 1 : FORWARD_SECONDS + 1 + horizon] / entry - 1.0
    peak = rets[np.arange(horizon), np.argmax(np.abs(rets), axis=1)]
    return feats[:horizon][::stride], (final[::stride] * 1e4).astype(np.float32), (peak[::stride] * 1e4).astype(np.float32)


def train_eval(target_kind, episodes, train_dates, val_dates, test_dates, costs, epochs, seeds):
    train_x, train_y = [], []
    for (date, _s), packed in episodes.items():
        if date in train_dates:
            x, yf, ym = packed
            train_x.append(x)
            train_y.append(ym if target_kind == "max" else yf)
    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)
    val_eps = {k: e for k, e in episodes.items() if k[0] in val_dates}
    test_eps = {k: e for k, e in episodes.items() if k[0] in test_dates}

    mean, std = train_x.mean(0), train_x.std(0) + 1e-6
    y_mean, y_std = float(train_y.mean()), float(train_y.std() + 1e-6)
    tx = torch.from_numpy(((train_x - mean) / std).astype(np.float32)).to(DEVICE)
    ty = torch.from_numpy(((train_y - y_mean) / y_std).astype(np.float32)).to(DEVICE)

    alls, tops = [], []
    for seed in range(seeds):
        torch.manual_seed(seed)
        model = QuantileMLP(train_x.shape[1]).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        n = len(tx)
        for _ in range(epochs):
            model.train()
            perm = torch.randperm(n, device=DEVICE)
            for start in range(0, n, 16384):
                idx = perm[start : start + 16384]
                opt.zero_grad()
                pinball_loss(model(tx[idx]), ty[idx]).backward()
                opt.step()
            sched.step()
        booster = TorchQuantiles(model.eval(), mean, std, y_mean, y_std)
        # the eval reuses the SAME hysteresis backtest; only the trained target differs
        best_scale, best = 0.5, -1e18
        for scale in (0.5, 1.0, 1.5, 2.5, 4.0, 6.0):
            r = portfolio_report(booster, val_report_eps, costs, scale, "v")["mean_day_pct"]
            if r > best:
                best, best_scale = r, scale
        rep = portfolio_report(booster, test_report_eps, costs, best_scale, "t")
        alls.append(rep["mean_day_pct"])
        by = portfolio_report(booster, val_report_eps, costs, best_scale, "v")["by_symbol_mean"]
        top = [s for s, m in by.items() if m > 0][:3]
        tops.append(portfolio_report(booster, {k: e for k, e in test_report_eps.items() if k[1] in top}, costs, best_scale, "tt")["mean_day_pct"])
    return float(np.mean(alls)), float(np.std(alls)), float(np.mean(tops))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", required=True)
    parser.add_argument("--days", type=int, default=35)
    parser.add_argument("--val-days", type=int, default=5)
    parser.add_argument("--test-days", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()
    dates = sorted(d.strip() for d in args.dates.split(",") if d.strip())[-args.days :]
    test_dates = set(dates[-args.test_days :])
    val_dates = set(dates[-args.test_days - args.val_days : -args.test_days])
    train_dates = set(dates[: -args.test_days - args.val_days])

    global val_report_eps, test_report_eps
    packed_episodes = {}
    report_episodes = {}
    for date in dates:
        for pair in PAIRS:
            e = build_episode(pair, date)
            if e is None:
                continue
            t = episode_targets(e, TRAIN_STRIDE)
            if t is None:
                continue
            packed_episodes[(date, pair)] = t
            report_episodes[(date, pair)] = e
    val_report_eps = {k: e for k, e in report_episodes.items() if k[0] in val_dates}
    test_report_eps = {k: e for k, e in report_episodes.items() if k[0] in test_dates}
    costs = {p: (TAKER_FEE_BPS + effective_spread_bps(p, sorted(train_dates)) / 2 + EXTRA_BPS) / 1e4 for p in PAIRS}
    print(f"pairs {len(PAIRS)} | episodes {len(packed_episodes)} | fee {TAKER_FEE_BPS}bps | epochs {args.epochs} seeds {args.seeds}", flush=True)

    for kind in ("final", "max"):
        started = time.time()
        all_mean, all_std, top_mean = train_eval(kind, packed_episodes, train_dates, val_dates, test_dates, costs, args.epochs, args.seeds)
        print(f"  target={kind:>5}: test_all {all_mean:+.3f}±{all_std:.3f}% | test_top3 {top_mean:+.3f}% | {time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
