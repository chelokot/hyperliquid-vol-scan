"""Fast subset test: peak-target model + MAKER take-profit exit (limit at the
predicted peak; taker entry, taker time-stop fallback) vs the taker-hysteresis
baseline, at realistic fees (taker 4.5 / maker 1.5 bps). Does earning the maker
rebate on the exit + capturing the predicted peak beat the taker baseline?"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from features_v2 import WARMUP, build_episode, effective_spread_bps
from nn_train import DEVICE, QuantileMLP, TorchQuantiles, pinball_loss
from xgb2_train import EXTRA_BPS, TAKER_FEE_BPS, TRAIN_STRIDE, portfolio_report
from target_ab import PAIRS, episode_targets

MAKER_FEE_BPS = 1.5


def train_model(target_kind, packed, train_dates, epochs, seed):
    xs, ys = [], []
    for (date, _s), (x, yf, ym) in packed.items():
        if date in train_dates:
            xs.append(x)
            ys.append(ym if target_kind == "max" else yf)
    train_x = np.concatenate(xs)
    train_y = np.concatenate(ys)
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
            opt.zero_grad()
            pinball_loss(model(tx[idx]), ty[idx]).backward()
            opt.step()
        sched.step()
    return TorchQuantiles(model.eval(), mean, std, y_mean, y_std)


def maker_exit_episode(booster, episode, enter_bps, tp_mult, half, taker_c, maker_c, max_hold):
    feats = episode["features"][WARMUP:]
    perp = episode["perp"][WARMUP:]
    q = booster.inplace_predict(feats)
    q25, q50, q75 = q[:, 1], q[:, 2], q[:, 3]
    pnl, i, n = 0.0, 1, len(perp)
    while i < n - max_hold - 1:
        side = 1 if q25[i] > enter_bps else (-1 if q75[i] < -enter_bps else 0)
        if side == 0:
            i += 1
            continue
        entry = perp[i] * (1 + side * half)
        tp = entry * (1 + side * tp_mult * abs(q50[i]) / 1e4)
        exit_px, fee, step = None, maker_c, max_hold
        for k in range(1, max_hold + 1):
            if side * (perp[i + k] - tp) >= 0:
                exit_px, fee, step = tp, maker_c, k
                break
        if exit_px is None:
            exit_px, fee = perp[i + max_hold] * (1 - side * half), taker_c
        pnl += side * (exit_px - entry) / entry - taker_c - fee
        i += step + 1
    return pnl * 100


def maker_portfolio(booster, episodes, params, max_hold):
    by_day = {}
    by_symbol = {}
    for (date, symbol), ep in episodes.items():
        p = params[symbol]
        v = maker_exit_episode(booster, ep, p["enter"], p["tp"], p["half"], p["taker"], p["maker"], max_hold)
        by_day.setdefault(date, []).append(v)
        by_symbol.setdefault(symbol, []).append(v)
    daily = [np.mean(v) for v in by_day.values()]
    return {"mean": float(np.mean(daily)), "worst": float(np.min(daily)),
            "pos": sum(1 for d in daily if d > 0), "days": len(daily),
            "by_symbol": {s: float(np.mean(v)) for s, v in by_symbol.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", required=True)
    ap.add_argument("--days", type=int, default=35)
    ap.add_argument("--val-days", type=int, default=5)
    ap.add_argument("--test-days", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--max-hold", type=int, default=90)
    args = ap.parse_args()
    dates = sorted(d.strip() for d in args.dates.split(",") if d.strip())[-args.days :]
    test_dates = set(dates[-args.test_days :])
    val_dates = set(dates[-args.test_days - args.val_days : -args.test_days])
    train_dates = set(dates[: -args.test_days - args.val_days])

    packed, report = {}, {}
    spreads = {}
    for p in PAIRS:
        spreads[p] = effective_spread_bps(p, sorted(train_dates))
    for date in dates:
        for pair in PAIRS:
            e = build_episode(pair, date)
            if e is None:
                continue
            t = episode_targets(e, TRAIN_STRIDE)
            if t is None:
                continue
            packed[(date, pair)] = t
            report[(date, pair)] = e
    val_eps = {k: e for k, e in report.items() if k[0] in val_dates}
    test_eps = {k: e for k, e in report.items() if k[0] in test_dates}
    taker_costs = {p: (TAKER_FEE_BPS + spreads[p] / 2 + EXTRA_BPS) / 1e4 for p in PAIRS}
    print(f"pairs {len(PAIRS)} | episodes {len(packed)} | taker {TAKER_FEE_BPS} maker {MAKER_FEE_BPS} | hold {args.max_hold}s | seeds {args.seeds}", flush=True)

    # baseline: FINAL-target model + taker hysteresis
    started = time.time()
    base_alls = []
    for seed in range(args.seeds):
        b = train_model("final", packed, train_dates, args.epochs, seed)
        best_s, best = 0.5, -1e18
        for sc in (0.5, 1.0, 1.5, 2.5, 4.0):
            r = portfolio_report(b, val_eps, taker_costs, sc, "v")["mean_day_pct"]
            if r > best:
                best, best_s = r, sc
        base_alls.append(portfolio_report(b, test_eps, taker_costs, best_s, "t")["mean_day_pct"])
    print(f"  TAKER baseline (final+hysteresis): test_all {np.mean(base_alls):+.3f}±{np.std(base_alls):.3f}%  [{time.time()-started:.0f}s]", flush=True)

    # maker: PEAK-target model + maker TP exit
    for seed in range(args.seeds):
        started = time.time()
        b = train_model("max", packed, train_dates, args.epochs, seed)
        best, best_cfg = -1e18, None
        for scale in (0.5, 1.0, 1.5, 2.5):
            for tp_mult in (0.5, 0.75, 1.0):
                params = {p: {"enter": scale * taker_costs[p] * 1e4, "tp": tp_mult,
                              "half": spreads[p] / 2 / 1e4,
                              "taker": (TAKER_FEE_BPS + EXTRA_BPS) / 1e4, "maker": MAKER_FEE_BPS / 1e4} for p in PAIRS}
                r = maker_portfolio(b, val_eps, params, args.max_hold)["mean"]
                if r > best:
                    best, best_cfg = r, (scale, tp_mult, params)
        scale, tp_mult, params = best_cfg
        rep = maker_portfolio(b, test_eps, params, args.max_hold)
        print(f"  MAKER seed {seed} (peak+TP, scale {scale} tp×{tp_mult}): test_all {rep['mean']:+.3f}% worst {rep['worst']:+.3f} {rep['pos']}/{rep['days']}  [{time.time()-started:.0f}s]", flush=True)
        if seed == args.seeds - 1:
            print("    per-pair:", {s: round(v, 2) for s, v in sorted(rep["by_symbol"].items(), key=lambda kv: -kv[1])})


if __name__ == "__main__":
    main()
