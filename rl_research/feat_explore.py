"""Fast feature playground: rebuild the feature matrix from raw per-second
series (perp + stock reconstructed from premium) so lag ladders and feature
sets can be swept without touching the cache. Ranks variants by val_corr
(stable) on a small subset; PnL shown as secondary."""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from features_v2 import SYMBOLS, WARMUP, build_episode, effective_spread_bps, rolling_std, rolling_sum
from nn_train import DEVICE, QuantileMLP, TorchQuantiles, pinball_loss
from xgb2_train import EXTRA_BPS, FORWARD_SECONDS, TAKER_FEE_BPS, TRAIN_STRIDE, portfolio_report

PAIRS = ["HOOD", "NVDA", "INTC", "MSTR", "CRWV", "TSM"]
# cached-feature column indices for the auxiliary (non-lag) block
AUX_COLS = [18, 19, 20, 21, 22, 23, 24]  # flow30/120, perp/stock prints, staleness, vwap_dev, session


def lag_return(series: np.ndarray, lag: int) -> np.ndarray:
    out = np.zeros(len(series))
    out[lag:] = (series[lag:] / series[:-lag] - 1) * 100
    return out


def lag_diff(series: np.ndarray, lag: int) -> np.ndarray:
    out = np.zeros(len(series))
    out[lag:] = series[lag:] - series[:-lag]
    return out


def build_features(episode: dict, pair: str, cfg: dict) -> np.ndarray:
    cached = episode["features"]
    perp = episode["perp"]
    premium = cached[:, 14].astype(np.float64)
    stock = perp * (1 + premium / 100)
    perp_ret_1s = lag_return(perp, 1)

    columns: list[np.ndarray] = []
    for lag in cfg["price_lags"]:
        columns.append(lag_return(perp, lag))
    for lag in cfg["price_lags"]:
        columns.append(lag_return(stock, lag))
    for lag in cfg["prem_lags"]:
        columns.append(lag_diff(premium, lag))
    columns.append(premium)
    z_windows = cfg.get("prem_z_windows") or [cfg["prem_z_window"]]
    base_z = None
    for window in z_windows:
        prem_mean = rolling_sum(premium, window) / window
        prem_std = rolling_std(premium, window)
        z = (premium - prem_mean) / (prem_std + 1e-4)
        if base_z is None:
            base_z = z
        columns.append(z)
    for vol_window in cfg["vol_windows"]:
        columns.append(rolling_std(perp_ret_1s, vol_window))

    premium_series = pd.Series(premium)
    for window in cfg.get("stretch_windows", []):
        columns.append(premium - rolling_sum(premium, window) / window)
    premium_diff_1s = lag_diff(premium, 1)
    for window in cfg.get("prem_vol_windows", []):
        columns.append(rolling_std(premium_diff_1s, window))
    for span in cfg.get("ema_stretch_spans", []):
        ema = premium_series.ewm(span=span, adjust=False).mean().to_numpy()
        columns.append(premium - ema)
    range_window = cfg.get("range_window")
    if range_window:
        low = premium_series.rolling(range_window, min_periods=1).min().to_numpy()
        high = premium_series.rolling(range_window, min_periods=1).max().to_numpy()
        columns.append((premium - low) / (high - low + 1e-4))

    perp_vol_60 = rolling_std(perp_ret_1s, 60)
    extras = cfg.get("extras", [])
    if "prem_accel" in extras:
        columns.append(premium - 2 * np.roll(premium, 30) + np.roll(premium, 60))
    if "leadlag" in extras:
        columns.append(lag_return(stock, 8) - lag_return(perp, 8))
    if "stock_vol_ratio" in extras:
        stock_vol_60 = rolling_std(lag_return(stock, 1), 60)
        columns.append(perp_vol_60 / (stock_vol_60 + 1e-4))
    if "regime" in extras:
        columns.append(base_z * perp_vol_60)

    matrix = np.stack(columns, axis=1)
    if cfg.get("aux", True):
        matrix = np.concatenate([matrix, cached[:, AUX_COLS]], axis=1)
    one_hot = np.zeros((len(perp), len(PAIRS)), dtype=np.float64)
    one_hot[:, PAIRS.index(pair)] = 1.0
    return np.concatenate([matrix, one_hot], axis=1).astype(np.float32)


def rows(features: np.ndarray, perp: np.ndarray, stride: int):
    feats = features[WARMUP:]
    px = perp[WARMUP:]
    horizon = len(px) - FORWARD_SECONDS - 1
    target = (px[FORWARD_SECONDS + 1 :] / px[1:-FORWARD_SECONDS] - 1) * 1e4
    return feats[:horizon:stride], target[::stride]


def evaluate(cfg, episodes, train_dates, val_dates, test_dates, costs, epochs, seeds):
    built = {k: {"features": build_features(e, k[1], cfg), "perp": e["perp"]} for k, e in episodes.items()}
    train_x, train_y = [], []
    for (date, _s), e in built.items():
        if date in train_dates:
            fx, fy = rows(e["features"], e["perp"], TRAIN_STRIDE)
            train_x.append(fx)
            train_y.append(fy)
    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)
    val_eps = {k: e for k, e in built.items() if k[0] in val_dates}
    test_eps = {k: e for k, e in built.items() if k[0] in test_dates}
    val_x, val_y = [], []
    for e in val_eps.values():
        fx, fy = rows(e["features"], e["perp"], TRAIN_STRIDE)
        val_x.append(fx)
        val_y.append(fy)
    val_x = np.concatenate(val_x)
    val_y = np.concatenate(val_y)

    mean, std = train_x.mean(0), train_x.std(0) + 1e-6
    y_mean, y_std = float(train_y.mean()), float(train_y.std() + 1e-6)
    tx = torch.from_numpy(((train_x - mean) / std).astype(np.float32)).to(DEVICE)
    ty = torch.from_numpy(((train_y - y_mean) / y_std).astype(np.float32)).to(DEVICE)
    vx = torch.from_numpy(((val_x - mean) / std).astype(np.float32)).to(DEVICE)

    corrs, all_means, worsts, pos_fracs, top_means = [], [], [], [], []
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
        preds = booster.inplace_predict(val_x)
        corrs.append(float(np.corrcoef(preds[:, 2], val_y)[0, 1]))
        best_scale, best = 0.5, -1e18
        for scale in (0.5, 1.0, 1.5):
            r = portfolio_report(booster, val_eps, costs, scale, "v")["mean_day_pct"]
            if r > best:
                best, best_scale = r, scale
        report_all = portfolio_report(booster, test_eps, costs, best_scale, "t")
        all_means.append(report_all["mean_day_pct"])
        worsts.append(report_all["worst_day_pct"])
        positive, total = report_all["positive_days"].split("/")
        pos_fracs.append(int(positive) / int(total))
        val_report = portfolio_report(booster, val_eps, costs, best_scale, "v")
        top = [s for s, m in val_report["by_symbol_mean"].items() if m > 0][:8]
        top_means.append(portfolio_report(booster, {k: e for k, e in test_eps.items() if k[1] in top}, costs, best_scale, "tt")["mean_day_pct"])
    return {
        "nfeat": train_x.shape[1],
        "corr": float(np.mean(corrs)), "corr_std": float(np.std(corrs)),
        "all": float(np.mean(all_means)), "all_std": float(np.std(all_means)),
        "worst": float(np.mean(worsts)), "pos": float(np.mean(pos_fracs)),
        "top": float(np.mean(top_means)), "top_std": float(np.std(top_means)),
    }


LV = {"price_lags": [3, 8, 21, 55, 144, 377], "prem_lags": [10, 30, 90, 300], "vol_windows": [60, 300, 900], "aux": True}
CONFIGS = [
    {"name": "baseline(old SOTA)", "price_lags": [3, 8, 21, 55, 144], "prem_lags": [10, 30, 60, 120], "prem_z_window": 600, "vol_windows": [60, 300], "aux": True},
    {"name": "lv+multiz", **LV, "prem_z_windows": [300, 900, 1800]},
    {"name": "mz+pv+st+range", **LV, "prem_z_windows": [300, 900, 1800], "prem_vol_windows": [60, 300], "stretch_windows": [120, 600], "range_window": 600},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", required=True)
    parser.add_argument("--days", type=int, default=35)
    parser.add_argument("--val-days", type=int, default=5)
    parser.add_argument("--test-days", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--pairs", default="liquid6", help="'liquid6', 'all', or comma list")
    args = parser.parse_args()
    global PAIRS
    if args.pairs == "all":
        PAIRS = list(SYMBOLS)
    elif args.pairs != "liquid6":
        PAIRS = args.pairs.split(",")
    dates = sorted(d.strip() for d in args.dates.split(",") if d.strip())[-args.days :]
    test_dates = set(dates[-args.test_days :])
    val_dates = set(dates[-args.test_days - args.val_days : -args.test_days])
    train_dates = set(dates[: -args.test_days - args.val_days])
    print(f"subset {len(dates)}d | pairs {PAIRS} | epochs {args.epochs} seeds {args.seeds}", flush=True)

    episodes = {}
    for date in dates:
        for pair in PAIRS:
            e = build_episode(pair, date)
            if e is not None:
                episodes[(date, pair)] = e
    costs = {p: (TAKER_FEE_BPS + effective_spread_bps(p, sorted(train_dates)) / 2 + EXTRA_BPS) / 1e4 for p in PAIRS}
    print(f"pairs {len(PAIRS)} | episodes {len(episodes)}", flush=True)

    results = []
    for cfg in CONFIGS:
        started = time.time()
        m = evaluate(cfg, episodes, train_dates, val_dates, test_dates, costs, args.epochs, args.seeds)
        m["name"] = cfg["name"]
        results.append(m)
        print(f"  {cfg['name']:>18} | feats {m['nfeat']:>2} | corr {m['corr']:+.4f} | all {m['all']:+.3f}% | top {m['top']:+.3f}% | {time.time()-started:.0f}s", flush=True)

    base = results[0]
    print(f"\n=== COMPARISON: new vs old SOTA ({len(PAIRS)} pairs, {len(dates)}d, {args.seeds} seeds) ===")
    header = f"{'config':>18} | {'feats':>5} | {'val_corr':>16} | {'test_all/day':>16} | {'worst':>7} | {'pos':>5} | {'test_top/day':>16}"
    print(header)
    print("-" * len(header))
    for m in results:
        print(
            f"{m['name']:>18} | {m['nfeat']:>5} | {m['corr']:+.4f}±{m['corr_std']:.4f} | "
            f"{m['all']:+.3f}±{m['all_std']:.3f}% | {m['worst']:+.3f} | {m['pos']*100:>4.0f}% | {m['top']:+.3f}±{m['top_std']:.3f}%"
        )
    print("\n=== Δ vs baseline (old SOTA) ===")
    for m in results[1:]:
        print(
            f"{m['name']:>18} | Δcorr {m['corr']-base['corr']:+.4f} | "
            f"Δall {m['all']-base['all']:+.3f}% ({(m['all']/base['all']-1)*100:+.1f}%) | "
            f"Δtop {m['top']-base['top']:+.3f}% ({(m['top']/base['top']-1)*100:+.1f}%) | Δworst {m['worst']-base['worst']:+.3f}"
        )


if __name__ == "__main__":
    main()
