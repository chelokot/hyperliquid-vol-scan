"""Neural (GPU) A/B: baseline 47 features vs +4 candidates, on our champion
model. Same days/split/architecture per arm; only the feature matrix differs.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from features_v2 import SYMBOLS, build_episode, effective_spread_bps
from features_extra import NEW_NAMES, augment, market_factor
from nn_train import DEVICE, QuantileMLP, TorchQuantiles, pinball_loss
from xgb2_train import (
    EXTRA_BPS,
    QUANTILES,
    TAKER_FEE_BPS,
    TRAIN_STRIDE,
    episode_rows,
    portfolio_report,
)


def build_matrix(episodes: dict, dates: set[str], stride: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for (date, _symbol), episode in episodes.items():
        if date in dates:
            features, targets = episode_rows(episode, stride)
            xs.append(features)
            ys.append(targets)
    return np.concatenate(xs), np.concatenate(ys)


def train_arm(name, episodes, train_dates, val_dates, test_dates, costs, epochs, batch, stride, seed):
    torch.manual_seed(seed)
    train_x, train_y = build_matrix(episodes, set(train_dates), stride)
    val_x, val_y = build_matrix(episodes, set(val_dates), stride)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0) + 1e-6
    y_mean, y_std = float(train_y.mean()), float(train_y.std() + 1e-6)
    tx = torch.from_numpy(((train_x - mean) / std).astype(np.float32)).to(DEVICE)
    ty = torch.from_numpy(((train_y - y_mean) / y_std).astype(np.float32)).to(DEVICE)
    vx = torch.from_numpy(((val_x - mean) / std).astype(np.float32)).to(DEVICE)
    vy = torch.from_numpy(((val_y - y_mean) / y_std).astype(np.float32)).to(DEVICE)

    model = QuantileMLP(train_x.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n = len(tx)
    started, best_val, best_state = time.time(), 1e18, None
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for start in range(0, n, batch):
            idx = perm[start : start + batch]
            optimizer.zero_grad()
            pinball_loss(model(tx[idx]), ty[idx]).backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_loss = pinball_loss(model(vx), vy).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    booster = TorchQuantiles(model, mean, std, y_mean, y_std)
    val_preds = booster.inplace_predict(val_x)
    correlation = float(np.corrcoef(val_preds[:, 2], val_y)[0, 1])
    val_episodes = {k: e for k, e in episodes.items() if k[0] in set(val_dates)}
    test_episodes = {k: e for k, e in episodes.items() if k[0] in set(test_dates)}

    best_scale, best_mean = 0.5, -1e18
    for enter_scale in (0.5, 1.0, 1.5, 2.5):
        report = portfolio_report(booster, val_episodes, costs, enter_scale, "v")
        if report["mean_day_pct"] > best_mean:
            best_mean, best_scale = report["mean_day_pct"], enter_scale
    report_all = portfolio_report(booster, test_episodes, costs, best_scale, "test all")
    val_report = portfolio_report(booster, val_episodes, costs, best_scale, "val all")
    top = [s for s, m in val_report["by_symbol_mean"].items() if m > 0][:8]
    report_top = portfolio_report(booster, {k: e for k, e in test_episodes.items() if k[1] in top}, costs, best_scale, "tt")
    return {
        "name": name,
        "features": train_x.shape[1],
        "val_corr": round(correlation, 4),
        "scale": best_scale,
        "test_all": report_all["mean_day_pct"],
        "test_all_worst": report_all["worst_day_pct"],
        "test_all_pos": report_all["positive_days"],
        "test_top": report_top["mean_day_pct"],
        "seconds": round(time.time() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", required=True)
    parser.add_argument("--val-days", type=int, default=7)
    parser.add_argument("--test-days", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--stride", type=int, default=TRAIN_STRIDE)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    dates = sorted(d.strip() for d in args.dates.split(",") if d.strip())
    test_dates = dates[-args.test_days :]
    val_dates = dates[-args.test_days - args.val_days : -args.test_days]
    train_dates = dates[: -args.test_days - args.val_days]
    print(f"device {DEVICE} | train {len(train_dates)}d | val {len(val_dates)}d | test {len(test_dates)}d", flush=True)

    costs = {}
    for symbol in SYMBOLS:
        spread = effective_spread_bps(symbol, train_dates)
        costs[symbol] = (TAKER_FEE_BPS + spread / 2 + EXTRA_BPS) / 1e4

    baseline, augmented = {}, {}
    for date in dates:
        day = {s: build_episode(s, date) for s in SYMBOLS}
        day = {s: e for s, e in day.items() if e is not None}
        if not day:
            continue
        factor = market_factor(list(day.values()))
        for symbol, episode in day.items():
            baseline[(date, symbol)] = episode
            augmented[(date, symbol)] = augment(episode, factor)
    print(f"episodes {len(baseline)} built", flush=True)

    arms = {"baseline-47": baseline, "augmented-51": augmented}
    runs = {name: [] for name in arms}
    for seed in range(args.seeds):
        for name, episodes in arms.items():
            result = train_arm(name, episodes, train_dates, val_dates, test_dates, costs, args.epochs, args.batch, args.stride, seed)
            runs[name].append(result)
            print(f"  seed {seed} {name}: val_corr {result['val_corr']:+.4f} test_all {result['test_all']:+.3f}% test_top {result['test_top']:+.3f}%", flush=True)

    print(f"\n=== A/B (neural GPU, {args.seeds} seeds, mean±std) ===")
    stats = {}
    for name, results in runs.items():
        agg = {key: np.array([r[key] for r in results], dtype=float) for key in ("val_corr", "test_all", "test_all_worst", "test_top")}
        stats[name] = agg
        print(
            f"{name:>13} | val_corr {agg['val_corr'].mean():+.4f}±{agg['val_corr'].std():.4f} | "
            f"test_all {agg['test_all'].mean():+.3f}±{agg['test_all'].std():.3f}% (worst {agg['test_all_worst'].mean():+.3f}) | "
            f"test_top {agg['test_top'].mean():+.3f}±{agg['test_top'].std():.3f}%"
        )
    print(f"\nновые фичи: {NEW_NAMES}")
    b, a = stats["baseline-47"], stats["augmented-51"]
    print(
        f"Δ val_corr {a['val_corr'].mean() - b['val_corr'].mean():+.4f} | "
        f"Δ test_all {a['test_all'].mean() - b['test_all'].mean():+.3f}% | "
        f"Δ test_top {a['test_top'].mean() - b['test_top'].mean():+.3f}% "
        f"(baseline std {b['test_top'].std():.3f}, aug std {a['test_top'].std():.3f})"
    )


if __name__ == "__main__":
    main()
