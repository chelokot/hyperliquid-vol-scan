"""GPU neural quantile regression (PyTorch-ROCm) over our features.

Predicts the same 5 forward-return quantiles as the GBDT, trained on the AMD
GPU. Exposes inplace_predict so the existing portfolio backtest is reused
unchanged, and logs live to mneme for a head-to-head with XGBoost/LightGBM.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from features_v2 import OUT_DIR, SYMBOLS, WARMUP, build_episode, effective_spread_bps
from xgb2_train import (
    EXTRA_BPS,
    QUANTILES,
    TAKER_FEE_BPS,
    TRAIN_STRIDE,
    episode_rows,
    portfolio_report,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUANTS = torch.tensor(QUANTILES, dtype=torch.float32, device=DEVICE)


FEATURE_CLIP = 6.0


class QuantileMLP(nn.Module):
    """MLP with a monotone quantile head: predicts q[0] then non-negative gaps."""

    def __init__(self, in_features: int, hidden: int = 256, n_quants: int = 5, dropout: float = 0.2) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
        )
        self.base = nn.Linear(hidden // 2, 1)
        self.gaps = nn.Linear(hidden // 2, n_quants - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(-FEATURE_CLIP, FEATURE_CLIP)
        h = self.body(x)
        base = self.base(h)
        gaps = torch.nn.functional.softplus(self.gaps(h))
        return torch.cat([base, base + torch.cumsum(gaps, dim=1)], dim=1)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = target.unsqueeze(1) - pred
    return torch.maximum(QUANTS * error, (QUANTS - 1) * error).mean()


class TorchQuantiles:
    """xgboost-booster-like interface for the portfolio backtest."""

    def __init__(self, model: nn.Module, mean: np.ndarray, std: np.ndarray, y_mean: float, y_std: float) -> None:
        self.model = model.eval()
        self.mean = mean
        self.std = std
        self.y_mean = y_mean
        self.y_std = y_std

    def inplace_predict(self, features: np.ndarray) -> np.ndarray:
        normalized = (features - self.mean) / self.std
        out = np.empty((len(features), len(QUANTILES)), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(features), 200_000):
                batch = torch.from_numpy(normalized[start : start + 200_000]).to(DEVICE)
                out[start : start + 200_000] = self.model(batch).cpu().numpy()
        return out * self.y_std + self.y_mean


def load_matrix(dates: list[str], stride: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for date in dates:
        for symbol in SYMBOLS:
            episode = build_episode(symbol, date)
            if episode is None:
                continue
            features, targets = episode_rows(episode, stride)
            xs.append(features)
            ys.append(targets)
    return np.concatenate(xs), np.concatenate(ys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", required=True)
    parser.add_argument("--val-days", type=int, default=7)
    parser.add_argument("--test-days", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--stride", type=int, default=TRAIN_STRIDE)
    parser.add_argument("--mneme", action="store_true")
    parser.add_argument("--mneme-name", default="nn-gpu")
    args = parser.parse_args()
    print(f"device: {DEVICE}", flush=True)
    dates = sorted(d.strip() for d in args.dates.split(",") if d.strip())
    test_dates = dates[-args.test_days :]
    val_dates = dates[-args.test_days - args.val_days : -args.test_days]
    train_dates = dates[: -args.test_days - args.val_days]

    tracker = None
    if args.mneme:
        sys.path.insert(0, str(OUT_DIR.parents[3] / "mneme" / "client"))
        import mneme

        tracker = mneme.init(
            project="trading-xgb",
            name=args.mneme_name,
            config={"engine": "torch-mlp", "device": DEVICE, "epochs": args.epochs, "batch": args.batch},
            tags=["neural", "gpu", "quantile"],
        )
        print(f"mneme run: {tracker.url}", flush=True)

    costs = {}
    for symbol in SYMBOLS:
        spread = effective_spread_bps(symbol, train_dates)
        costs[symbol] = (TAKER_FEE_BPS + spread / 2 + EXTRA_BPS) / 1e4

    train_x, train_y = load_matrix(train_dates, args.stride)
    val_x, val_y = load_matrix(val_dates, args.stride)
    print(f"rows: train {len(train_x)}, val {len(val_x)} (stride {args.stride})", flush=True)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0) + 1e-6

    y_mean = float(train_y.mean())
    y_std = float(train_y.std() + 1e-6)
    tx = torch.from_numpy(((train_x - mean) / std).astype(np.float32)).to(DEVICE)
    ty = torch.from_numpy(((train_y - y_mean) / y_std).astype(np.float32)).to(DEVICE)
    vx = torch.from_numpy(((val_x - mean) / std).astype(np.float32)).to(DEVICE)
    vy = torch.from_numpy(((val_y - y_mean) / y_std).astype(np.float32)).to(DEVICE)

    val_episodes: dict[tuple[str, str], dict] = {}
    test_episodes: dict[tuple[str, str], dict] = {}
    for date in val_dates + test_dates:
        for symbol in SYMBOLS:
            episode = build_episode(symbol, date)
            if episode is None:
                continue
            (val_episodes if date in val_dates else test_episodes)[(date, symbol)] = episode

    model = QuantileMLP(train_x.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    n = len(tx)
    started = time.time()
    best_val = 1e18
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        total = 0.0
        for start in range(0, n, args.batch):
            idx = perm[start : start + args.batch]
            optimizer.zero_grad()
            loss = pinball_loss(model(tx[idx]), ty[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        scheduler.step()
        train_loss = total / n
        model.eval()
        with torch.no_grad():
            val_loss = pinball_loss(model(vx), vy).item()
        if tracker:
            tracker.log({"train/pinball": train_loss, "val/pinball": val_loss}, step=epoch)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch}: train {train_loss:.4f} val {val_loss:.4f}", flush=True)
    train_time = time.time() - started
    print(f"\nобучение на {DEVICE}: {train_time:.1f}с ({args.epochs} эпох)", flush=True)
    if best_state:
        model.load_state_dict(best_state)

    booster = TorchQuantiles(model, mean, std, y_mean, y_std)
    val_preds = booster.inplace_predict(val_x)
    coverage = {
        f"q{int(a * 100)}": round(float((val_y <= val_preds[:, i]).mean()), 3)
        for i, a in enumerate(QUANTILES)
    }
    correlation = float(np.corrcoef(val_preds[:, 2], val_y)[0, 1])
    print(f"калибровка: {coverage}")
    print(f"val correlation(q50, forward): {correlation:.4f}")

    best_scale, best_mean = 0.5, -1e18
    for enter_scale in (0.5, 1.0, 1.5, 2.5):
        report = portfolio_report(booster, val_episodes, costs, enter_scale, "v")
        if report["mean_day_pct"] > best_mean:
            best_mean, best_scale = report["mean_day_pct"], enter_scale
    report_all = portfolio_report(booster, test_episodes, costs, best_scale, "test all")
    val_report = portfolio_report(booster, val_episodes, costs, best_scale, "val all")
    top = [s for s, m in val_report["by_symbol_mean"].items() if m > 0][:8]
    report_top = portfolio_report(booster, {k: e for k, e in test_episodes.items() if k[1] in top}, costs, best_scale, "tt")
    import json

    print(f"test все пары: {report_all['mean_day_pct']}%/день {report_all['positive_days']}")
    print(f"test top-8:   {report_top['mean_day_pct']}%/день {report_top['positive_days']}")

    if tracker:
        tracker.log({f"calib/{k}": v for k, v in coverage.items()}, step=args.epochs)
        tracker.finish(
            summary={
                "val_corr": correlation,
                "test_all_mean_day_pct": report_all["mean_day_pct"],
                "test_top_mean_day_pct": report_top["mean_day_pct"],
                "train_seconds": round(train_time, 1),
            }
        )


if __name__ == "__main__":
    main()
