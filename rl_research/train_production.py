"""Train the production seed-ensemble on the new SOTA features (features_v2),
report the honest canonical metrics, and save a self-contained bundle the live
engine loads. The ensemble averages K independently-seeded MLPs to collapse the
seed variance we measured earlier."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from features_v2 import FEATURE_NAMES, OUT_DIR, SYMBOLS, WARMUP, build_episode, effective_spread_bps
from nn_train import DEVICE, QuantileMLP, TorchQuantiles, pinball_loss
from xgb2_train import DELAY, EXTRA_BPS, QUANTILES, TAKER_FEE_BPS, TRAIN_STRIDE, episode_rows, portfolio_report, quantile_positions


class EnsembleQuantiles:
    def __init__(self, members: list[TorchQuantiles]) -> None:
        self.members = members

    def inplace_predict(self, features: np.ndarray) -> np.ndarray:
        return np.mean([member.inplace_predict(features) for member in self.members], axis=0)


def ensemble_episode_pnl(members, episode, scale, cost) -> float:
    """Decision-level ensemble: each member votes a position from its OWN
    quantiles, then we average the positions. Preserves each model's signal
    strength (unlike averaging predictions, which shrinks the quantile spread)."""
    features = episode["features"][WARMUP:]
    perp = episode["perp"][WARMUP:]
    enter = scale * cost * 1e4
    member_positions = [quantile_positions(member.inplace_predict(features), enter) for member in members]
    positions = np.mean(member_positions, axis=0)
    positions = np.concatenate([np.zeros(DELAY), positions[:-DELAY]])
    returns = np.zeros(len(perp))
    returns[:-1] = perp[1:] / perp[:-1] - 1
    changes = np.abs(np.diff(positions, prepend=0.0))
    return float((positions * returns - cost * changes).sum() * 100)


def ensemble_portfolio(members, episodes, costs, scale) -> dict:
    by_day: dict[str, list[float]] = {}
    by_symbol: dict[str, list[float]] = {}
    for (date, symbol), episode in episodes.items():
        pnl = ensemble_episode_pnl(members, episode, scale, costs[symbol])
        by_day.setdefault(date, []).append(pnl)
        by_symbol.setdefault(symbol, []).append(pnl)
    daily = [float(np.mean(values)) for values in by_day.values()]
    return {
        "mean_day_pct": round(float(np.mean(daily)), 3),
        "worst_day_pct": round(float(np.min(daily)), 3),
        "positive_days": f"{sum(1 for d in daily if d > 0)}/{len(daily)}",
        "by_symbol_mean": {s: float(np.mean(v)) for s, v in sorted(by_symbol.items(), key=lambda kv: -np.mean(kv[1]))},
    }


def load_matrix(dates, stride):
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


def train_one(tx, ty, vx, vy, in_features, epochs, batch, seed):
    torch.manual_seed(seed)
    model = QuantileMLP(in_features).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n = len(tx)
    best_val, best_state = 1e18, None
    for _ in range(epochs):
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
    return model.eval()


def full_report(booster, episodes, costs, scale, label):
    report = portfolio_report(booster, episodes, costs, scale, label)
    return report["mean_day_pct"], report["worst_day_pct"], report["positive_days"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", required=True)
    parser.add_argument("--val-days", type=int, default=7)
    parser.add_argument("--test-days", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    dates = sorted(d.strip() for d in args.dates.split(",") if d.strip())
    test_dates = dates[-args.test_days :]
    val_dates = dates[-args.test_days - args.val_days : -args.test_days]
    train_dates = dates[: -args.test_days - args.val_days]
    print(f"device {DEVICE} | features {len(FEATURE_NAMES)} | train {len(train_dates)}d val {len(val_dates)}d test {len(test_dates)}d", flush=True)

    costs = {s: (TAKER_FEE_BPS + effective_spread_bps(s, train_dates) / 2 + EXTRA_BPS) / 1e4 for s in SYMBOLS}
    train_x, train_y = load_matrix(train_dates, TRAIN_STRIDE)
    val_x, val_y = load_matrix(val_dates, TRAIN_STRIDE)
    print(f"rows: train {len(train_x)}, val {len(val_x)}", flush=True)

    mean, std = train_x.mean(0), train_x.std(0) + 1e-6
    y_mean, y_std = float(train_y.mean()), float(train_y.std() + 1e-6)
    tx = torch.from_numpy(((train_x - mean) / std).astype(np.float32)).to(DEVICE)
    ty = torch.from_numpy(((train_y - y_mean) / y_std).astype(np.float32)).to(DEVICE)
    vx = torch.from_numpy(((val_x - mean) / std).astype(np.float32)).to(DEVICE)
    vy = torch.from_numpy(((val_y - y_mean) / y_std).astype(np.float32)).to(DEVICE)

    val_episodes, test_episodes = {}, {}
    for date in val_dates + test_dates:
        for symbol in SYMBOLS:
            episode = build_episode(symbol, date)
            if episode is not None:
                (val_episodes if date in val_dates else test_episodes)[(date, symbol)] = episode

    started = time.time()
    members, single_corrs, single_alls = [], [], []
    for seed in range(args.seeds):
        model = train_one(tx, ty, vx, vy, train_x.shape[1], args.epochs, args.batch, seed)
        member = TorchQuantiles(model, mean, std, y_mean, y_std)
        members.append(member)
        preds = member.inplace_predict(val_x)
        single_corrs.append(float(np.corrcoef(preds[:, 2], val_y)[0, 1]))
        best_scale, best = 0.5, -1e18
        for scale in (0.25, 0.35, 0.5, 1.0, 1.5, 2.5):
            r = portfolio_report(member, val_episodes, costs, scale, "v")["mean_day_pct"]
            if r > best:
                best, best_scale = r, scale
        single_alls.append(full_report(member, test_episodes, costs, best_scale, "t")[0])
        print(f"  seed {seed}: val_corr {single_corrs[-1]:+.4f} test_all {single_alls[-1]:+.3f}%", flush=True)

    ens_corr = float(np.corrcoef(EnsembleQuantiles(members).inplace_predict(val_x)[:, 2], val_y)[0, 1])
    best_scale, best = 0.25, -1e18
    for scale in (0.25, 0.35, 0.5, 1.0, 1.5, 2.5):
        r = ensemble_portfolio(members, val_episodes, costs, scale)["mean_day_pct"]
        if r > best:
            best, best_scale = r, scale
    test_report = ensemble_portfolio(members, test_episodes, costs, best_scale)
    ens_all, ens_worst, ens_pos = test_report["mean_day_pct"], test_report["worst_day_pct"], test_report["positive_days"]
    val_report = ensemble_portfolio(members, val_episodes, costs, best_scale)
    top = [s for s, m in val_report["by_symbol_mean"].items() if m > 0][:8]
    top_report = ensemble_portfolio(members, {k: e for k, e in test_episodes.items() if k[1] in top}, costs, best_scale)
    ens_top, ens_top_pos = top_report["mean_day_pct"], top_report["positive_days"]
    print(f"\nобучение {args.seeds} сидов: {time.time()-started:.0f}с", flush=True)

    print(f"\n=== PRODUCTION ENSEMBLE ({args.seeds} seeds, {len(FEATURE_NAMES)} features, decision-level) ===")
    print(f"одиночные сиды:  val_corr {np.mean(single_corrs):+.4f}±{np.std(single_corrs):.4f} | test_all {np.mean(single_alls):+.3f}±{np.std(single_alls):.3f}%")
    print(f"АНСАМБЛЬ:        val_corr {ens_corr:+.4f} | test_all {ens_all:+.3f}% (worst {ens_worst:+.3f}, {ens_pos}) | test_top {ens_top:+.3f}% ({ens_top_pos}) | scale {best_scale}")

    bundle = {
        "states": [member.model.state_dict() for member in members],
        "mean": mean, "std": std, "y_mean": y_mean, "y_std": y_std,
        "n_features": int(train_x.shape[1]), "feature_names": FEATURE_NAMES,
        "symbols": SYMBOLS, "quantiles": list(QUANTILES), "enter_scale": float(best_scale),
        "costs_bps": {s: round(c * 1e4, 3) for s, c in costs.items()}, "top_symbols": top,
    }
    torch.save(bundle, OUT_DIR / "production_ensemble.pt")
    json.dump(
        {"n_features": int(train_x.shape[1]), "seeds": args.seeds, "enter_scale": float(best_scale),
         "test_all_mean_day_pct": round(ens_all, 3), "test_top_mean_day_pct": round(ens_top, 3),
         "test_worst_day_pct": round(ens_worst, 3), "val_corr": round(ens_corr, 4),
         "costs_bps": bundle["costs_bps"], "top_symbols": top},
        (OUT_DIR / "production_config.json").open("w"), indent=1,
    )
    print(f"\nсохранено: {OUT_DIR/'production_ensemble.pt'}")


if __name__ == "__main__":
    main()
