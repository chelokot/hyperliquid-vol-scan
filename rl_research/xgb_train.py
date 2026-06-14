from __future__ import annotations

import argparse

import numpy as np
import xgboost as xgb

from rl_pairs_train import OUT_DIR, WARMUP
from rl_ppo_train import load_split

TRAIN_STRIDE = 3
FEATURE_NAMES = (
    [f"stock_ret_{lag}s" for lag in (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144)]
    + [f"perp_ret_{lag}s" for lag in (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144)]
    + [f"premium_delta_{lag}s" for lag in (10, 30, 60, 120)]
    + ["premium", "perp_vol_60s", "perp_vol_300s", "stock_prints_10s", "perp_prints_10s", "session_frac"]
)


def build_matrix(
    entries: list, stride: int, forward_seconds: int
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, np.ndarray, np.ndarray]]]:
    feature_rows = []
    targets = []
    per_episode = []
    for name, episode in entries:
        features = episode["features"][WARMUP:]
        perp = episode["perp"][WARMUP:]
        horizon = len(perp) - forward_seconds - 1
        forward_bps = (perp[forward_seconds + 1 :] / perp[1 : -forward_seconds] - 1) * 1e4
        usable_features = features[:horizon]
        per_episode.append((name, features, perp))
        feature_rows.append(usable_features[::stride])
        targets.append(forward_bps[::stride])
    return np.concatenate(feature_rows), np.concatenate(targets), per_episode


def hysteresis_positions(predictions: np.ndarray, enter_bps: float, exit_bps: float) -> np.ndarray:
    positions = np.zeros(len(predictions))
    position = 0.0
    for index, prediction in enumerate(predictions):
        if position == 0.0:
            if prediction > enter_bps:
                position = 1.0
            elif prediction < -enter_bps:
                position = -1.0
        elif position == 1.0 and prediction < exit_bps:
            position = 1.0 if prediction > enter_bps else 0.0
            if prediction < -enter_bps:
                position = -1.0
        elif position == -1.0 and prediction > -exit_bps:
            position = -1.0 if prediction < -enter_bps else 0.0
            if prediction > enter_bps:
                position = 1.0
        positions[index] = position
    return positions


def strategy_pnl(
    model: xgb.Booster, per_episode: list, enter_bps: float, exit_bps: float, cost: float, delay: int
) -> tuple[float, float, int]:
    total = 0.0
    activity = 0.0
    trades = 0
    for _name, features, perp in per_episode:
        predictions = model.predict(xgb.DMatrix(features, feature_names=FEATURE_NAMES))
        positions = hysteresis_positions(predictions, enter_bps, exit_bps)
        if delay:
            positions = np.concatenate([np.zeros(delay), positions[:-delay]])
        returns = np.zeros(len(perp))
        returns[:-1] = perp[1:] / perp[:-1] - 1
        changes = np.abs(np.diff(positions, prepend=0.0))
        pnl = (positions * returns - cost * changes).sum() * 100
        total += pnl
        activity += (positions != 0).mean()
        trades += int((changes > 0).sum())
    count = len(per_episode)
    return total / count, activity / count * 100, trades


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", required=True)
    parser.add_argument("--holdout-days", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--forward", type=int, default=60)
    parser.add_argument("--cost", type=float, default=0.0002)
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--split", default="pairs")
    args = parser.parse_args()
    dates = sorted(date.strip() for date in args.dates.split(",") if date.strip())
    train, validation, test, test_pure = load_split(dates, args.holdout_days, args.split)
    print(
        f"episodes: train {len(train)}, val {len(validation)}, test {len(test)} (pure {len(test_pure)}) | "
        f"forward {args.forward}s, cost {args.cost * 1e4:.1f} bps",
        flush=True,
    )

    train_x, train_y, _ = build_matrix(train, TRAIN_STRIDE, args.forward)
    val_x, val_y, val_episodes = build_matrix(validation, TRAIN_STRIDE, args.forward)
    print(f"train rows: {len(train_x)}, val rows: {len(val_x)}", flush=True)
    train_matrix = xgb.DMatrix(train_x, label=train_y, feature_names=FEATURE_NAMES)
    val_matrix = xgb.DMatrix(val_x, label=val_y, feature_names=FEATURE_NAMES)
    params = {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 50,
        "nthread": 8,
        "eval_metric": "rmse",
    }
    model = xgb.train(
        params,
        train_matrix,
        num_boost_round=args.rounds,
        evals=[(train_matrix, "train"), (val_matrix, "val")],
        early_stopping_rounds=30,
        verbose_eval=25,
    )
    correlation = np.corrcoef(model.predict(val_matrix), val_y)[0, 1]
    print(f"\nval correlation(prediction, forward {args.forward}s return): {correlation:.4f}")

    print("\ntop features by gain:")
    importance = model.get_score(importance_type="gain")
    for feature_name, gain in sorted(importance.items(), key=lambda item: -item[1])[:12]:
        print(f"  {feature_name:>20}: {gain:.1f}")

    print("\nhysteresis sweep on validation (pnl %/эпизод, активность, переключений):")
    best_combo = None
    best_pnl = -1e18
    for enter_bps in (5.0, 8.0, 12.0, 18.0, 25.0):
        for exit_bps in (0.0, 2.0, enter_bps / 3):
            pnl, activity, trades = strategy_pnl(model, val_episodes, enter_bps, exit_bps, args.cost, args.delay)
            marker = ""
            if pnl > best_pnl:
                best_pnl, best_combo = pnl, (enter_bps, exit_bps)
                marker = " <-"
            print(
                f"  вход {enter_bps:>5.1f} / выход {exit_bps:>4.1f} bps: {pnl:+7.3f}% | "
                f"activity {activity:5.1f}% | trades {trades}{marker}"
            )

    enter_bps, exit_bps = best_combo
    print(f"\nfinal (вход {enter_bps} / выход {exit_bps} bps, выбрано по validation):")
    for name, entries in (("train", train), ("validation", validation), ("test", test), ("test_pure", test_pure)):
        if not entries:
            continue
        _, _, episodes = build_matrix(entries, TRAIN_STRIDE, args.forward)
        pnl, activity, trades = strategy_pnl(model, episodes, enter_bps, exit_bps, args.cost, args.delay)
        print(f"{name:>10}: {pnl:+.3f}%/эпизод | activity {activity:.1f}% | {trades} переключений")
    model.save_model(str(OUT_DIR / "xgb_model.json"))


if __name__ == "__main__":
    main()
