from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np

from features_v2 import OUT_DIR, SYMBOLS, WARMUP, build_episode, effective_spread_bps
from xgb2_train import (
    EXTRA_BPS,
    FORWARD_SECONDS,
    QUANTILES,
    TAKER_FEE_BPS,
    TRAIN_STRIDE,
    episode_rows,
    portfolio_report,
    quantile_positions,
)


class LGBQuantiles:
    """Mimics the xgboost booster interface: inplace_predict -> [N, len(QUANTILES)]."""

    def __init__(self, models: list[lgb.Booster]) -> None:
        self.models = models

    def inplace_predict(self, features: np.ndarray) -> np.ndarray:
        stacked = np.stack([model.predict(features) for model in self.models], axis=1)
        # enforce monotone quantiles (independent models can cross)
        return np.sort(stacked, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", required=True)
    parser.add_argument("--val-days", type=int, default=7)
    parser.add_argument("--test-days", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument("--mneme", action="store_true")
    parser.add_argument("--mneme-name", default="lgb")
    args = parser.parse_args()
    dates = sorted(date.strip() for date in args.dates.split(",") if date.strip())
    test_dates = dates[-args.test_days :]
    val_dates = dates[-args.test_days - args.val_days : -args.test_days]
    train_dates = dates[: -args.test_days - args.val_days]
    print(f"train {len(train_dates)}d | val {val_dates} | test {test_dates}", flush=True)

    tracker = None
    if args.mneme:
        import sys as _sys

        _sys.path.insert(0, str(OUT_DIR.parents[3] / "mneme" / "client"))
        import mneme

        tracker = mneme.init(
            project="trading-xgb",
            name=args.mneme_name,
            config={"engine": "lightgbm", "quantiles": list(QUANTILES), "rounds": args.rounds},
            tags=["lightgbm", "quantile"],
        )
        print(f"mneme run: {tracker.url}", flush=True)

    costs = {}
    for symbol in SYMBOLS:
        spread = effective_spread_bps(symbol, train_dates)
        costs[symbol] = (TAKER_FEE_BPS + spread / 2 + EXTRA_BPS) / 1e4

    feature_blocks, target_blocks = [], []
    for date in train_dates:
        for symbol in SYMBOLS:
            episode = build_episode(symbol, date)
            if episode is None:
                continue
            features, targets = episode_rows(episode, TRAIN_STRIDE)
            feature_blocks.append(features)
            target_blocks.append(targets)
    train_x = np.concatenate(feature_blocks)
    train_y = np.concatenate(target_blocks)
    del feature_blocks, target_blocks

    val_episodes: dict[tuple[str, str], dict] = {}
    test_episodes: dict[tuple[str, str], dict] = {}
    for date in val_dates + test_dates:
        for symbol in SYMBOLS:
            episode = build_episode(symbol, date)
            if episode is None:
                continue
            (val_episodes if date in val_dates else test_episodes)[(date, symbol)] = episode
    val_blocks_x, val_blocks_y = [], []
    for episode in val_episodes.values():
        features, targets = episode_rows(episode, TRAIN_STRIDE)
        val_blocks_x.append(features)
        val_blocks_y.append(targets)
    val_x = np.concatenate(val_blocks_x)
    val_y = np.concatenate(val_blocks_y)
    print(f"rows: train {len(train_x)}, val {len(val_x)}", flush=True)

    train_set = lgb.Dataset(train_x, label=train_y, free_raw_data=False)
    val_set = lgb.Dataset(val_x, label=val_y, reference=train_set, free_raw_data=False)

    started = time.time()
    models = []
    for alpha in QUANTILES:
        params = {
            "objective": "quantile",
            "alpha": float(alpha),
            "num_leaves": 63,
            "max_depth": 6,
            "learning_rate": 0.05,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "min_child_samples": 50,
            "num_threads": 8,
            "verbose": -1,
        }
        callbacks = [lgb.early_stopping(30, verbose=False)]
        if tracker:
            tag = int(alpha * 100)

            def log_round(env, tag=tag):
                if env.evaluation_result_list:
                    tracker.log({f"val/q{tag}": env.evaluation_result_list[0][2]}, step=env.iteration)

            callbacks.append(log_round)
        model = lgb.train(
            params, train_set, num_boost_round=args.rounds, valid_sets=[val_set], callbacks=callbacks
        )
        models.append(model)
        print(f"  q{int(alpha * 100)}: best_iter {model.best_iteration}", flush=True)
    train_time = time.time() - started
    print(f"\nобучение 5 моделей: {train_time:.1f}с", flush=True)

    booster = LGBQuantiles(models)
    val_preds = booster.inplace_predict(val_x)
    coverage = {
        f"q{int(alpha * 100)}": round(float((val_y <= val_preds[:, index]).mean()), 3)
        for index, alpha in enumerate(QUANTILES)
    }
    correlation = float(np.corrcoef(val_preds[:, 2], val_y)[0, 1])
    print(f"калибровка: {coverage}")
    print(f"val correlation(q50, forward): {correlation:.4f}")

    best_scale, best_mean = None, -1e18
    for enter_scale in (0.5, 1.0, 1.5, 2.5):
        report = portfolio_report(booster, val_episodes, costs, enter_scale, f"val {enter_scale}")
        if report["mean_day_pct"] > best_mean:
            best_mean, best_scale = report["mean_day_pct"], enter_scale

    report_all = portfolio_report(booster, test_episodes, costs, best_scale, "test all")
    val_report = portfolio_report(booster, val_episodes, costs, best_scale, "val all")
    top_symbols = [s for s, m in val_report["by_symbol_mean"].items() if m > 0][:8]
    test_top = {k: ep for k, ep in test_episodes.items() if k[1] in top_symbols}
    report_top = portfolio_report(booster, test_top, costs, best_scale, "test top")
    print(f"\n=== ПОРТФЕЛЬ все пары: {json.dumps(report_all, ensure_ascii=False)}")
    print(f"=== ПОРТФЕЛЬ top-8: {json.dumps(report_top, ensure_ascii=False)}")

    if tracker:
        final = int(max(m.best_iteration for m in models))
        tracker.log({f"calib/{k}": v for k, v in coverage.items()}, step=final)
        tracker.log(
            {
                "result/val_corr": correlation,
                "result/test_all_mean_day_pct": report_all["mean_day_pct"],
                "result/test_top_mean_day_pct": report_top["mean_day_pct"],
                "result/train_seconds": train_time,
            },
            step=final,
        )
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
