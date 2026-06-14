"""A/B: baseline 47 features vs +4 new (market factor / idio residual, rolling
Roll spread, perp/stock vol ratio). Same days, same split, same XGB config —
only the feature matrix differs. Reports val corr and portfolio PnL per arm.
"""

from __future__ import annotations

import argparse

import numpy as np
import xgboost as xgb

from features_v2 import SYMBOLS, build_episode, effective_spread_bps, rolling_std, rolling_sum
from xgb2_train import (
    EXTRA_BPS,
    QUANTILES,
    TAKER_FEE_BPS,
    TRAIN_STRIDE,
    episode_rows,
    portfolio_report,
)

STOCK_RET_3S = 5  # column index of stock_ret_3s
PREMIUM = 14
PERP_VOL_60 = 16
NEW_NAMES = ["market_ret_3s", "idio_ret_3s", "roll_spread_60s", "perp_stock_vol_ratio"]


def roll_spread(perp: np.ndarray, window: int = 60) -> np.ndarray:
    returns = np.zeros(len(perp))
    returns[1:] = (perp[1:] / perp[:-1] - 1) * 100
    lagged = np.zeros(len(perp))
    lagged[1:] = returns[:-1]
    cross = rolling_sum(returns * lagged, window) / window
    mean_now = rolling_sum(returns, window) / window
    mean_lag = rolling_sum(lagged, window) / window
    autocov = cross - mean_now * mean_lag
    return 2.0 * np.sqrt(np.maximum(-autocov, 0.0))


def augment(episode: dict, market_ret_3s: np.ndarray) -> dict:
    features = episode["features"]
    perp = episode["perp"]
    stock = perp * (1 + features[:, PREMIUM] / 100)
    stock_returns = np.zeros(len(perp))
    stock_returns[1:] = (stock[1:] / stock[:-1] - 1) * 100
    stock_vol_60 = rolling_std(stock_returns, 60)
    extra = np.stack(
        [
            market_ret_3s,
            features[:, STOCK_RET_3S] - market_ret_3s,
            roll_spread(perp),
            features[:, PERP_VOL_60] / (stock_vol_60 + 1e-4),
        ],
        axis=1,
    ).astype(np.float32)
    return {"features": np.concatenate([features, extra], axis=1), "perp": perp}


def market_factor(episodes_for_date: list[dict]) -> np.ndarray:
    stacked = np.stack([ep["features"][:, STOCK_RET_3S] for ep in episodes_for_date], axis=0)
    return stacked.mean(axis=0)


def run_arm(name: str, episodes: dict, train_dates, val_dates, test_dates, costs, rounds) -> dict:
    train_blocks_x, train_blocks_y = [], []
    for (date, _symbol), episode in episodes.items():
        if date in train_dates:
            features, targets = episode_rows(episode, TRAIN_STRIDE)
            train_blocks_x.append(features)
            train_blocks_y.append(targets)
    train_x = np.concatenate(train_blocks_x)
    train_y = np.concatenate(train_blocks_y)

    val_episodes = {k: e for k, e in episodes.items() if k[0] in val_dates}
    test_episodes = {k: e for k, e in episodes.items() if k[0] in test_dates}
    val_blocks_x, val_blocks_y = [], []
    for episode in val_episodes.values():
        features, targets = episode_rows(episode, TRAIN_STRIDE)
        val_blocks_x.append(features)
        val_blocks_y.append(targets)
    val_x = np.concatenate(val_blocks_x)
    val_y = np.concatenate(val_blocks_y)

    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=QUANTILES,
        n_estimators=rounds,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=50,
        n_jobs=16,
        early_stopping_rounds=30,
        eval_metric="quantile",
    )
    model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=False)
    booster = model.get_booster()
    val_preds = booster.inplace_predict(val_x)
    correlation = float(np.corrcoef(val_preds[:, 2], val_y)[0, 1])

    best_scale, best_mean = 0.5, -1e18
    for enter_scale in (0.5, 1.0, 1.5, 2.5):
        report = portfolio_report(booster, val_episodes, costs, enter_scale, "v")
        if report["mean_day_pct"] > best_mean:
            best_mean, best_scale = report["mean_day_pct"], enter_scale
    report_all = portfolio_report(booster, test_episodes, costs, best_scale, "test all")
    val_report = portfolio_report(booster, val_episodes, costs, best_scale, "val all")
    top = [s for s, m in val_report["by_symbol_mean"].items() if m > 0][:8]
    report_top = portfolio_report(booster, {k: e for k, e in test_episodes.items() if k[1] in top}, costs, best_scale, "tt")

    importance = booster.get_score(importance_type="gain")
    feature_count = train_x.shape[1]
    new_gain = {
        NEW_NAMES[index - 47]: round(importance.get(f"f{index}", 0.0), 1)
        for index in range(47, feature_count)
    }
    return {
        "name": name,
        "features": feature_count,
        "best_iter": int(model.best_iteration or rounds),
        "val_corr": round(correlation, 4),
        "scale": best_scale,
        "test_all": report_all["mean_day_pct"],
        "test_all_worst": report_all["worst_day_pct"],
        "test_all_pos": report_all["positive_days"],
        "test_top": report_top["mean_day_pct"],
        "new_gain": new_gain,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", required=True)
    parser.add_argument("--val-days", type=int, default=3)
    parser.add_argument("--test-days", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=600)
    args = parser.parse_args()
    dates = sorted(d.strip() for d in args.dates.split(",") if d.strip())
    test_dates = dates[-args.test_days :]
    val_dates = dates[-args.test_days - args.val_days : -args.test_days]
    train_dates = dates[: -args.test_days - args.val_days]
    print(f"train {len(train_dates)}d | val {val_dates} | test {test_dates}", flush=True)

    costs = {}
    for symbol in SYMBOLS:
        spread = effective_spread_bps(symbol, train_dates)
        costs[symbol] = (TAKER_FEE_BPS + spread / 2 + EXTRA_BPS) / 1e4

    baseline: dict[tuple[str, str], dict] = {}
    augmented: dict[tuple[str, str], dict] = {}
    for date in dates:
        day_episodes = {}
        for symbol in SYMBOLS:
            episode = build_episode(symbol, date)
            if episode is not None:
                day_episodes[symbol] = episode
        if not day_episodes:
            continue
        factor = market_factor(list(day_episodes.values()))
        for symbol, episode in day_episodes.items():
            baseline[(date, symbol)] = episode
            augmented[(date, symbol)] = augment(episode, factor)
    print(f"episodes: {len(baseline)} | building done", flush=True)

    base = run_arm("baseline-47", baseline, train_dates, val_dates, test_dates, costs, args.rounds)
    aug = run_arm("augmented-51", augmented, train_dates, val_dates, test_dates, costs, args.rounds)

    print("\n=== A/B ===")
    for arm in (base, aug):
        print(
            f"{arm['name']:>13} | feats {arm['features']} iter {arm['best_iter']:>3} | "
            f"val_corr {arm['val_corr']:+.4f} | test_all {arm['test_all']:+.3f}% "
            f"(worst {arm['test_all_worst']:+.3f}, {arm['test_all_pos']}) | test_top {arm['test_top']:+.3f}%"
        )
    print(f"\nновые фичи (gain в augmented арме): {aug['new_gain']}")
    d_corr = aug["val_corr"] - base["val_corr"]
    d_all = aug["test_all"] - base["test_all"]
    d_top = aug["test_top"] - base["test_top"]
    print(f"Δ val_corr {d_corr:+.4f} | Δ test_all {d_all:+.3f}% | Δ test_top {d_top:+.3f}%")


if __name__ == "__main__":
    main()
