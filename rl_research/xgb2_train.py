from __future__ import annotations

import argparse
import json

import numpy as np
import xgboost as xgb

from features_v2 import FEATURE_NAMES, OUT_DIR, SYMBOLS, WARMUP, build_episode, effective_spread_bps

QUANTILES = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
FORWARD_SECONDS = 60
TRAIN_STRIDE = 10
TAKER_FEE_BPS = 4.5  # real HL taker (userCrossRate 0.045%); ~4.32 after 4% referral, kept conservative
EXTRA_BPS = 0.5
DELAY = 1


def episode_rows(episode: dict, stride: int) -> tuple[np.ndarray, np.ndarray]:
    features = episode["features"][WARMUP:]
    perp = episode["perp"][WARMUP:]
    horizon = len(perp) - FORWARD_SECONDS - 1
    target_bps = (perp[FORWARD_SECONDS + 1 :] / perp[1 : -FORWARD_SECONDS] - 1) * 1e4
    return features[:horizon:stride], target_bps[::stride]


def step_position(position: float, q25: float, q50: float, q75: float, enter_bps: float) -> float:
    """One hysteresis decision step, shared by the backtest and the live engine
    so they cannot diverge. Returns the new position in {-1, 0, +1}."""
    if position == 0.0:
        if q25 > enter_bps:
            return 1.0
        if q75 < -enter_bps:
            return -1.0
        return 0.0
    if position == 1.0 and q50 <= 0:
        if q75 < -enter_bps:
            return -1.0
        return 1.0 if q25 > enter_bps else 0.0
    if position == -1.0 and q50 >= 0:
        if q25 > enter_bps:
            return 1.0
        return -1.0 if q75 < -enter_bps else 0.0
    return position


def quantile_positions(quantile_preds: np.ndarray, enter_bps: float) -> np.ndarray:
    q25 = quantile_preds[:, 1]
    q50 = quantile_preds[:, 2]
    q75 = quantile_preds[:, 3]
    positions = np.zeros(len(q50))
    position = 0.0
    for index in range(len(q50)):
        position = step_position(position, q25[index], q50[index], q75[index], enter_bps)
        positions[index] = position
    return positions


def episode_strategy_pnl(model, episode: dict, enter_scale: float, cost: float) -> tuple[float, float, int]:
    features = episode["features"][WARMUP:]
    perp = episode["perp"][WARMUP:]
    quantile_preds = model.inplace_predict(features)
    positions = quantile_positions(quantile_preds, enter_scale * cost * 1e4)
    positions = np.concatenate([np.zeros(DELAY), positions[:-DELAY]])
    returns = np.zeros(len(perp))
    returns[:-1] = perp[1:] / perp[:-1] - 1
    changes = np.abs(np.diff(positions, prepend=0.0))
    pnl = float((positions * returns - cost * changes).sum() * 100)
    return pnl, float((positions != 0).mean() * 100), int((changes > 0).sum())


def portfolio_report(model, episodes: dict, costs: dict[str, float], enter_scale: float, label: str) -> dict:
    by_day: dict[str, list[float]] = {}
    by_symbol: dict[str, list[float]] = {}
    for (date, symbol), episode in episodes.items():
        pnl, _activity, _trades = episode_strategy_pnl(model, episode, enter_scale, costs[symbol])
        by_day.setdefault(date, []).append(pnl)
        by_symbol.setdefault(symbol, []).append(pnl)
    daily = {date: float(np.mean(values)) for date, values in sorted(by_day.items())}
    series = list(daily.values())
    report = {
        "label": label,
        "daily_portfolio_pct": {date: round(value, 3) for date, value in daily.items()},
        "mean_day_pct": round(float(np.mean(series)), 3),
        "worst_day_pct": round(float(np.min(series)), 3),
        "positive_days": f"{sum(1 for value in series if value > 0)}/{len(series)}",
        "by_symbol_mean": {
            symbol: round(float(np.mean(values)), 3) for symbol, values in sorted(by_symbol.items(), key=lambda kv: -np.mean(kv[1]))
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", required=True)
    parser.add_argument("--val-days", type=int, default=3)
    parser.add_argument("--test-days", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument("--mneme", action="store_true", help="log live to the mneme tracker")
    parser.add_argument("--mneme-name", default=None)
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
            config={
                "train_days": len(train_dates),
                "val_days": args.val_days,
                "test_days": args.test_days,
                "rounds": args.rounds,
                "forward_s": FORWARD_SECONDS,
                "stride": TRAIN_STRIDE,
                "quantiles": list(QUANTILES),
                "pairs": len(SYMBOLS),
            },
            tags=["quantile", "perp-forward"],
        )
        print(f"mneme run: {tracker.url}", flush=True)

    costs = {}
    for symbol in SYMBOLS:
        spread = effective_spread_bps(symbol, train_dates)
        costs[symbol] = (TAKER_FEE_BPS + spread / 2 + EXTRA_BPS) / 1e4
    print("издержки (bps за смену позиции):", {s: round(c * 1e4, 1) for s, c in sorted(costs.items())}, flush=True)

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

    callbacks = [tracker.xgb_callback(names=["train", "val"])] if tracker else None
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=QUANTILES,
        n_estimators=args.rounds,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=50,
        n_jobs=16,
        early_stopping_rounds=30,
        eval_metric="quantile",
        callbacks=callbacks,
    )
    eval_set = [(train_x, train_y), (val_x, val_y)] if tracker else [(val_x, val_y)]
    model.fit(train_x, train_y, eval_set=eval_set, verbose=50)

    val_preds = model.get_booster().inplace_predict(val_x)
    coverage = {
        f"q{int(alpha * 100)}": round(float((val_y <= val_preds[:, index]).mean()), 3)
        for index, alpha in enumerate(QUANTILES)
    }
    print(f"\nкалибровка квантилей на val (доля y ниже квантиля, идеал = сам квантиль): {coverage}")
    correlation = float(np.corrcoef(val_preds[:, 2], val_y)[0, 1])
    print(f"val correlation(q50, forward): {correlation:.4f}")

    importance = model.get_booster().get_score(importance_type="gain")
    named = sorted(importance.items(), key=lambda item: -item[1])[:15]
    print("\nтоп фич по gain:")
    for key, gain in named:
        index = int(key[1:]) if key.startswith("f") else None
        name = FEATURE_NAMES[index] if index is not None and index < len(FEATURE_NAMES) else key
        print(f"  {name:>22}: {gain:.1f}")

    booster = model.get_booster()

    best_scale = None
    best_mean = -1e18
    for enter_scale in (0.5, 1.0, 1.5, 2.5):
        report = portfolio_report(booster, val_episodes, costs, enter_scale, f"val scale={enter_scale}")
        marker = ""
        if report["mean_day_pct"] > best_mean:
            best_mean = report["mean_day_pct"]
            best_scale = enter_scale
            marker = " <-"
        print(f"val scale {enter_scale}: средний день {report['mean_day_pct']:+.3f}%, худший {report['worst_day_pct']:+.3f}%{marker}")

    print(f"\n=== ПОРТФЕЛЬ все {len(SYMBOLS)} пар, scale={best_scale} ===")
    report_all = portfolio_report(booster, test_episodes, costs, best_scale, "test all")
    print(json.dumps(report_all, indent=1, ensure_ascii=False))

    val_report = portfolio_report(booster, val_episodes, costs, best_scale, "val all")
    top_symbols = [symbol for symbol, mean in val_report["by_symbol_mean"].items() if mean > 0][:8]
    print(f"\n=== ПОРТФЕЛЬ top-{len(top_symbols)} по validation: {top_symbols} ===")
    test_top = {key: ep for key, ep in test_episodes.items() if key[1] in top_symbols}
    report_top = portfolio_report(booster, test_top, costs, best_scale, "test top")
    print(json.dumps(report_top, indent=1, ensure_ascii=False))

    if tracker:
        final_step = int(model.best_iteration or args.rounds)
        tracker.log({f"calib/{k}": v for k, v in coverage.items()}, step=final_step)
        tracker.log(
            {
                "result/val_corr": correlation,
                "result/test_all_mean_day_pct": report_all["mean_day_pct"],
                "result/test_all_worst_day_pct": report_all["worst_day_pct"],
                "result/test_top_mean_day_pct": report_top["mean_day_pct"],
                "result/enter_scale": float(best_scale),
            },
            step=final_step,
        )
        tracker.finish(
            summary={
                "val_corr": correlation,
                "test_all_mean_day_pct": report_all["mean_day_pct"],
                "test_top_mean_day_pct": report_top["mean_day_pct"],
                "best_iteration": final_step,
            }
        )

    model.save_model(str(OUT_DIR / "xgb2_quantile_model.json"))
    json.dump(
        {"costs_bps": {s: round(c * 1e4, 2) for s, c in costs.items()}, "enter_scale": best_scale, "top_symbols": top_symbols},
        (OUT_DIR / "xgb2_config.json").open("w"),
    )


if __name__ == "__main__":
    main()
