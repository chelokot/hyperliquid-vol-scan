"""Isolated engine/device benchmark, logged to mneme so results live in the dashboard."""

from __future__ import annotations

import sys
import time

import numpy as np
import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, "/var/home/chelokot/Documents/Projects/mneme/client")
import mneme

ROWS, FEATURES, ROUNDS = 2_000_000, 47, 100
rng = np.random.default_rng(0)
X = rng.standard_normal((ROWS, FEATURES)).astype(np.float32)
y = (X[:, 0] * 0.3 + X[:, 1] * 0.2 + rng.standard_normal(ROWS) * 0.5).astype(np.float32)
dataset = lgb.Dataset(X, label=y)

CONFIGS = [
    ("xgb-cpu", "xgboost", "cpu", None),
    ("lgb-cpu", "lightgbm", "cpu", None),
    ("lgb-gpu-rocm", "lightgbm", "gpu", 0),
    ("lgb-gpu-rusticl", "lightgbm", "gpu", 1),
]


def run_one(engine: str, device: str, platform: int | None) -> float:
    start = time.time()
    if engine == "xgboost":
        model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=np.array([0.5]),
            n_estimators=ROUNDS,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=50,
            n_jobs=8,
        )
        model.fit(X, y)
    else:
        params = {
            "objective": "quantile",
            "alpha": 0.5,
            "num_leaves": 63,
            "max_depth": 6,
            "learning_rate": 0.05,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "min_child_samples": 50,
            "verbose": -1,
        }
        if device == "gpu":
            params |= {"device": "gpu", "gpu_platform_id": platform, "gpu_device_id": 0}
        else:
            params["num_threads"] = 8
        lgb.train(params, dataset, num_boost_round=ROUNDS)
    return time.time() - start


def main() -> None:
    for name, engine, device, platform in CONFIGS:
        try:
            seconds = run_one(engine, device, platform)
            ms_per_round = seconds / ROUNDS * 1000
            print(f"{name}: {seconds:.1f}s ({ms_per_round:.0f}ms/round)", flush=True)
        except Exception as exc:
            print(f"{name}: FAILED — {str(exc)[:120]}", flush=True)
            continue
        run = mneme.init(
            project="engine-benchmark",
            name=name,
            config={"engine": engine, "device": device, "rows": ROWS, "rounds": ROUNDS},
        )
        run.finish(summary={"seconds": round(seconds, 1), "ms_per_round": round(ms_per_round, 0)})


if __name__ == "__main__":
    main()
