from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from rl_policy_train import OUT_DIR, PolicyShape, build_features, load_ticks, run_episode

DATA_PATH = OUT_DIR / "spcx_leadlag.jsonl"
PROGRESS_PATH = OUT_DIR / "rl_train_progress.jsonl"
WEIGHTS_PATH = OUT_DIR / "rl_policy_weights_best_val.npy"
POPULATION = 48
ELITE_COUNT = 10
TRAIN_FRACTION = 0.7
RELOAD_EVERY = 15
STD_FLOOR = 0.05
STALE_LIMIT = 40


def split_data() -> tuple[list, np.ndarray, list, np.ndarray]:
    ticks = load_ticks(DATA_PATH)
    features = build_features(ticks)
    train_end = int(len(ticks) * TRAIN_FRACTION)
    return ticks[:train_end], features[:train_end], ticks[train_end:], features[train_end:]


def main() -> None:
    shape = PolicyShape()
    rng = np.random.default_rng(int(time.time()))
    mean = np.zeros(shape.param_count)
    std = np.ones(shape.param_count)
    best_train_params = mean.copy()
    best_train_score = -1e18
    best_val_params = mean.copy()
    best_val_score = -1e18
    stale_generations = 0
    generation = 0
    train_ticks, train_features, val_ticks, val_features = split_data()

    while True:
        generation += 1
        if generation % RELOAD_EVERY == 0:
            train_ticks, train_features, val_ticks, val_features = split_data()
            best_train_score, _, _ = run_episode(best_train_params, shape, train_ticks, train_features)
            best_val_score, _, _ = run_episode(best_val_params, shape, val_ticks, val_features)

        samples = rng.normal(mean, std, size=(POPULATION, shape.param_count))
        scored = []
        for candidate in samples:
            train_pnl, train_trades, _ = run_episode(candidate, shape, train_ticks, train_features)
            score = train_pnl - (0.05 if train_trades == 0 else 0.0)
            scored.append((score, train_pnl, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        elites = np.array([candidate for _, _, candidate in scored[:ELITE_COUNT]])
        mean = 0.3 * mean + 0.7 * elites.mean(axis=0)
        std = np.maximum(0.3 * std + 0.7 * elites.std(axis=0), STD_FLOOR)

        improved = False
        if scored[0][1] > best_train_score:
            best_train_score = scored[0][1]
            best_train_params = scored[0][2].copy()
            improved = True
        for _, _, candidate in scored[:ELITE_COUNT]:
            val_pnl, val_trades, _ = run_episode(candidate, shape, val_ticks, val_features)
            if val_pnl > best_val_score and val_trades > 0:
                best_val_score = val_pnl
                best_val_params = candidate.copy()
                np.save(WEIGHTS_PATH, best_val_params)
                improved = True
        stale_generations = 0 if improved else stale_generations + 1
        if stale_generations >= STALE_LIMIT:
            std = np.minimum(std * 3.0, 1.0)
            stale_generations = 0

        val_of_best_train, val_trades_bt, _ = run_episode(best_train_params, shape, val_ticks, val_features)
        record = {
            "time_ms": int(time.time() * 1000),
            "generation": generation,
            "train_ticks": len(train_ticks),
            "val_ticks": len(val_ticks),
            "best_train_pnl": round(best_train_score, 4),
            "val_of_best_train": round(val_of_best_train, 4),
            "best_val_pnl": round(best_val_score, 4),
        }
        with PROGRESS_PATH.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
