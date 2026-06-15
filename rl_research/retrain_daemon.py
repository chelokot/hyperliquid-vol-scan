"""Hourly seamless retrainer. Rebuilds the ensemble on the full archive PLUS the
live bars accumulated in the store (today), writes the bundle ATOMICALLY
(temp + os.replace) so the running engine hot-swaps it between decisions. Runs at
low priority; GPU training does not contend with the engine's tiny CPU inference.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from features_v2 import (
    FEATURE_NAMES, OUT_DIR, SYMBOLS, WARMUP, build_episode, compute_features,
    effective_spread_bps, forward_fill,
)
from nn_train import TorchQuantiles
from train_production import EnsembleQuantiles, ensemble_portfolio, train_one
from xgb2_train import EXTRA_BPS, FORWARD_SECONDS, QUANTILES, TAKER_FEE_BPS, TRAIN_STRIDE, episode_rows
from live_store import LiveStore

BUNDLE = OUT_DIR / "production_ensemble.pt"


def build_live_episode(store: LiveStore, date: str, symbol: str):
    rows = store.load_bars(date, symbol)
    if len(rows) < WARMUP + FORWARD_SECONDS + 200:
        return None
    n = rows[-1][0] + 1
    perp = np.full(n, np.nan); stock = np.full(n, np.nan)
    buy = np.zeros(n); sell = np.zeros(n); pcount = np.zeros(n); scount = np.zeros(n)
    for sec, pl, pb, ps, pc, sl, sc in rows:
        if pl is not None:
            perp[sec] = pl
        if sl is not None:
            stock[sec] = sl
        buy[sec] = pb or 0.0; sell[sec] = ps or 0.0; pcount[sec] = pc or 0; scount[sec] = sc or 0
    if np.isnan(perp).all() or np.isnan(stock).all():
        return None
    perp = forward_fill(perp); stock = forward_fill(stock)
    feats = compute_features(symbol, perp, stock, buy, sell, pcount, scount, np.arange(n, dtype=np.float64))
    return {"features": feats, "perp": perp}


def retrain_once(seeds: int, epochs: int) -> None:
    dates = sorted(os.listdir(OUT_DIR / "hl_archive" / "trades"))
    test_dates = dates[-12:]
    val_dates = dates[-19:-12]
    train_dates = dates[:-19]
    costs = {s: (TAKER_FEE_BPS + effective_spread_bps(s, train_dates) / 2 + EXTRA_BPS) / 1e4 for s in SYMBOLS}

    xs, ys = [], []
    for d in train_dates:
        for s in SYMBOLS:
            e = build_episode(s, d)
            if e is None:
                continue
            fx, fy = episode_rows(e, TRAIN_STRIDE)
            xs.append(fx); ys.append(fy)
    # live data from the store (today)
    store = LiveStore()
    today = time.strftime("%Y%m%d", time.gmtime())
    live_used = []
    for s in SYMBOLS:
        e = build_live_episode(store, today, s)
        if e is None:
            continue
        fx, fy = episode_rows(e, TRAIN_STRIDE)
        if len(fx):
            xs.append(fx); ys.append(fy); live_used.append(s)
    train_x = np.concatenate(xs); train_y = np.concatenate(ys)

    val_episodes = {}
    for d in val_dates:
        for s in SYMBOLS:
            e = build_episode(s, d)
            if e is not None:
                val_episodes[(d, s)] = e
    vxs, vys = [], []
    for e in val_episodes.values():
        fx, fy = episode_rows(e, TRAIN_STRIDE)
        vxs.append(fx); vys.append(fy)
    val_x, val_y = np.concatenate(vxs), np.concatenate(vys)

    mean, std = train_x.mean(0), train_x.std(0) + 1e-6
    y_mean, y_std = float(train_y.mean()), float(train_y.std() + 1e-6)
    from nn_train import DEVICE
    tx = torch.from_numpy(((train_x - mean) / std).astype(np.float32)).to(DEVICE)
    ty = torch.from_numpy(((train_y - y_mean) / y_std).astype(np.float32)).to(DEVICE)
    vx = torch.from_numpy(((val_x - mean) / std).astype(np.float32)).to(DEVICE)
    vy = torch.from_numpy(((val_y - y_mean) / y_std).astype(np.float32)).to(DEVICE)

    models = [train_one(tx, ty, vx, vy, train_x.shape[1], epochs, 16384, seed) for seed in range(seeds)]
    members = [TorchQuantiles(m, mean, std, y_mean, y_std) for m in models]

    best_scale, best = 0.5, -1e18
    for scale in (0.5, 1.0, 1.5, 2.5, 4.0, 6.0):
        r = ensemble_portfolio(members, val_episodes, costs, scale)["mean_day_pct"]
        if r > best:
            best, best_scale = r, scale
    val_corr = float(np.corrcoef(EnsembleQuantiles(members).inplace_predict(val_x)[:, 2], val_y)[0, 1])

    bundle = {
        "states": [m.state_dict() for m in models],
        "mean": mean, "std": std, "y_mean": y_mean, "y_std": y_std,
        "n_features": int(train_x.shape[1]), "feature_names": FEATURE_NAMES,
        "symbols": SYMBOLS, "quantiles": list(QUANTILES), "enter_scale": float(best_scale),
        "costs_bps": {s: round(c * 1e4, 3) for s, c in costs.items()},
    }
    tmp = str(BUNDLE) + ".tmp"
    torch.save(bundle, tmp)
    os.replace(tmp, BUNDLE)  # atomic -> engine hot-swaps
    summary = {"val_corr": round(val_corr, 4), "scale": best_scale, "rows": int(len(train_x)), "live_symbols": live_used}
    store.record_model(str(BUNDLE), summary)
    print(f"[{time.strftime('%H:%M:%S')}] retrained: {summary}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=3600)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    try:
        os.nice(10)  # low priority so it never starves the trading engine
    except Exception:
        pass
    while True:
        started = time.time()
        try:
            retrain_once(args.seeds, args.epochs)
        except Exception as exc:
            print(f"retrain failed (keeping current model): {exc}", flush=True)
        if args.once:
            break
        time.sleep(max(0.0, args.interval - (time.time() - started)))


if __name__ == "__main__":
    main()
