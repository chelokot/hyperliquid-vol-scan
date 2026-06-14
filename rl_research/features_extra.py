"""Candidate features layered on top of the cached 47-column episodes:
market factor / idiosyncratic residual, rolling Roll spread, perp/stock vol
ratio. Shared by the XGBoost and neural A/B harnesses so both test the same
augmentation."""

from __future__ import annotations

import numpy as np

from features_v2 import rolling_std, rolling_sum

STOCK_RET_3S = 5
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


def market_factor(episodes_for_date: list[dict]) -> np.ndarray:
    stacked = np.stack([ep["features"][:, STOCK_RET_3S] for ep in episodes_for_date], axis=0)
    return stacked.mean(axis=0)


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
