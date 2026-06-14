"""Live half of the single feature path. A FeatureStream keeps per-second ring
buffers for one symbol and produces the current feature vector by calling the
exact same features_v2.compute_features used by the offline backtest, so live
and training features cannot drift apart."""

from __future__ import annotations

from collections import deque

import numpy as np

from features_v2 import (
    PREMIUM_RANGE_WINDOW,
    PREMIUM_STRETCH_WINDOWS,
    PREMIUM_VOL_WINDOWS,
    PREMIUM_Z_WINDOWS,
    PERP_VOL_WINDOWS,
    PRICE_LAGS,
    PREMIUM_LAGS,
    WARMUP,
    compute_features,
)

# buffer must cover the LONGEST causal window so a mid-session feature value
# matches the backtest exactly (a too-short buffer silently truncates windows)
LONGEST_WINDOW = max(
    max(PRICE_LAGS), max(PREMIUM_LAGS), max(PREMIUM_Z_WINDOWS), max(PERP_VOL_WINDOWS),
    max(PREMIUM_STRETCH_WINDOWS), max(PREMIUM_VOL_WINDOWS), PREMIUM_RANGE_WINDOW,
)
BUFFER_SECONDS = LONGEST_WINDOW + 64


class FeatureStream:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.perp = deque(maxlen=BUFFER_SECONDS)
        self.stock = deque(maxlen=BUFFER_SECONDS)
        self.perp_buy = deque(maxlen=BUFFER_SECONDS)
        self.perp_sell = deque(maxlen=BUFFER_SECONDS)
        self.perp_counts = deque(maxlen=BUFFER_SECONDS)
        self.stock_counts = deque(maxlen=BUFFER_SECONDS)
        self.session_seconds = deque(maxlen=BUFFER_SECONDS)
        self._last_perp: float | None = None
        self._last_stock: float | None = None

    def push_second(
        self,
        session_second: int,
        perp_price: float | None,
        perp_buy: float,
        perp_sell: float,
        perp_count: int,
        stock_price: float | None,
        stock_count: int,
    ) -> None:
        if perp_price is not None:
            self._last_perp = perp_price
        if stock_price is not None:
            self._last_stock = stock_price
        if self._last_perp is None or self._last_stock is None:
            return  # nothing to forward-fill from yet
        self.perp.append(self._last_perp)
        self.stock.append(self._last_stock)
        self.perp_buy.append(perp_buy)
        self.perp_sell.append(perp_sell)
        self.perp_counts.append(perp_count)
        self.stock_counts.append(stock_count)
        self.session_seconds.append(float(session_second))

    def ready(self) -> bool:
        return len(self.perp) > WARMUP

    def latest_features(self) -> np.ndarray | None:
        if not self.ready():
            return None
        matrix = compute_features(
            self.symbol,
            np.array(self.perp),
            np.array(self.stock),
            np.array(self.perp_buy),
            np.array(self.perp_sell),
            np.array(self.perp_counts),
            np.array(self.stock_counts),
            np.array(self.session_seconds),
        )
        return matrix[-1]
