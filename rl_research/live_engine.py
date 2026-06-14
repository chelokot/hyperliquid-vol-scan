"""Live trading engine for the stock/perp dislocation-reversion ensemble.

Single process:
  HL trades websocket (perp)  +  Finnhub trades websocket (stock)
    -> per-second aggregation -> live_features.FeatureStream (SAME compute_features as backtest)
    -> per-member hysteresis (xgb2_train.step_position, SAME as backtest)
    -> averaged target position -> position reconciliation via HL market orders.

Safe by default: without --live it runs in dry-run (logs intended orders, places
nothing). Stock leg uses Finnhub realtime trades (FINNHUB_API_KEY); perp uses the
Hyperliquid SDK with the API wallet (HYPERLIQUID_SECRET_KEY).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import websockets
import yaml

from features_v2 import PAIRS, RTH_START_UTC, SECONDS, OUT_DIR
from live_features import FeatureStream
from nn_train import QuantileMLP
from xgb2_train import step_position

FINNHUB_WS = "wss://ws.finnhub.io"
FLATTEN_BEFORE_CLOSE_SEC = 120  # stop opening and flatten this long before the close


@dataclass
class EngineConfig:
    account_address: str
    leverage: dict[str, int]
    base_url: str = "https://api.hyperliquid.xyz"
    secret_key_env: str = "HYPERLIQUID_SECRET_KEY"
    finnhub_key_env: str = "FINNHUB_API_KEY"
    budget_safety: float = 0.9  # fraction of equity used as margin (headroom vs liquidation)
    rebalance_fraction: float = 0.34  # only adjust when |target-current| exceeds this share of the symbol notional
    min_order_notional_usd: float = 11.0
    bundle_path: str = str(OUT_DIR / "production_ensemble.pt")
    events_path: str = str(OUT_DIR / "live_engine_events.jsonl")
    symbols: list[str] = field(default_factory=list)


class SecondBar:
    """Thread-safe per-symbol accumulator for the current second."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.perp_last: float | None = None
        self.stock_last: float | None = None
        self.perp_buy = 0.0
        self.perp_sell = 0.0
        self.perp_count = 0
        self.stock_count = 0

    def on_perp_trade(self, price: float, size: float, is_buy: bool) -> None:
        with self.lock:
            self.perp_last = price
            if is_buy:
                self.perp_buy += price * size
            else:
                self.perp_sell += price * size
            self.perp_count += 1

    def on_stock_trade(self, price: float) -> None:
        with self.lock:
            self.stock_last = price
            self.stock_count += 1

    def roll(self) -> tuple[float | None, float, float, int, float | None, int]:
        with self.lock:
            snapshot = (self.perp_last, self.perp_buy, self.perp_sell, self.perp_count, self.stock_last, self.stock_count)
            self.perp_buy = self.perp_sell = 0.0
            self.perp_count = self.stock_count = 0
            return snapshot


class Ensemble:
    """CPU inference: K MLPs, each casts a hysteresis vote; the engine averages them."""

    def __init__(self, bundle: dict) -> None:
        self.mean = bundle["mean"].astype(np.float32)
        self.std = bundle["std"].astype(np.float32)
        self.y_mean = bundle["y_mean"]
        self.y_std = bundle["y_std"]
        self.models = []
        for state in bundle["states"]:
            model = QuantileMLP(bundle["n_features"])
            model.load_state_dict(state)
            self.models.append(model.cpu().eval())

    def quantiles(self, features: np.ndarray) -> list[np.ndarray]:
        normalized = torch.from_numpy(((features - self.mean) / self.std).astype(np.float32)).reshape(1, -1)
        out = []
        with torch.no_grad():
            for model in self.models:
                out.append(model(normalized).numpy()[0] * self.y_std + self.y_mean)
        return out


class LiveEngine:
    def __init__(self, config: EngineConfig, live: bool) -> None:
        self.config = config
        self.live = live
        self.symbols = config.symbols
        self.coin = {s: PAIRS[s] for s in self.symbols}
        self.dex = {s: PAIRS[s].split(":")[0] for s in self.symbols}
        self.bars = {s: SecondBar() for s in self.symbols}
        self.streams = {s: FeatureStream(s) for s in self.symbols}
        self.member_positions = {s: None for s in self.symbols}  # set on first ready tick
        self.local_szi = {s: 0.0 for s in self.symbols}
        self.last_target = {s: 0.0 for s in self.symbols}
        self.events_path = Path(config.events_path)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

        bundle = torch.load(config.bundle_path, weights_only=False)
        self.ensemble = Ensemble(bundle)
        self.enter_scale = bundle["enter_scale"]
        self.costs_bps = bundle["costs_bps"]
        self.symbol_notional: dict[str, float] = {}

        import eth_account
        from hyperliquid.exchange import Exchange
        from hyperliquid.info import Info

        dexes = sorted({"", *self.dex.values()})
        account = eth_account.Account.from_key(os.environ[config.secret_key_env])
        self.info = Info(config.base_url, perp_dexs=dexes)
        self.exchange = Exchange(account, config.base_url, account_address=config.account_address, perp_dexs=dexes)
        self.size_decimals = {s: self.info.asset_to_sz_decimals[self.info.name_to_asset(self.coin[s])] for s in self.symbols}
        self.finnhub_key = os.environ[config.finnhub_key_env]

    def emit(self, event: dict) -> None:
        event["time_ms"] = int(time.time() * 1000)
        with self.events_path.open("a") as handle:
            handle.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)

    # ---- equity / positions ----
    def refresh_account(self) -> None:
        # cash and xyz are SEPARATE builder dexes with SEPARATE margin: size each
        # symbol only from the equity actually deposited in its own dex.
        dex_equity: dict[str, float] = {}
        for dex in sorted(set(self.dex.values())):
            state = self.info.post("/info", {"type": "clearinghouseState", "user": self.config.account_address, "dex": dex})
            dex_equity[dex] = float(state["marginSummary"]["accountValue"])
            for asset_position in state["assetPositions"]:
                coin = asset_position["position"]["coin"]
                for symbol in self.symbols:
                    if self.coin[symbol] == coin:
                        self.local_szi[symbol] = float(asset_position["position"]["szi"])
        dex_counts = {dex: sum(1 for s in self.symbols if self.dex[s] == dex) for dex in dex_equity}
        shares = {}
        for dex, equity in dex_equity.items():
            if equity == 0.0 and not self.live:
                equity = 14.0 * dex_counts[dex]  # nominal so dry-run shows intended sizing before funding
            shares[dex] = equity * self.config.budget_safety / dex_counts[dex]
        self.symbol_notional = {s: shares[self.dex[s]] * self.config.leverage[s] for s in self.symbols}
        self.emit({"type": "account", "dex_equity": {d: round(e, 2) for d, e in dex_equity.items()},
                   "share_margin": {d: round(s, 2) for d, s in shares.items()}})

    # ---- websocket feeds ----
    def start_perp_feed(self) -> None:
        def handler(message: dict) -> None:
            for trade in message.get("data", []):
                symbol = next((s for s in self.symbols if self.coin[s] == trade["coin"]), None)
                if symbol is None:
                    continue
                self.bars[symbol].on_perp_trade(float(trade["px"]), float(trade["sz"]), trade["side"] == "B")

        for symbol in self.symbols:
            self.info.subscribe({"type": "trades", "coin": self.coin[symbol]}, handler)

    def start_stock_feed(self) -> None:
        thread = threading.Thread(target=lambda: asyncio.run(self._stock_loop()), daemon=True)
        thread.start()

    async def _stock_loop(self) -> None:
        url = f"{FINNHUB_WS}?token={self.finnhub_key}"
        while True:
            try:
                async with websockets.connect(url) as ws:
                    for symbol in self.symbols:
                        await ws.send(json.dumps({"type": "subscribe", "symbol": symbol}))
                    async for raw in ws:
                        message = json.loads(raw)
                        if message.get("type") != "trade":
                            continue
                        for trade in message["data"]:
                            if trade["s"] in self.bars:
                                self.bars[trade["s"]].on_stock_trade(float(trade["p"]))
            except Exception as exc:
                self.emit({"type": "stock_feed_error", "error": str(exc)})
                await asyncio.sleep(3)

    # ---- decision + execution ----
    def decide(self, symbol: str) -> float | None:
        features = self.streams[symbol].latest_features()
        if features is None:
            return None
        quantiles = self.ensemble.quantiles(features)
        enter_bps = self.enter_scale * self.costs_bps[symbol]
        if self.member_positions[symbol] is None:
            self.member_positions[symbol] = [0.0] * len(quantiles)
        votes = []
        for index, quantile in enumerate(quantiles):
            new = step_position(self.member_positions[symbol][index], quantile[1], quantile[2], quantile[3], enter_bps)
            self.member_positions[symbol][index] = new
            votes.append(new)
        return float(np.mean(votes))

    def reconcile(self, symbol: str, target_position: float, perp_price: float) -> None:
        target_notional = target_position * self.symbol_notional[symbol]
        current_notional = self.local_szi[symbol] * perp_price
        delta = target_notional - current_notional
        threshold = max(self.config.min_order_notional_usd, self.config.rebalance_fraction * self.symbol_notional[symbol])
        if abs(delta) < threshold:
            return
        size = math.floor(abs(delta) / perp_price * 10 ** self.size_decimals[symbol]) / 10 ** self.size_decimals[symbol]
        if size * perp_price < self.config.min_order_notional_usd:
            return
        is_buy = delta > 0
        event = {"type": "order", "symbol": symbol, "side": "buy" if is_buy else "sell", "size": size,
                 "perp": perp_price, "target_pos": round(target_position, 2), "live": self.live}
        if self.live:
            result = self.exchange.market_open(self.coin[symbol], is_buy, size)
            event["result_status"] = str(result.get("status"))
        self.local_szi[symbol] += size if is_buy else -size
        self.emit(event)

    def flatten_all(self, reason: str) -> None:
        for symbol in self.symbols:
            if abs(self.local_szi[symbol]) < 1e-9:
                continue
            if self.live:
                self.exchange.market_close(self.coin[symbol])
            self.emit({"type": "flatten", "symbol": symbol, "szi": self.local_szi[symbol], "reason": reason, "live": self.live})
            self.local_szi[symbol] = 0.0
            self.member_positions[symbol] = None

    # ---- main loop ----
    def session_second(self) -> int:
        return int(time.time()) % 86400 - RTH_START_UTC

    def run(self) -> None:
        self.refresh_account()
        if self.live:
            for symbol in self.symbols:
                try:
                    self.exchange.update_leverage(self.config.leverage[symbol], self.coin[symbol], is_cross=True)
                except Exception as exc:
                    self.emit({"type": "leverage_error", "symbol": symbol, "requested": self.config.leverage[symbol], "error": str(exc)})
        self.start_perp_feed()
        self.start_stock_feed()
        self.emit({"type": "started", "live": self.live, "symbols": self.symbols, "enter_scale": self.enter_scale})

        last_account_refresh = time.time()
        flattened = False
        while True:
            tick_started = time.time()
            second = self.session_second()
            in_session = 0 <= second < SECONDS
            tradeable = in_session and second < SECONDS - FLATTEN_BEFORE_CLOSE_SEC

            for symbol in self.symbols:
                perp_last, perp_buy, perp_sell, perp_count, stock_last, stock_count = self.bars[symbol].roll()
                self.streams[symbol].push_second(second, perp_last, perp_buy, perp_sell, perp_count, stock_last, stock_count)

            if tradeable:
                flattened = False
                if tick_started - last_account_refresh >= 60:
                    self.refresh_account()
                    last_account_refresh = tick_started
                for symbol in self.symbols:
                    target = self.decide(symbol)
                    perp_price = self.bars[symbol].perp_last
                    if target is not None:
                        self.last_target[symbol] = target
                        if perp_price:
                            self.reconcile(symbol, target, perp_price)
            elif in_session and not flattened:
                self.flatten_all("near_close")
                flattened = True

            if second % 60 == 0:
                self.emit({"type": "heartbeat", "session_second": second, "in_session": in_session,
                           "ready": sum(self.streams[s].ready() for s in self.symbols),
                           "targets": {s: round(t, 2) for s, t in self.last_target.items() if abs(t) > 1e-9},
                           "positions": {s: round(v, 4) for s, v in self.local_szi.items() if abs(v) > 1e-9}})
            time.sleep(max(0.0, 1.0 - (time.time() - tick_started)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--live", action="store_true", help="place real orders (default: dry-run, logs only)")
    args = parser.parse_args()
    config = EngineConfig(**yaml.safe_load(Path(args.config).read_text()))
    LiveEngine(config, live=args.live).run()


if __name__ == "__main__":
    main()
