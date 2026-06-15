"""Live trading engine for the stock/perp dislocation-reversion ensemble.

  HL trades websocket (perp) + Finnhub trades websocket (stock)
    -> per-second aggregation -> live_features.FeatureStream (SAME compute_features as backtest)
    -> per-member hysteresis (xgb2_train.step_position, SAME as backtest)  [1 Hz, lags in absolute seconds]
    -> averaged target -> reconciliation via HL market orders  [polled at 0.5 s for faster fill reaction]

Source of truth: every per-second bar, trade, and state snapshot is persisted to
live_store (sqlite) so restarts re-warm instantly and the dashboard + hourly
retrainer read live data. The trainer writes a new bundle atomically; the engine
hot-swaps it between decisions. Safe by default: dry-run unless --live.
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
from live_store import LiveStore
from nn_train import QuantileMLP
from xgb2_train import step_position

FINNHUB_WS = "wss://ws.finnhub.io"
FLATTEN_BEFORE_CLOSE_SEC = 120
POLL_INTERVAL = 0.5
HOTSWAP_CHECK_SEC = 15


@dataclass
class EngineConfig:
    account_address: str
    leverage: dict[str, int]
    base_url: str = "https://api.hyperliquid.xyz"
    secret_key_env: str = "HYPERLIQUID_SECRET_KEY"
    finnhub_key_env: str = "FINNHUB_API_KEY"
    budget_safety: float = 0.9
    rebalance_fraction: float = 0.34
    min_order_notional_usd: float = 11.0
    bundle_path: str = str(OUT_DIR / "production_ensemble.pt")
    events_path: str = str(OUT_DIR / "live_engine_events.jsonl")
    symbols: list[str] = field(default_factory=list)


class SecondBar:
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

    def roll(self):
        with self.lock:
            snap = (self.perp_last, self.perp_buy, self.perp_sell, self.perp_count, self.stock_last, self.stock_count)
            self.perp_buy = self.perp_sell = 0.0
            self.perp_count = self.stock_count = 0
            return snap


class Ensemble:
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
        self.member_positions = {s: None for s in self.symbols}
        self.local_szi = {s: 0.0 for s in self.symbols}
        self.last_target = {s: 0.0 for s in self.symbols}
        self.last_pred = {s: 0.0 for s in self.symbols}
        self.symbol_notional: dict[str, float] = {}
        self.last_dex_equity: dict[str, float] = {}
        self.events_path = Path(config.events_path)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.store = LiveStore()

        self.bundle_path = config.bundle_path
        self.bundle_mtime = os.path.getmtime(self.bundle_path)
        bundle = torch.load(self.bundle_path, weights_only=False)
        self.ensemble = Ensemble(bundle)
        self.enter_scale = bundle["enter_scale"]
        self.costs_bps = bundle["costs_bps"]

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

    @staticmethod
    def today() -> str:
        return time.strftime("%Y%m%d", time.gmtime())

    # ---- restart-resilient warmup from the store ----
    def warmup_from_store(self) -> None:
        date = self.today()
        loaded = {}
        for s in self.symbols:
            rows = self.store.load_bars(date, s)
            for sec, pl, pb, ps, pc, sl, sc in rows:
                self.streams[s].push_second(sec, pl, pb, ps, pc, sl, sc)
            loaded[s] = len(rows)
        self.emit({"type": "warmup", "bars_loaded": loaded, "ready": {s: self.streams[s].ready() for s in self.symbols}})

    # ---- equity / positions ----
    def refresh_account(self) -> None:
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
                equity = 14.0 * dex_counts[dex]
            shares[dex] = equity * self.config.budget_safety / dex_counts[dex]
        self.symbol_notional = {s: shares[self.dex[s]] * self.config.leverage[s] for s in self.symbols}
        self.last_dex_equity = {d: round(e, 2) for d, e in dex_equity.items()}
        self.emit({"type": "account", "dex_equity": self.last_dex_equity, "share_margin": {d: round(s, 2) for d, s in shares.items()}})

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
        threading.Thread(target=lambda: asyncio.run(self._stock_loop()), daemon=True).start()

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

    # ---- decision (1 Hz) + execution (0.5 s) ----
    def decide(self, symbol: str) -> float | None:
        features = self.streams[symbol].latest_features()
        if features is None:
            return None
        quantiles = self.ensemble.quantiles(features)
        self.last_pred[symbol] = float(np.mean([q[2] for q in quantiles]))
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
        target_notional = target_position * self.symbol_notional.get(symbol, 0.0)
        current_notional = self.local_szi[symbol] * perp_price
        delta = target_notional - current_notional
        threshold = max(self.config.min_order_notional_usd, self.config.rebalance_fraction * self.symbol_notional.get(symbol, 0.0))
        if abs(delta) < threshold:
            return
        size = math.floor(abs(delta) / perp_price * 10 ** self.size_decimals[symbol]) / 10 ** self.size_decimals[symbol]
        if size * perp_price < self.config.min_order_notional_usd:
            return
        is_buy = delta > 0
        result = ""
        if self.live:
            result = str(self.exchange.market_open(self.coin[symbol], is_buy, size).get("status"))
        self.local_szi[symbol] += size if is_buy else -size
        self.store.record_trade(symbol, "buy" if is_buy else "sell", size, perp_price, round(target_position, 3), round(size * perp_price, 2), "order", self.live, result)
        self.emit({"type": "order", "symbol": symbol, "side": "buy" if is_buy else "sell", "size": size, "perp": perp_price, "target_pos": round(target_position, 2), "live": self.live, "result": result})

    def flatten_all(self, reason: str) -> None:
        for symbol in self.symbols:
            if abs(self.local_szi[symbol]) < 1e-9:
                continue
            result = ""
            if self.live:
                result = str(self.exchange.market_close(self.coin[symbol]).get("status"))
            self.store.record_trade(symbol, "close", self.local_szi[symbol], self.bars[symbol].perp_last or 0.0, 0.0, 0.0, "flatten", self.live, result)
            self.emit({"type": "flatten", "symbol": symbol, "szi": self.local_szi[symbol], "reason": reason, "live": self.live})
            self.local_szi[symbol] = 0.0
            self.member_positions[symbol] = None

    # ---- hot-swap ----
    def maybe_hotswap(self) -> None:
        mtime = os.path.getmtime(self.bundle_path)
        if mtime <= self.bundle_mtime:
            return
        try:
            bundle = torch.load(self.bundle_path, weights_only=False)
            self.ensemble = Ensemble(bundle)
            self.enter_scale = bundle["enter_scale"]
            self.costs_bps = bundle["costs_bps"]
            self.bundle_mtime = mtime
            self.emit({"type": "model_swapped", "n_features": bundle["n_features"], "enter_scale": self.enter_scale})
        except Exception as exc:
            self.emit({"type": "hotswap_error", "error": str(exc)})

    def record_metric(self) -> None:
        equity = sum(self.last_dex_equity.values()) if self.last_dex_equity else 0.0
        syms = {}
        for s in self.symbols:
            perp, stock = self.bars[s].perp_last, self.bars[s].stock_last
            syms[s] = {
                "pred": round(self.last_pred.get(s, 0.0), 3),
                "prem": round((stock / perp - 1) * 100, 4) if (perp and stock) else None,
                "szi": round(self.local_szi[s], 6),
                "target": round(self.last_target.get(s, 0.0), 3),
            }
        self.store.record_metric({"equity": round(equity, 2), "syms": syms})

    def record_state(self, second: int, in_session: bool) -> None:
        self.store.record_state({
            "second": second, "in_session": in_session, "live": self.live,
            "model_mtime": self.bundle_mtime, "dex_equity": self.last_dex_equity,
            "symbols": {s: {
                "target": round(self.last_target.get(s, 0.0), 3),
                "szi": round(self.local_szi[s], 6),
                "perp": self.bars[s].perp_last,
                "stock": self.bars[s].stock_last,
                "pred_bps": round(self.last_pred.get(s, 0.0), 3),
                "notional": round(self.symbol_notional.get(s, 0.0), 2),
                "ready": self.streams[s].ready(),
            } for s in self.symbols},
        })

    # ---- main loop ----
    def session_second(self) -> int:
        return int(time.time()) % 86400 - RTH_START_UTC

    def run(self) -> None:
        self.refresh_account()
        self.warmup_from_store()
        if self.live:
            for symbol in self.symbols:
                try:
                    self.exchange.update_leverage(self.config.leverage[symbol], self.coin[symbol], is_cross=True)
                except Exception as exc:
                    self.emit({"type": "leverage_error", "symbol": symbol, "requested": self.config.leverage[symbol], "error": str(exc)})
        self.start_perp_feed()
        self.start_stock_feed()
        self.emit({"type": "started", "live": self.live, "symbols": self.symbols, "enter_scale": self.enter_scale, "poll": POLL_INTERVAL})

        last_second = -1
        last_account_refresh = 0.0
        last_hotswap = time.time()
        last_metric = 0.0
        flattened = False
        while True:
            poll_start = time.time()
            second = self.session_second()
            in_session = 0 <= second < SECONDS
            tradeable = in_session and second < SECONDS - FLATTEN_BEFORE_CLOSE_SEC

            # 1 Hz: roll bars -> persist -> features -> hysteresis decision
            if second != last_second:
                date = self.today()
                ts_ms = int(time.time() * 1000)
                for symbol in self.symbols:
                    pl, pb, ps, pc, sl, sc = self.bars[symbol].roll()
                    self.streams[symbol].push_second(second, pl, pb, ps, pc, sl, sc)
                    if in_session:
                        self.store.record_bar(date, symbol, second, ts_ms, pl, pb, ps, pc, sl, sc)
                if in_session:
                    self.store.commit()
                if tradeable:
                    for symbol in self.symbols:
                        target = self.decide(symbol)
                        if target is not None:
                            self.last_target[symbol] = target
                if second % 60 == 0:
                    self.emit({"type": "heartbeat", "session_second": second, "in_session": in_session,
                               "ready": sum(self.streams[s].ready() for s in self.symbols),
                               "targets": {s: round(t, 2) for s, t in self.last_target.items() if abs(t) > 1e-9},
                               "positions": {s: round(v, 4) for s, v in self.local_szi.items() if abs(v) > 1e-9}})
                last_second = second

            if tradeable:
                flattened = False
                if poll_start - last_account_refresh >= 60:
                    self.refresh_account()
                    last_account_refresh = poll_start
                for symbol in self.symbols:  # execution polled at POLL_INTERVAL
                    perp_price = self.bars[symbol].perp_last
                    if perp_price:
                        self.reconcile(symbol, self.last_target.get(symbol, 0.0), perp_price)
            elif in_session and not flattened:
                self.flatten_all("near_close")
                flattened = True

            if poll_start - last_hotswap >= HOTSWAP_CHECK_SEC:
                self.maybe_hotswap()
                last_hotswap = poll_start

            self.record_state(second, in_session)
            if poll_start - last_metric >= 20:
                self.record_metric()
                last_metric = poll_start
            time.sleep(max(0.0, POLL_INTERVAL - (time.time() - poll_start)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--live", action="store_true", help="place real orders (default: dry-run, logs only)")
    args = parser.parse_args()
    config = EngineConfig(**yaml.safe_load(Path(args.config).read_text()))
    LiveEngine(config, live=args.live).run()


if __name__ == "__main__":
    main()
