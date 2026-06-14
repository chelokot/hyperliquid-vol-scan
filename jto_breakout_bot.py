from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import requests
import yaml
from pydantic import BaseModel, Field


class Config(BaseModel):
    mode: Literal["paper", "live"] = "paper"
    base_url: str = "https://api.hyperliquid.xyz"
    account_address: str
    secret_key_env: str = "HYPERLIQUID_SECRET_KEY"
    coin: str = "JTO"
    leverage: int = 5
    cross_margin: bool = True
    entry_pct: float = Field(gt=0)
    take_profit_pct: float = Field(gt=0)
    stop_loss_pct: float = Field(gt=0)
    order_notional_usd: float = Field(gt=0)
    max_spread_pct: float = Field(gt=0)
    market_slippage: float = Field(gt=0)
    poll_interval_sec: float = Field(gt=0)
    state_path: str
    events_path: str


@dataclass
class State:
    phase: Literal["flat", "long", "short"]
    anchor_price: float
    entry_price: float | None
    position_size: float
    tp_cloid: str | None
    sl_cloid: str | None
    started_at_ms: int
    updated_at_ms: int


class InfoClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def post(self, payload: dict[str, Any]) -> Any:
        last_error: Exception | None = None
        for attempt in range(8):
            try:
                response = requests.post(f"{self.base_url}/info", json=payload, timeout=20)
                if response.status_code == 429:
                    time.sleep(0.8 * (attempt + 1))
                    continue
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(0.5 * (attempt + 1))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Hyperliquid API request failed")

    def mid_and_spread(self, coin: str) -> tuple[float, float]:
        book = self.post({"type": "l2Book", "coin": coin})
        bids, asks = book["levels"]
        bid = float(bids[0]["px"])
        ask = float(asks[0]["px"])
        mid = (bid + ask) / 2
        return mid, (ask - bid) / mid * 100

    def spot_usdc_available(self, user: str) -> float:
        state = self.post({"type": "spotClearinghouseState", "user": user})
        for token, amount in state.get("tokenToAvailableAfterMaintenance", []):
            if token == 0:
                return float(amount)
        return 0.0


class BreakoutBot:
    def __init__(self, config: Config, root: Path) -> None:
        self.config = config
        self.root = root
        self.info_client = InfoClient(config.base_url)
        self.state_path = self.resolve_path(config.state_path)
        self.events_path = self.resolve_path(config.events_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.root / candidate

    def record(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {"time_ms": int(time.time() * 1000), "type": event_type, **payload}
        with self.events_path.open("a") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def save(self) -> None:
        self.state_path.write_text(json.dumps(asdict(self.state), indent=2, sort_keys=True))

    def load_or_create_state(self) -> None:
        if self.state_path.exists():
            self.state = State(**json.loads(self.state_path.read_text()))
            return
        mid, _ = self.info_client.mid_and_spread(self.config.coin)
        now_ms = int(time.time() * 1000)
        self.state = State(
            phase="flat",
            anchor_price=mid,
            entry_price=None,
            position_size=0.0,
            tp_cloid=None,
            sl_cloid=None,
            started_at_ms=now_ms,
            updated_at_ms=now_ms,
        )
        self.save()
        self.record("started", {"anchor_price": mid})

    def cloid(self, purpose: str) -> Any:
        from hyperliquid.utils.types import Cloid

        seed = f"jto-breakout-v1:{purpose}:{self.state.started_at_ms}:{self.state.updated_at_ms}".encode()
        return Cloid("0x" + hashlib.blake2b(seed, digest_size=16).hexdigest())

    def rounded_price(self, exchange: Any, price: float) -> float:
        size_decimals = exchange.info.asset_to_sz_decimals[exchange.info.name_to_asset(self.config.coin)]
        return round(float(f"{price:.5g}"), 6 - size_decimals)

    def rounded_size(self, exchange: Any, price: float) -> float:
        size_decimals = exchange.info.asset_to_sz_decimals[exchange.info.name_to_asset(self.config.coin)]
        scale = 10**size_decimals
        return math.floor((self.config.order_notional_usd / price) * scale) / scale

    def position(self, info: Any, address: str) -> tuple[float, float | None]:
        state = info.user_state(address)
        for asset_position in state["assetPositions"]:
            position = asset_position["position"]
            if position["coin"] == self.config.coin:
                return float(position["szi"]), float(position["entryPx"])
        return 0.0, None

    def open_orders(self, info: Any, address: str) -> list[dict[str, Any]]:
        return [order for order in info.frontend_open_orders(address) if order.get("coin") == self.config.coin]

    def place_exits(self, exchange: Any, size: float, entry_price: float, side: Literal["long", "short"]) -> None:
        abs_size = abs(size)
        if side == "long":
            tp_price = self.rounded_price(exchange, entry_price * (1 + self.config.take_profit_pct / 100))
            sl_price = self.rounded_price(exchange, entry_price * (1 - self.config.stop_loss_pct / 100))
            close_is_buy = False
        else:
            tp_price = self.rounded_price(exchange, entry_price * (1 - self.config.take_profit_pct / 100))
            sl_price = self.rounded_price(exchange, entry_price * (1 + self.config.stop_loss_pct / 100))
            close_is_buy = True
        tp_cloid = self.cloid("tp")
        sl_cloid = self.cloid("sl")
        tp_result = exchange.order(
            self.config.coin,
            close_is_buy,
            abs_size,
            tp_price,
            {"limit": {"tif": "Gtc"}},
            reduce_only=True,
            cloid=tp_cloid,
        )
        sl_result = exchange.order(
            self.config.coin,
            close_is_buy,
            abs_size,
            sl_price,
            {"trigger": {"triggerPx": sl_price, "isMarket": True, "tpsl": "sl"}},
            reduce_only=True,
            cloid=sl_cloid,
        )
        self.state.tp_cloid = tp_cloid.to_raw()
        self.state.sl_cloid = sl_cloid.to_raw()
        self.record("place_exits", {"tp_price": tp_price, "sl_price": sl_price, "tp_result": tp_result, "sl_result": sl_result})

    def cancel_known_exits(self, exchange: Any) -> None:
        from hyperliquid.utils.types import Cloid

        for raw in [self.state.tp_cloid, self.state.sl_cloid]:
            if raw is None:
                continue
            result = exchange.cancel_by_cloid(self.config.coin, Cloid(raw))
            self.record("cancel_exit", {"cloid": raw, "result": result})
        self.state.tp_cloid = None
        self.state.sl_cloid = None

    def step_paper(self) -> None:
        self.load_or_create_state()
        mid, spread = self.info_client.mid_and_spread(self.config.coin)
        self.record("paper_tick", {"mid": mid, "spread_pct": spread, "phase": self.state.phase})
        self.state.updated_at_ms = int(time.time() * 1000)
        self.save()

    def step_live(self, info: Any, exchange: Any, address: str) -> None:
        self.load_or_create_state()
        mid, spread = self.info_client.mid_and_spread(self.config.coin)
        spot_available = self.info_client.spot_usdc_available(address)
        upper = self.state.anchor_price * (1 + self.config.entry_pct / 100)
        lower = self.state.anchor_price * (1 - self.config.entry_pct / 100)
        distance_to_entry_pct = min(abs(upper / mid - 1), abs(mid / lower - 1)) * 100
        if spread > self.config.max_spread_pct:
            self.record(
                "paused",
                {
                    "reason": "spread",
                    "spread_pct": spread,
                    "mid": mid,
                    "lower_entry": lower,
                    "upper_entry": upper,
                    "distance_to_entry_pct": distance_to_entry_pct,
                },
            )
            return
        position_size, entry_price = self.position(info, address)
        orders = self.open_orders(info, address)
        if abs(position_size) < 1e-12:
            if self.state.phase != "flat":
                self.cancel_known_exits(exchange)
                self.state.phase = "flat"
                self.state.entry_price = None
                self.state.position_size = 0.0
                self.state.anchor_price = mid
            if mid >= upper or mid <= lower:
                is_buy = mid >= upper
                order_size = self.rounded_size(exchange, mid)
                cloid = self.cloid("entry")
                result = exchange.market_open(
                    self.config.coin,
                    is_buy,
                    order_size,
                    slippage=self.config.market_slippage,
                    cloid=cloid,
                )
                self.record(
                    "entry",
                    {
                        "side": "long" if is_buy else "short",
                        "mid": mid,
                        "size": order_size,
                        "spot_available": spot_available,
                        "result": result,
                    },
                )
                time.sleep(1)
                position_size, entry_price = self.position(info, address)
                if entry_price is not None and abs(position_size) > 0:
                    self.state.phase = "long" if position_size > 0 else "short"
                    self.state.entry_price = entry_price
                    self.state.position_size = position_size
                    self.place_exits(exchange, position_size, entry_price, self.state.phase)
            else:
                self.record(
                    "live_tick",
                    {
                        "mid": mid,
                        "spread_pct": spread,
                        "phase": "flat",
                        "anchor": self.state.anchor_price,
                        "lower_entry": lower,
                        "upper_entry": upper,
                        "distance_to_entry_pct": distance_to_entry_pct,
                        "spot_available": spot_available,
                    },
                )
        else:
            self.state.phase = "long" if position_size > 0 else "short"
            self.state.position_size = position_size
            self.state.entry_price = entry_price
            if not any(order.get("cloid") == self.state.tp_cloid for order in orders) or not any(order.get("cloid") == self.state.sl_cloid for order in orders):
                self.place_exits(exchange, position_size, entry_price or mid, self.state.phase)
            self.record("live_tick", {"mid": mid, "spread_pct": spread, "phase": self.state.phase, "position_size": position_size, "entry_price": entry_price, "spot_available": spot_available})
        self.state.updated_at_ms = int(time.time() * 1000)
        self.save()

    def run(self, once: bool, allow_live: bool) -> None:
        if self.config.mode == "paper":
            while True:
                self.step_paper()
                if once:
                    return
                time.sleep(self.config.poll_interval_sec)
        if not allow_live:
            raise RuntimeError("Live mode requires --i-understand-live-trading")
        if not os.getenv(self.config.secret_key_env):
            raise RuntimeError(f"Set {self.config.secret_key_env}")
        import eth_account
        from hyperliquid.exchange import Exchange
        from hyperliquid.info import Info

        account = eth_account.Account.from_key(os.environ[self.config.secret_key_env])
        info = Info(self.config.base_url, skip_ws=True)
        exchange = Exchange(account, self.config.base_url, account_address=self.config.account_address)
        exchange.update_leverage(self.config.leverage, self.config.coin, is_cross=self.config.cross_margin)
        while True:
            try:
                self.step_live(info, exchange, self.config.account_address)
            except Exception as exc:
                self.record("error", {"error": repr(exc)})
            if once:
                return
            time.sleep(self.config.poll_interval_sec)


def load_config(path: Path) -> Config:
    return Config.model_validate(yaml.safe_load(path.read_text()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="jto_breakout_bot_config.yaml")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--i-understand-live-trading", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    BreakoutBot(load_config(config_path), root).run(args.once, args.i_understand_live_trading)


if __name__ == "__main__":
    main()
