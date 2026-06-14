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
from pydantic import BaseModel, Field, field_validator


class BotConfig(BaseModel):
    mode: Literal["paper", "live"] = "paper"
    base_url: str = "https://api.hyperliquid.xyz"
    account_address: str = ""
    secret_key_env: str = "HYPERLIQUID_SECRET_KEY"
    coin: str = "BTC"
    leverage: int = 40
    cross_margin: bool = True
    grid_step_pct: float = Field(gt=0)
    order_notional_usd: float = Field(gt=0)
    max_abs_position_usd: float = Field(gt=0)
    max_levels_from_anchor: int = Field(gt=0)
    max_spread_pct: float = Field(gt=0)
    poll_interval_sec: float = Field(gt=0)
    state_path: str
    events_path: str

    @field_validator("coin")
    @classmethod
    def normalize_coin(cls, value: str) -> str:
        return value.strip().upper()


@dataclass
class BotState:
    anchor_price: float
    current_level: int
    cash_usd: float
    position_size: float
    fees_usd: float
    last_mid: float
    started_at_ms: int
    updated_at_ms: int


@dataclass
class LiveBotState:
    anchor_price: float
    current_level: int
    active_buy_cloid: str | None
    active_buy_level: int | None
    active_sell_cloid: str | None
    active_sell_level: int | None
    last_mid: float
    started_at_ms: int
    updated_at_ms: int


class HyperliquidInfoClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def post(self, payload: dict[str, Any]) -> Any:
        response = requests.post(f"{self.base_url}/info", json=payload, timeout=20)
        response.raise_for_status()
        return response.json()

    def l2_book(self, coin: str) -> dict[str, Any]:
        return self.post({"type": "l2Book", "coin": coin})

    def mid_and_spread(self, coin: str) -> tuple[float, float]:
        book = self.l2_book(coin)
        bids, asks = book["levels"]
        if not bids or not asks:
            raise RuntimeError(f"No L2 book for {coin}")
        bid = float(bids[0]["px"])
        ask = float(asks[0]["px"])
        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid * 100
        return mid, spread_pct


class PaperGridBot:
    def __init__(self, config: BotConfig, root: Path) -> None:
        self.config = config
        self.root = root
        self.info = HyperliquidInfoClient(config.base_url)
        self.state_path = self.resolve_path(config.state_path)
        self.events_path = self.resolve_path(config.events_path)
        self.step = config.grid_step_pct / 100
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.root / candidate

    def level_price(self, level: int) -> float:
        return self.state.anchor_price * ((1.0 + self.step) ** level)

    def price_level(self, price: float) -> int:
        raw_level = math.log(price / self.state.anchor_price) / math.log(1.0 + self.step)
        return math.floor(raw_level) if raw_level >= 0 else math.ceil(raw_level)

    def load_or_create_state(self) -> None:
        if self.state_path.exists():
            self.state = BotState(**json.loads(self.state_path.read_text()))
            return
        mid, _ = self.info.mid_and_spread(self.config.coin)
        now_ms = int(time.time() * 1000)
        self.state = BotState(
            anchor_price=mid,
            current_level=0,
            cash_usd=0.0,
            position_size=0.0,
            fees_usd=0.0,
            last_mid=mid,
            started_at_ms=now_ms,
            updated_at_ms=now_ms,
        )
        self.save_state()
        self.record_event("started", {"anchor_price": mid})

    def save_state(self) -> None:
        self.state_path.write_text(json.dumps(asdict(self.state), indent=2, sort_keys=True))

    def record_event(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "time_ms": int(time.time() * 1000),
            "type": event_type,
            **payload,
        }
        with self.events_path.open("a") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def paper_fee(self) -> float:
        return self.config.order_notional_usd * 0.00015

    def equity(self, mid: float) -> float:
        return self.state.cash_usd + self.state.position_size * mid - self.state.fees_usd

    def position_notional(self, mid: float) -> float:
        return abs(self.state.position_size * mid)

    def can_trade(self, side: Literal["buy", "sell"], price: float) -> bool:
        size = self.config.order_notional_usd / price
        next_position = self.state.position_size + size if side == "buy" else self.state.position_size - size
        return abs(next_position * price) <= self.config.max_abs_position_usd

    def fill_buy(self, price: float) -> None:
        if not self.can_trade("buy", price):
            self.record_event("blocked", {"side": "buy", "price": price, "reason": "max_abs_position_usd"})
            return
        size = self.config.order_notional_usd / price
        self.state.position_size += size
        self.state.cash_usd -= self.config.order_notional_usd
        self.state.fees_usd += self.paper_fee()
        self.record_event("paper_fill", {"side": "buy", "price": price, "size": size})

    def fill_sell(self, price: float) -> None:
        if not self.can_trade("sell", price):
            self.record_event("blocked", {"side": "sell", "price": price, "reason": "max_abs_position_usd"})
            return
        size = self.config.order_notional_usd / price
        self.state.position_size -= size
        self.state.cash_usd += self.config.order_notional_usd
        self.state.fees_usd += self.paper_fee()
        self.record_event("paper_fill", {"side": "sell", "price": price, "size": size})

    def step_once(self) -> None:
        mid, spread_pct = self.info.mid_and_spread(self.config.coin)
        if spread_pct > self.config.max_spread_pct:
            self.record_event("paused", {"reason": "spread", "spread_pct": spread_pct, "mid": mid})
            return
        target_level = self.price_level(mid)
        target_level = max(-self.config.max_levels_from_anchor, min(self.config.max_levels_from_anchor, target_level))
        while self.state.current_level < target_level:
            self.state.current_level += 1
            self.fill_sell(self.level_price(self.state.current_level))
        while self.state.current_level > target_level:
            self.state.current_level -= 1
            self.fill_buy(self.level_price(self.state.current_level))
        now_ms = int(time.time() * 1000)
        self.state.last_mid = mid
        self.state.updated_at_ms = now_ms
        self.save_state()
        self.record_event(
            "tick",
            {
                "mid": mid,
                "spread_pct": spread_pct,
                "level": self.state.current_level,
                "position_notional": self.position_notional(mid),
                "equity": self.equity(mid),
            },
        )

    def run(self, once: bool) -> None:
        self.load_or_create_state()
        while True:
            self.step_once()
            if once:
                return
            time.sleep(self.config.poll_interval_sec)


class LiveGridBot:
    def __init__(self, config: BotConfig, root: Path) -> None:
        self.config = config
        self.root = root
        self.state_path = self.resolve_path(config.state_path)
        self.events_path = self.resolve_path(config.events_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.step = config.grid_step_pct / 100

    def resolve_path(self, path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.root / candidate

    def record_event(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "time_ms": int(time.time() * 1000),
            "type": event_type,
            **payload,
        }
        with self.events_path.open("a") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def save_state(self) -> None:
        self.state_path.write_text(json.dumps(asdict(self.state), indent=2, sort_keys=True))

    def level_price(self, level: int) -> float:
        return self.state.anchor_price * ((1.0 + self.step) ** level)

    def price_level(self, price: float) -> int:
        raw_level = math.log(price / self.state.anchor_price) / math.log(1.0 + self.step)
        return math.floor(raw_level) if raw_level >= 0 else math.ceil(raw_level)

    def make_cloid(self, side: Literal["buy", "sell"], level: int) -> Any:
        from hyperliquid.utils.types import Cloid

        seed = f"grid:v1:{self.config.coin}:{side}:{level}:{self.state.started_at_ms}".encode()
        return Cloid("0x" + hashlib.blake2b(seed, digest_size=16).hexdigest())

    def load_or_create_state(self, mid: float) -> None:
        if self.state_path.exists():
            self.state = LiveBotState(**json.loads(self.state_path.read_text()))
            return
        now_ms = int(time.time() * 1000)
        self.state = LiveBotState(
            anchor_price=mid,
            current_level=0,
            active_buy_cloid=None,
            active_buy_level=None,
            active_sell_cloid=None,
            active_sell_level=None,
            last_mid=mid,
            started_at_ms=now_ms,
            updated_at_ms=now_ms,
        )
        self.save_state()
        self.record_event("started", {"anchor_price": mid})

    def current_position_size(self, user_state: dict[str, Any]) -> float:
        for asset_position in user_state["assetPositions"]:
            position = asset_position["position"]
            if position["coin"] == self.config.coin:
                return float(position["szi"])
        return 0.0

    def can_place_side(self, side: Literal["buy", "sell"], current_position_size: float, price: float) -> bool:
        order_size = self.config.order_notional_usd / price
        next_position = current_position_size + order_size if side == "buy" else current_position_size - order_size
        return abs(next_position * price) <= self.config.max_abs_position_usd

    def cancel_cloid(self, exchange: Any, cloid_raw: str | None, side: Literal["buy", "sell"]) -> None:
        if cloid_raw is None:
            return
        from hyperliquid.utils.types import Cloid

        result = exchange.cancel_by_cloid(self.config.coin, Cloid(cloid_raw))
        self.record_event("cancel_by_cloid", {"side": side, "cloid": cloid_raw, "result": result})

    def place_limit(self, exchange: Any, side: Literal["buy", "sell"], level: int, price: float) -> str | None:
        cloid = self.make_cloid(side, level)
        size_decimals = exchange.info.asset_to_sz_decimals[exchange.info.name_to_asset(self.config.coin)]
        price = round(float(f"{price:.5g}"), 6 - size_decimals)
        size_scale = 10**size_decimals
        size = math.floor((self.config.order_notional_usd / price) * size_scale) / size_scale
        if size <= 0:
            raise RuntimeError(f"Order size rounded to zero for {self.config.coin}")
        result = exchange.order(
            self.config.coin,
            side == "buy",
            size,
            price,
            {"limit": {"tif": "Alo"}},
            reduce_only=False,
            cloid=cloid,
        )
        self.record_event(
            "place_order",
            {
                "side": side,
                "level": level,
                "price": price,
                "size": size,
                "cloid": cloid.to_raw(),
                "result": result,
            },
        )
        statuses = result.get("response", {}).get("data", {}).get("statuses", [])
        if result.get("status") != "ok" or not statuses or "error" in statuses[0]:
            return None
        return cloid.to_raw()

    def reconcile_once(self, info: Any, exchange: Any, address: str) -> None:
        mids = info.all_mids()
        mid = float(mids[self.config.coin])
        book = info.l2_snapshot(self.config.coin)
        bids, asks = book["levels"]
        bid = float(bids[0]["px"])
        ask = float(asks[0]["px"])
        spread_pct = (ask - bid) / ((ask + bid) / 2) * 100
        self.load_or_create_state(mid)
        if spread_pct > self.config.max_spread_pct:
            self.record_event("paused", {"reason": "spread", "mid": mid, "spread_pct": spread_pct})
            return
        target_level = self.price_level(mid)
        target_level = max(-self.config.max_levels_from_anchor, min(self.config.max_levels_from_anchor, target_level))
        self.state.current_level = target_level
        user_state = info.user_state(address)
        position_size = self.current_position_size(user_state)
        buy_level = target_level - 1
        sell_level = target_level + 1
        buy_price = self.level_price(buy_level)
        sell_price = self.level_price(sell_level)

        if self.can_place_side("buy", position_size, buy_price):
            if self.state.active_buy_level != buy_level:
                self.cancel_cloid(exchange, self.state.active_buy_cloid, "buy")
                self.state.active_buy_cloid = self.place_limit(exchange, "buy", buy_level, buy_price)
                self.state.active_buy_level = buy_level if self.state.active_buy_cloid is not None else None
        else:
            self.cancel_cloid(exchange, self.state.active_buy_cloid, "buy")
            self.state.active_buy_cloid = None
            self.state.active_buy_level = None
            self.record_event("skip_order", {"side": "buy", "reason": "max_abs_position_usd"})
        if self.can_place_side("sell", position_size, sell_price):
            if self.state.active_sell_level != sell_level:
                self.cancel_cloid(exchange, self.state.active_sell_cloid, "sell")
                self.state.active_sell_cloid = self.place_limit(exchange, "sell", sell_level, sell_price)
                self.state.active_sell_level = sell_level if self.state.active_sell_cloid is not None else None
        else:
            self.cancel_cloid(exchange, self.state.active_sell_cloid, "sell")
            self.state.active_sell_cloid = None
            self.state.active_sell_level = None
            self.record_event("skip_order", {"side": "sell", "reason": "max_abs_position_usd"})

        self.state.last_mid = mid
        self.state.updated_at_ms = int(time.time() * 1000)
        self.save_state()
        self.record_event(
            "live_tick",
            {
                "mid": mid,
                "spread_pct": spread_pct,
                "level": target_level,
                "position_size": position_size,
                "position_notional": abs(position_size * mid),
                "account_value": user_state["marginSummary"]["accountValue"],
            },
        )

    def run(self, once: bool) -> None:
        if not os.getenv(self.config.secret_key_env):
            raise RuntimeError(f"Set {self.config.secret_key_env} before live trading")
        if not self.config.account_address:
            raise RuntimeError("Set account_address to the main Hyperliquid account, not the API wallet address")
        try:
            import eth_account
            from hyperliquid.exchange import Exchange
            from hyperliquid.info import Info
        except ImportError as exc:
            raise RuntimeError("Install live dependencies: pip install -r requirements.txt") from exc
        account = eth_account.Account.from_key(os.environ[self.config.secret_key_env])
        address = self.config.account_address or account.address
        info = Info(self.config.base_url, skip_ws=True)
        exchange = Exchange(account, self.config.base_url, account_address=address)
        exchange.update_leverage(self.config.leverage, self.config.coin, is_cross=self.config.cross_margin)
        while True:
            self.reconcile_once(info, exchange, address)
            if once:
                return
            time.sleep(self.config.poll_interval_sec)


def load_config(path: Path) -> BotConfig:
    return BotConfig.model_validate(yaml.safe_load(path.read_text()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="grid_bot_config.example.yaml")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--i-understand-live-trading", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_config(config_path)
    if config.mode == "live" and not args.i_understand_live_trading:
        raise RuntimeError("Live mode requires --i-understand-live-trading")
    bot = PaperGridBot(config, root) if config.mode == "paper" else LiveGridBot(config, root)
    bot.run(once=args.once)


if __name__ == "__main__":
    main()
