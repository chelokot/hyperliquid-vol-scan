from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Literal

import requests
import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "rl_research"))
from rl_policy_train import StockOutlierFilter

YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0"}


class Config(BaseModel):
    base_url: str = "https://api.hyperliquid.xyz"
    account_address: str
    secret_key_env: str = "HYPERLIQUID_SECRET_KEY"
    dex: str = "xyz"
    coin: str = "xyz:SPCX"
    stock_symbol: str = "SPCX"
    leverage: int = 3
    threshold_pct: float = Field(gt=0)
    max_stock_age_sec: float = Field(gt=0)
    poll_interval_sec: float = Field(gt=0)
    min_order_notional_usd: float = 12.0
    events_path: str


class YahooQuote(BaseModel):
    price: float
    bar_ts: int


def yahoo_last(symbol: str) -> YahooQuote:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1m&range=1d&includePrePost=true"
    response = requests.get(url, headers=YAHOO_HEADERS, timeout=5)
    response.raise_for_status()
    result = response.json()["chart"]["result"][0]
    timestamps = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]
    for index in range(len(closes) - 1, -1, -1):
        if closes[index] is not None:
            return YahooQuote(price=closes[index], bar_ts=timestamps[index])
    raise RuntimeError("no yahoo price")


class LeadLagBot:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.events_path = ROOT / config.events_path
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        import eth_account
        from hyperliquid.exchange import Exchange
        from hyperliquid.info import Info

        account = eth_account.Account.from_key(os.environ[config.secret_key_env])
        self.info = Info(config.base_url, skip_ws=True, perp_dexs=["", config.dex])
        self.exchange = Exchange(
            account,
            config.base_url,
            account_address=config.account_address,
            perp_dexs=["", config.dex],
        )
        self.size_decimals = self.info.asset_to_sz_decimals[self.info.name_to_asset(config.coin)]
        self.outlier_filter = StockOutlierFilter()

    def emit(self, event: dict[str, Any]) -> None:
        event["time_ms"] = int(time.time() * 1000)
        with self.events_path.open("a") as handle:
            handle.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)

    def perp_state(self) -> tuple[float, float]:
        state = self.info.post("/info", {"type": "clearinghouseState", "user": self.config.account_address, "dex": self.config.dex})
        spot = self.info.post("/info", {"type": "spotClearinghouseState", "user": self.config.account_address})
        spot_usdc = next((float(balance["total"]) for balance in spot["balances"] if balance["coin"] == "USDC"), 0.0)
        equity = float(state["marginSummary"]["accountValue"]) + spot_usdc
        position_size = 0.0
        for asset_position in state["assetPositions"]:
            if asset_position["position"]["coin"] == self.config.coin:
                position_size = float(asset_position["position"]["szi"])
        return equity, position_size

    def perp_mid(self) -> float:
        book = self.info.post("/info", {"type": "l2Book", "coin": self.config.coin})
        bids, asks = book["levels"]
        return (float(bids[0]["px"]) + float(asks[0]["px"])) / 2

    def check_funding(self) -> None:
        equity, _ = self.perp_state()
        if equity < self.config.min_order_notional_usd:
            self.emit({"type": "funding_required", "detail": "unified equity too small; deposit USDC", "equity": equity})

    def close_position(self, position_size: float, reason: str) -> None:
        result = self.exchange.market_close(self.config.coin)
        self.emit({"type": "close", "size": position_size, "reason": reason, "result_status": str(result.get("status"))})

    def open_position(self, side: Literal["long", "short"], equity: float, perp_price: float, premium_pct: float) -> None:
        notional = equity * self.config.leverage * 0.95
        if notional < self.config.min_order_notional_usd:
            self.emit({"type": "skip_open", "reason": "notional_too_small", "notional": notional})
            return
        size = round(notional / perp_price, self.size_decimals)
        result = self.exchange.market_open(self.config.coin, side == "long", size)
        self.emit(
            {
                "type": "open",
                "side": side,
                "size": size,
                "perp_price": perp_price,
                "premium_pct": premium_pct,
                "result_status": str(result.get("status")),
            }
        )

    def run(self) -> None:
        self.check_funding()
        self.exchange.update_leverage(self.config.leverage, self.config.coin, is_cross=True)
        equity, position_size = self.perp_state()
        self.emit({"type": "started", "equity": equity, "adopted_position": position_size})
        last_tick_log = 0.0
        while True:
            loop_started = time.time()
            try:
                self.step(loop_started, last_tick_log)
                if loop_started - last_tick_log >= 60:
                    last_tick_log = loop_started
            except Exception as exc:
                self.emit({"type": "error", "error": str(exc)})
                time.sleep(5)
            time.sleep(max(0.0, self.config.poll_interval_sec - (time.time() - loop_started)))

    def step(self, now: float, last_tick_log: float) -> None:
        quote = yahoo_last(self.config.stock_symbol)
        stock_age = now - quote.bar_ts
        perp_price = self.perp_mid()
        stock_price = self.outlier_filter.accept(int(now * 1000), quote.price, perp_price)
        equity, position_size = self.perp_state()
        current: Literal["flat", "long", "short"] = "flat" if position_size == 0 else ("long" if position_size > 0 else "short")
        premium_pct = (stock_price / perp_price - 1) * 100

        if stock_age > self.config.max_stock_age_sec:
            if current != "flat":
                self.close_position(position_size, "stale_stock_data")
            if now - last_tick_log >= 60:
                self.emit({"type": "standby", "stock_age_sec": round(stock_age), "premium_pct": round(premium_pct, 3)})
            return

        desired: Literal["flat", "long", "short"] = current
        if premium_pct >= self.config.threshold_pct:
            desired = "long"
        elif premium_pct <= -self.config.threshold_pct:
            desired = "short"

        if now - last_tick_log >= 60:
            self.emit(
                {
                    "type": "tick",
                    "stock": stock_price,
                    "perp": perp_price,
                    "premium_pct": round(premium_pct, 3),
                    "stock_age_sec": round(stock_age),
                    "position": current,
                    "equity": equity,
                }
            )

        if desired == current:
            return
        if current != "flat":
            self.close_position(position_size, f"signal_flip_to_{desired}")
            equity, _ = self.perp_state()
        self.open_position(desired, equity, perp_price, premium_pct)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--i-understand-live-trading", action="store_true", required=True)
    args = parser.parse_args()
    config = Config(**yaml.safe_load(Path(args.config).read_text()))
    LeadLagBot(config).run()


if __name__ == "__main__":
    main()
