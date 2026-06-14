from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Literal

import requests


API_URL = "https://api.hyperliquid.xyz/info"
OUT_DIR = Path(__file__).resolve().parent / "out"
MAKER_FEE = 0.00015
TAKER_FEE = 0.00045
UNIT_NOTIONAL = 100.0
COINS = (
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "HYPE",
    "ZEC",
    "IO",
    "PENDLE",
    "JTO",
    "CHIP",
    "VVV",
    "LIT",
    "GALA",
    "MORPHO",
    "IP",
    "ONDO",
    "DOGE",
    "SUI",
    "LINK",
    "AAVE",
)
ENTRY_PCTS = (0.005, 0.0075, 0.01, 0.015, 0.02)
TP_PCTS = (0.005, 0.0075, 0.01, 0.015, 0.02)
SL_PCTS = (0.005, 0.0075, 0.01, 0.015, 0.02, 0.03)
MODES: tuple[Literal["breakout", "fade"], ...] = ("breakout", "fade")
SIDES: tuple[Literal["both", "long", "short"], ...] = ("both", "long", "short")


@dataclass(frozen=True)
class Candle:
    time_ms: int
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class Market:
    coin: str
    max_leverage: int
    day_ntl_vlm: float
    funding: float
    spread_pct: float


@dataclass(frozen=True)
class Strategy:
    mode: Literal["breakout", "fade"]
    side_filter: Literal["both", "long", "short"]
    entry_pct: float
    tp_pct: float
    sl_pct: float


@dataclass(frozen=True)
class WindowResult:
    pnl: float
    capital_at_risk: float
    trades: int

    @property
    def multiplier(self) -> float:
        return 1.0 + self.pnl / self.capital_at_risk if self.capital_at_risk > 0 else 1.0


def post(payload: dict[str, Any]) -> Any:
    last_error: Exception | None = None
    for attempt in range(8):
        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            if response.status_code == 429:
                time.sleep(0.7 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError("Hyperliquid API rate limit")


def load_markets() -> dict[str, Market]:
    meta, contexts = post({"type": "metaAndAssetCtxs"})
    markets: dict[str, Market] = {}
    for universe_item, context in zip(meta["universe"], contexts, strict=True):
        coin = universe_item["name"]
        if coin not in COINS:
            continue
        spread_pct = live_spread_pct(coin)
        markets[coin] = Market(
            coin=coin,
            max_leverage=int(universe_item["maxLeverage"]),
            day_ntl_vlm=float(context.get("dayNtlVlm") or 0.0),
            funding=float(context.get("funding") or 0.0),
            spread_pct=spread_pct,
        )
    return markets


def live_spread_pct(coin: str) -> float:
    book = post({"type": "l2Book", "coin": coin})
    bids, asks = book["levels"]
    bid = float(bids[0]["px"])
    ask = float(asks[0]["px"])
    return (ask - bid) / ((ask + bid) / 2) * 100


def load_candles(coin: str, interval: str, lookback_ms: int) -> list[Candle]:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - lookback_ms
    raw = post(
        {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
            },
        }
    )
    return [
        Candle(
            time_ms=int(candle["t"]),
            open=float(candle["o"]),
            high=float(candle["h"]),
            low=float(candle["l"]),
            close=float(candle["c"]),
        )
        for candle in sorted(raw, key=lambda item: int(item["t"]))
    ]


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def side_pnl(side: Literal["long", "short"], entry_price: float, exit_price: float, entry_fee: float, exit_fee: float) -> float:
    quantity = UNIT_NOTIONAL / entry_price
    gross = quantity * (exit_price - entry_price) if side == "long" else quantity * (entry_price - exit_price)
    return gross - UNIT_NOTIONAL * entry_fee - quantity * exit_price * exit_fee


def simulate_window(candles: list[Candle], market: Market, strategy: Strategy) -> WindowResult:
    anchor = candles[0].close
    position_side: Literal["flat", "long", "short"] = "flat"
    entry_price = 0.0
    take_profit = 0.0
    stop_loss = 0.0
    entry_fee = MAKER_FEE
    pnl = 0.0
    trades = 0
    max_drawdown = 0.0
    peak_equity = 0.0
    max_notional = 0.0

    for candle in candles[1:]:
        if position_side == "long":
            quantity = UNIT_NOTIONAL / entry_price
            close_equity = pnl + quantity * (candle.close - entry_price) - UNIT_NOTIONAL * entry_fee
            low_equity = pnl + quantity * (candle.low - entry_price) - UNIT_NOTIONAL * entry_fee
            peak_equity = max(peak_equity, close_equity)
            max_drawdown = max(max_drawdown, peak_equity - low_equity)
            if candle.low <= stop_loss:
                pnl += side_pnl("long", entry_price, stop_loss, entry_fee, TAKER_FEE)
                anchor = stop_loss
                position_side = "flat"
                trades += 1
                continue
            if candle.high >= take_profit:
                pnl += side_pnl("long", entry_price, take_profit, entry_fee, MAKER_FEE)
                anchor = take_profit
                position_side = "flat"
                trades += 1
                continue
        elif position_side == "short":
            quantity = UNIT_NOTIONAL / entry_price
            close_equity = pnl + quantity * (entry_price - candle.close) - UNIT_NOTIONAL * entry_fee
            high_equity = pnl + quantity * (entry_price - candle.high) - UNIT_NOTIONAL * entry_fee
            peak_equity = max(peak_equity, close_equity)
            max_drawdown = max(max_drawdown, peak_equity - high_equity)
            if candle.high >= stop_loss:
                pnl += side_pnl("short", entry_price, stop_loss, entry_fee, TAKER_FEE)
                anchor = stop_loss
                position_side = "flat"
                trades += 1
                continue
            if candle.low <= take_profit:
                pnl += side_pnl("short", entry_price, take_profit, entry_fee, MAKER_FEE)
                anchor = take_profit
                position_side = "flat"
                trades += 1
                continue
        else:
            lower = anchor * (1.0 - strategy.entry_pct)
            upper = anchor * (1.0 + strategy.entry_pct)
            touched_lower = candle.low <= lower
            touched_upper = candle.high >= upper
            if touched_lower and touched_upper:
                continue
            allow_long = strategy.side_filter in {"both", "long"}
            allow_short = strategy.side_filter in {"both", "short"}
            if strategy.mode == "breakout":
                if allow_long and touched_upper:
                    position_side = "long"
                    entry_price = upper
                    take_profit = entry_price * (1.0 + strategy.tp_pct)
                    stop_loss = entry_price * (1.0 - strategy.sl_pct)
                    entry_fee = TAKER_FEE
                    max_notional = UNIT_NOTIONAL
                elif allow_short and touched_lower:
                    position_side = "short"
                    entry_price = lower
                    take_profit = entry_price * (1.0 - strategy.tp_pct)
                    stop_loss = entry_price * (1.0 + strategy.sl_pct)
                    entry_fee = TAKER_FEE
                    max_notional = UNIT_NOTIONAL
            else:
                if allow_long and touched_lower:
                    position_side = "long"
                    entry_price = lower
                    take_profit = entry_price * (1.0 + strategy.tp_pct)
                    stop_loss = entry_price * (1.0 - strategy.sl_pct)
                    entry_fee = MAKER_FEE
                    max_notional = UNIT_NOTIONAL
                elif allow_short and touched_upper:
                    position_side = "short"
                    entry_price = upper
                    take_profit = entry_price * (1.0 - strategy.tp_pct)
                    stop_loss = entry_price * (1.0 + strategy.sl_pct)
                    entry_fee = MAKER_FEE
                    max_notional = UNIT_NOTIONAL

    final_close = candles[-1].close
    if position_side == "long":
        pnl += side_pnl("long", entry_price, final_close, entry_fee, TAKER_FEE)
    elif position_side == "short":
        pnl += side_pnl("short", entry_price, final_close, entry_fee, TAKER_FEE)

    margin = max_notional / market.max_leverage if max_notional > 0 else 1.0
    return WindowResult(pnl=pnl, capital_at_risk=max(1.0, margin + max_drawdown), trades=trades)


def evaluate(candles: list[Candle], market: Market, strategy: Strategy, window_points: int, shift_points: int) -> dict[str, float]:
    results = [
        simulate_window(candles[index : index + window_points], market, strategy)
        for index in range(0, len(candles) - window_points + 1, shift_points)
    ]
    multipliers = [result.multiplier for result in results]
    trade_counts = [result.trades for result in results]
    return {
        "windows": float(len(results)),
        "geo_profit_pct": (geometric_mean(multipliers) - 1.0) * 100,
        "median_profit_pct": (median(multipliers) - 1.0) * 100,
        "positive_window_pct": sum(1 for result in results if result.pnl > 0) / len(results) * 100,
        "active_window_pct": sum(1 for result in results if result.trades > 0) / len(results) * 100,
        "worst_window_pct": (min(multipliers) - 1.0) * 100,
        "avg_trades": mean(trade_counts),
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    markets = load_markets()
    strategies = [
        Strategy(mode, side, entry_pct, tp_pct, sl_pct)
        for mode in MODES
        for side in SIDES
        for entry_pct in ENTRY_PCTS
        for tp_pct in TP_PCTS
        for sl_pct in SL_PCTS
    ]
    rows: list[dict[str, Any]] = []
    for coin in COINS:
        if coin not in markets:
            continue
        market = markets[coin]
        candles_15m = load_candles(coin, "15m", 30 * 24 * 60 * 60 * 1000)
        candles_1m = load_candles(coin, "1m", 5000 * 60 * 1000)
        for strategy in strategies:
            score_15m = evaluate(candles_15m, market, strategy, 24 * 4, 4)
            if score_15m["active_window_pct"] < 35:
                continue
            if score_15m["worst_window_pct"] < -45:
                continue
            score_1m = evaluate(candles_1m, market, strategy, 24 * 60, 60)
            combined = (
                min(score_15m["geo_profit_pct"], score_1m["geo_profit_pct"]) * 1.8
                + min(score_15m["positive_window_pct"], score_1m["positive_window_pct"]) * 0.15
                + min(score_15m["active_window_pct"], score_1m["active_window_pct"]) * 0.08
                - max(0.0, -min(score_15m["worst_window_pct"], score_1m["worst_window_pct"]) - 20.0) * 1.1
                - market.spread_pct * 90
            )
            rows.append(
                {
                    "coin": coin,
                    "mode": strategy.mode,
                    "side": strategy.side_filter,
                    "entry_pct": strategy.entry_pct * 100,
                    "tp_pct": strategy.tp_pct * 100,
                    "sl_pct": strategy.sl_pct * 100,
                    "max_leverage": market.max_leverage,
                    "day_ntl_vlm": market.day_ntl_vlm,
                    "spread_pct": market.spread_pct,
                    "geo_15m_30d_pct": score_15m["geo_profit_pct"],
                    "geo_1m_recent_pct": score_1m["geo_profit_pct"],
                    "positive_15m_pct": score_15m["positive_window_pct"],
                    "positive_1m_pct": score_1m["positive_window_pct"],
                    "active_15m_pct": score_15m["active_window_pct"],
                    "active_1m_pct": score_1m["active_window_pct"],
                    "worst_15m_pct": score_15m["worst_window_pct"],
                    "worst_1m_pct": score_1m["worst_window_pct"],
                    "trades_15m": score_15m["avg_trades"],
                    "trades_1m": score_1m["avg_trades"],
                    "combined_score": combined,
                }
            )
        print(f"loaded {coin}: {len([row for row in rows if row['coin'] == coin])} survivors", flush=True)

    rows.sort(key=lambda row: row["combined_score"], reverse=True)
    output_path = OUT_DIR / f"hyperliquid_advanced_search_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(output_path)
    for row in rows[:40]:
        print(
            f"{row['coin']:>7} {row['mode']} {row['side']} "
            f"entry={row['entry_pct']:.2f}% tp={row['tp_pct']:.2f}% sl={row['sl_pct']:.2f}% "
            f"geo30={row['geo_15m_30d_pct']:.2f}% geo1m={row['geo_1m_recent_pct']:.2f}% "
            f"win={min(row['positive_15m_pct'], row['positive_1m_pct']):.1f}% "
            f"worst={min(row['worst_15m_pct'], row['worst_1m_pct']):.1f}% "
            f"score={row['combined_score']:.1f}"
        )


if __name__ == "__main__":
    main()
