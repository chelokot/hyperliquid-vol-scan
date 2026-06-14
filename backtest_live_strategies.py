from __future__ import annotations

import csv
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Literal

import requests


API_URL = "https://api.hyperliquid.xyz/info"
OUT_DIR = Path(__file__).resolve().parent / "out"
INTERVAL = "15m"
LOOKBACK_DAYS = 30
WINDOW_POINTS = 24 * 4
SHIFT_POINTS = 4
UNIT_NOTIONAL = 100.0
MAKER_FEE = 0.00015
TAKER_FEE = 0.00045
STEPS = (0.01, 0.015, 0.02, 0.03, 0.04, 0.05)
STOP_MULTIPLIERS = (1.0, 1.5, 2.0, 3.0)
STRATEGIES: tuple[str, ...] = ("fade_both", "long_dip", "short_rip", "breakout_both")


@dataclass(frozen=True)
class Market:
    name: str
    max_leverage: int
    day_ntl_vlm: float
    open_interest: float
    funding: float


@dataclass(frozen=True)
class Candle:
    time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class WindowResult:
    pnl: float
    capital_at_risk: float
    trades: int
    wins: int
    losses: int

    @property
    def multiplier(self) -> float:
        return 1.0 + self.pnl / self.capital_at_risk if self.capital_at_risk > 0 else 1.0


def post(payload: dict[str, Any]) -> Any:
    last_error: Exception | None = None
    for attempt in range(8):
        try:
            response = requests.post(API_URL, json=payload, timeout=30)
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
    raise RuntimeError("Hyperliquid API rate limit")


def as_float(value: Any) -> float:
    return 0.0 if value is None else float(value)


def load_markets() -> list[Market]:
    meta, contexts = post({"type": "metaAndAssetCtxs"})
    markets: list[Market] = []
    for universe_item, context in zip(meta["universe"], contexts, strict=True):
        if universe_item.get("isDelisted"):
            continue
        markets.append(
            Market(
                name=universe_item["name"],
                max_leverage=int(universe_item["maxLeverage"]),
                day_ntl_vlm=as_float(context.get("dayNtlVlm")),
                open_interest=as_float(context.get("openInterest")),
                funding=as_float(context.get("funding")),
            )
        )
    return markets


def load_candles(coin: str, start_ms: int, end_ms: int) -> list[Candle]:
    raw_candles = post(
        {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": INTERVAL,
                "startTime": start_ms,
                "endTime": end_ms,
            },
        }
    )
    candles = sorted(raw_candles, key=lambda candle: int(candle["t"]))
    return [
        Candle(
            time_ms=int(candle["t"]),
            open=float(candle["o"]),
            high=float(candle["h"]),
            low=float(candle["l"]),
            close=float(candle["c"]),
            volume=float(candle["v"]),
        )
        for candle in candles
    ]


def geometric_mean(multipliers: list[float]) -> float:
    if not multipliers or any(multiplier <= 0 for multiplier in multipliers):
        return 0.0
    return math.exp(sum(math.log(multiplier) for multiplier in multipliers) / len(multipliers))


def live_spread_pct(coin: str) -> float:
    book = post({"type": "l2Book", "coin": coin})
    bids, asks = book["levels"]
    if not bids or not asks:
        return 999.0
    bid = float(bids[0]["px"])
    ask = float(asks[0]["px"])
    return (ask - bid) / ((ask + bid) / 2) * 100


def close_trade(
    side: Literal["long", "short"],
    entry_price: float,
    exit_price: float,
    entry_fee_rate: float,
    close_fee_rate: float,
) -> float:
    quantity = UNIT_NOTIONAL / entry_price
    gross = quantity * (exit_price - entry_price) if side == "long" else quantity * (entry_price - exit_price)
    return gross - UNIT_NOTIONAL * entry_fee_rate - quantity * exit_price * close_fee_rate


def simulate_window(
    candles: list[Candle],
    leverage: int,
    strategy: str,
    step: float,
    stop_multiplier: float,
) -> WindowResult:
    anchor = candles[0].close
    side: Literal["flat", "long", "short"] = "flat"
    entry_price = 0.0
    take_profit = 0.0
    stop_loss = 0.0
    pnl = 0.0
    wins = 0
    losses = 0
    trades = 0
    peak_equity = 0.0
    max_drawdown = 0.0
    max_notional = 0.0
    entry_fee_rate = MAKER_FEE

    for candle in candles[1:]:
        equity = pnl
        if side == "long":
            quantity = UNIT_NOTIONAL / entry_price
            equity += quantity * (candle.close - entry_price) - UNIT_NOTIONAL * entry_fee_rate
            adverse_equity = pnl + quantity * (candle.low - entry_price) - UNIT_NOTIONAL * entry_fee_rate
            peak_equity = max(peak_equity, equity)
            max_drawdown = max(max_drawdown, peak_equity - adverse_equity)
            max_notional = max(max_notional, UNIT_NOTIONAL)
            if candle.low <= stop_loss:
                trade_pnl = close_trade("long", entry_price, stop_loss, entry_fee_rate, TAKER_FEE)
                pnl += trade_pnl
                losses += int(trade_pnl < 0)
                wins += int(trade_pnl > 0)
                trades += 1
                anchor = stop_loss
                side = "flat"
                continue
            if candle.high >= take_profit:
                trade_pnl = close_trade("long", entry_price, take_profit, entry_fee_rate, MAKER_FEE)
                pnl += trade_pnl
                losses += int(trade_pnl < 0)
                wins += int(trade_pnl > 0)
                trades += 1
                anchor = take_profit
                side = "flat"
                continue
        elif side == "short":
            quantity = UNIT_NOTIONAL / entry_price
            equity += quantity * (entry_price - candle.close) - UNIT_NOTIONAL * entry_fee_rate
            adverse_equity = pnl + quantity * (entry_price - candle.high) - UNIT_NOTIONAL * entry_fee_rate
            peak_equity = max(peak_equity, equity)
            max_drawdown = max(max_drawdown, peak_equity - adverse_equity)
            max_notional = max(max_notional, UNIT_NOTIONAL)
            if candle.high >= stop_loss:
                trade_pnl = close_trade("short", entry_price, stop_loss, entry_fee_rate, TAKER_FEE)
                pnl += trade_pnl
                losses += int(trade_pnl < 0)
                wins += int(trade_pnl > 0)
                trades += 1
                anchor = stop_loss
                side = "flat"
                continue
            if candle.low <= take_profit:
                trade_pnl = close_trade("short", entry_price, take_profit, entry_fee_rate, MAKER_FEE)
                pnl += trade_pnl
                losses += int(trade_pnl < 0)
                wins += int(trade_pnl > 0)
                trades += 1
                anchor = take_profit
                side = "flat"
                continue
        else:
            lower = anchor * (1.0 - step)
            upper = anchor * (1.0 + step)
            touched_lower = candle.low <= lower
            touched_upper = candle.high >= upper
            if touched_lower and touched_upper:
                continue
            if strategy in {"fade_both", "long_dip"} and touched_lower:
                side = "long"
                entry_price = lower
                entry_fee_rate = MAKER_FEE
                take_profit = anchor
                stop_loss = entry_price * (1.0 - step * stop_multiplier)
                max_notional = max(max_notional, UNIT_NOTIONAL)
                continue
            if strategy in {"fade_both", "short_rip"} and touched_upper:
                side = "short"
                entry_price = upper
                entry_fee_rate = MAKER_FEE
                take_profit = anchor
                stop_loss = entry_price * (1.0 + step * stop_multiplier)
                max_notional = max(max_notional, UNIT_NOTIONAL)
                continue
            if strategy == "breakout_both" and touched_upper:
                side = "long"
                entry_price = upper
                entry_fee_rate = TAKER_FEE
                take_profit = entry_price * (1.0 + step)
                stop_loss = anchor
                max_notional = max(max_notional, UNIT_NOTIONAL)
                continue
            if strategy == "breakout_both" and touched_lower:
                side = "short"
                entry_price = lower
                entry_fee_rate = TAKER_FEE
                take_profit = entry_price * (1.0 - step)
                stop_loss = anchor
                max_notional = max(max_notional, UNIT_NOTIONAL)
                continue

    if side == "long":
        quantity = UNIT_NOTIONAL / entry_price
        pnl += quantity * (candles[-1].close - entry_price) - UNIT_NOTIONAL * entry_fee_rate
    elif side == "short":
        quantity = UNIT_NOTIONAL / entry_price
        pnl += quantity * (entry_price - candles[-1].close) - UNIT_NOTIONAL * entry_fee_rate

    margin_required = max_notional / leverage if max_notional > 0 else 1.0
    capital_at_risk = max(1.0, margin_required + max_drawdown)
    return WindowResult(
        pnl=pnl,
        capital_at_risk=capital_at_risk,
        trades=trades,
        wins=wins,
        losses=losses,
    )


def analyze_market(market: Market, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    candles = load_candles(market.name, start_ms, end_ms)
    if len(candles) < WINDOW_POINTS + SHIFT_POINTS:
        return []
    spread_pct = live_spread_pct(market.name)
    rows: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        for step in STEPS:
            for stop_multiplier in STOP_MULTIPLIERS:
                results: list[WindowResult] = []
                for start_index in range(0, len(candles) - WINDOW_POINTS + 1, SHIFT_POINTS):
                    window = candles[start_index : start_index + WINDOW_POINTS]
                    results.append(simulate_window(window, market.max_leverage, strategy, step, stop_multiplier))
                if len(results) < 600:
                    continue
                multipliers = [result.multiplier for result in results]
                trade_counts = [result.trades for result in results]
                if sum(trade_counts) == 0:
                    continue
                positive_windows = sum(1 for result in results if result.pnl > 0) / len(results) * 100
                active_windows = sum(1 for result in results if result.trades > 0) / len(results) * 100
                worst_window = (min(multipliers) - 1.0) * 100
                geo_profit = (geometric_mean(multipliers) - 1.0) * 100
                stability_score = (
                    geo_profit
                    + positive_windows * 0.2
                    + active_windows * 0.1
                    + min(market.day_ntl_vlm / 20_000_000, 5.0)
                    - max(0.0, -worst_window - 20.0) * 1.2
                    - spread_pct * 120.0
                )
                rows.append(
                    {
                        "coin": market.name,
                        "strategy": strategy,
                        "step_pct": step * 100,
                        "stop_multiplier": stop_multiplier,
                        "max_leverage": market.max_leverage,
                        "windows": len(results),
                        "geo_risk_profit_pct": geo_profit,
                        "median_risk_profit_pct": (median(multipliers) - 1.0) * 100,
                        "mean_risk_profit_pct": mean((multiplier - 1.0) * 100 for multiplier in multipliers),
                        "positive_window_pct": positive_windows,
                        "active_window_pct": active_windows,
                        "worst_window_risk_profit_pct": worst_window,
                        "best_window_risk_profit_pct": (max(multipliers) - 1.0) * 100,
                        "avg_trades_per_window": mean(trade_counts),
                        "total_trades": sum(trade_counts),
                        "day_ntl_vlm": market.day_ntl_vlm,
                        "open_interest": market.open_interest,
                        "funding_hourly_pct": market.funding * 100,
                        "live_spread_pct": spread_pct,
                        "candles": len(candles),
                        "stability_score": stability_score,
                    }
                )
    return rows


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - LOOKBACK_DAYS * 24 * 60 * 60 * 1000
    markets = [market for market in load_markets() if market.day_ntl_vlm >= 1_000_000]
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(analyze_market, market, start_ms, end_ms): market.name
            for market in markets
        }
        for future in as_completed(futures):
            coin = futures[future]
            try:
                market_rows = future.result()
            except Exception as exc:
                print(f"skip {coin}: {exc}", flush=True)
                continue
            rows.extend(market_rows)
            print(f"loaded {coin}: {len(market_rows)} strategies", flush=True)
    if not rows:
        raise RuntimeError("No strategies analyzed")
    rows.sort(key=lambda row: row["stability_score"], reverse=True)
    output_path = OUT_DIR / f"hyperliquid_live_strategy_sweep_{time.strftime('%Y%m%d_%H%M%S', time.gmtime(end_ms / 1000))}.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(output_path)
    for row in rows[:30]:
        print(
            f"{row['coin']:>10} {row['strategy']} step={row['step_pct']:.2f}% "
            f"sl={row['stop_multiplier']:.1f} geo={row['geo_risk_profit_pct']:.2f}% "
            f"win={row['positive_window_pct']:.1f}% active={row['active_window_pct']:.1f}% "
            f"worst={row['worst_window_risk_profit_pct']:.1f}% "
            f"spread={row['live_spread_pct']:.3f}% score={row['stability_score']:.1f}"
        )


if __name__ == "__main__":
    main()
