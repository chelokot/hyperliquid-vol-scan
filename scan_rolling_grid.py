from __future__ import annotations

import csv
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import requests


API_URL = "https://api.hyperliquid.xyz/info"
OUT_DIR = Path(__file__).resolve().parent / "out"
INTERVAL = "15m"
INTERVAL_MS = 15 * 60 * 1000
WINDOW_HOURS = 24
SHIFT_HOURS = 1
LOOKBACK_DAYS = 30
WINDOW_POINTS = WINDOW_HOURS * 60 // 15
SHIFT_POINTS = SHIFT_HOURS * 60 // 15
UNIT_NOTIONAL = 100.0
MAKER_FEE = 0.00015
TAKER_FEE = 0.00045
GRID_STEPS = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05)


@dataclass(frozen=True)
class Market:
    name: str
    max_leverage: int
    day_ntl_vlm: float
    open_interest: float
    funding: float


@dataclass(frozen=True)
class WindowResult:
    pnl: float
    margin_required: float
    capital_at_risk: float
    trades: int
    max_drawdown: float

    @property
    def margin_multiplier(self) -> float:
        return 1.0 + self.pnl / self.margin_required if self.margin_required > 0 else 1.0

    @property
    def risk_multiplier(self) -> float:
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
            time.sleep(0.6 * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError("Hyperliquid API rate limit")


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


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


def load_closes(coin: str, start_ms: int, end_ms: int) -> list[float]:
    candles = post(
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
    candles = sorted(candles, key=lambda candle: int(candle["t"]))
    return [float(candle["c"]) for candle in candles]


def grid_window(prices: list[float], leverage: int, step: float, fee_rate: float) -> WindowResult:
    start_price = prices[0]
    current_level = 0
    cash = 0.0
    quantity = 0.0
    fees = 0.0
    trades = 0
    peak_equity = 0.0
    max_drawdown = 0.0
    max_abs_notional = 0.0

    def level_price(level: int) -> float:
        return start_price * ((1.0 + step) ** level)

    def price_level(price: float) -> int:
        raw_level = math.log(price / start_price) / math.log(1.0 + step)
        return math.floor(raw_level) if raw_level >= 0 else math.ceil(raw_level)

    for price in prices[1:]:
        target_level = price_level(price)
        while current_level < target_level:
            current_level += 1
            execution_price = level_price(current_level)
            trade_quantity = UNIT_NOTIONAL / execution_price
            cash += UNIT_NOTIONAL
            quantity -= trade_quantity
            fees += UNIT_NOTIONAL * fee_rate
            trades += 1
        while current_level > target_level:
            current_level -= 1
            execution_price = level_price(current_level)
            trade_quantity = UNIT_NOTIONAL / execution_price
            cash -= UNIT_NOTIONAL
            quantity += trade_quantity
            fees += UNIT_NOTIONAL * fee_rate
            trades += 1
        equity = cash + quantity * price - fees
        peak_equity = max(peak_equity, equity)
        max_drawdown = max(max_drawdown, peak_equity - equity)
        max_abs_notional = max(max_abs_notional, abs(quantity * price))

    final_equity = cash + quantity * prices[-1] - fees
    margin_required = max(max_abs_notional / leverage, 1.0)
    capital_at_risk = margin_required + max_drawdown
    return WindowResult(
        pnl=final_equity,
        margin_required=margin_required,
        capital_at_risk=capital_at_risk,
        trades=trades,
        max_drawdown=max_drawdown,
    )


def geometric_mean(multipliers: list[float]) -> float:
    if not multipliers or any(multiplier <= 0 for multiplier in multipliers):
        return 0.0
    return math.exp(sum(math.log(multiplier) for multiplier in multipliers) / len(multipliers))


def max_drawdown(values: list[float]) -> float:
    peak = values[0]
    drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        drawdown = max(drawdown, peak - value)
    return drawdown


def analyze_market(market: Market, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    closes = load_closes(market.name, start_ms, end_ms)
    if len(closes) < WINDOW_POINTS + SHIFT_POINTS:
        return []
    rows: list[dict[str, Any]] = []
    for fee_name, fee_rate in (("maker", MAKER_FEE), ("taker", TAKER_FEE)):
        for step in GRID_STEPS:
            results: list[WindowResult] = []
            equity_curve = [1.0]
            for start_index in range(0, len(closes) - WINDOW_POINTS + 1, SHIFT_POINTS):
                prices = closes[start_index : start_index + WINDOW_POINTS]
                result = grid_window(prices, market.max_leverage, step, fee_rate)
                results.append(result)
                equity_curve.append(equity_curve[-1] * result.risk_multiplier)
            if len(results) < 600:
                continue
            margin_multipliers = [result.margin_multiplier for result in results]
            risk_multipliers = [result.risk_multiplier for result in results]
            pnls = [result.pnl for result in results]
            rows.append(
                {
                    "coin": market.name,
                    "fee_mode": fee_name,
                    "step_pct": step * 100,
                    "max_leverage": market.max_leverage,
                    "windows": len(results),
                    "geo_margin_multiplier": geometric_mean(margin_multipliers),
                    "geo_margin_profit_pct": (geometric_mean(margin_multipliers) - 1.0) * 100,
                    "geo_risk_multiplier": geometric_mean(risk_multipliers),
                    "geo_risk_profit_pct": (geometric_mean(risk_multipliers) - 1.0) * 100,
                    "median_risk_profit_pct": (median(risk_multipliers) - 1.0) * 100,
                    "mean_pnl": mean(pnls),
                    "median_pnl": median(pnls),
                    "positive_window_pct": sum(1 for result in results if result.pnl > 0) / len(results) * 100,
                    "worst_window_risk_profit_pct": (min(risk_multipliers) - 1.0) * 100,
                    "best_window_risk_profit_pct": (max(risk_multipliers) - 1.0) * 100,
                    "compounded_risk_multiplier": equity_curve[-1],
                    "compounded_risk_max_drawdown_pct": max_drawdown(equity_curve) / max(equity_curve) * 100,
                    "avg_trades": mean(result.trades for result in results),
                    "avg_capital_at_risk": mean(result.capital_at_risk for result in results),
                    "day_ntl_vlm": market.day_ntl_vlm,
                    "open_interest": market.open_interest,
                    "funding_hourly_pct": market.funding * 100,
                    "candles": len(closes),
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
    rows.sort(key=lambda row: row["geo_risk_profit_pct"], reverse=True)
    output_path = OUT_DIR / f"hyperliquid_rolling_grid_{time.strftime('%Y%m%d_%H%M%S', time.gmtime(end_ms / 1000))}.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(output_path)
    for row in rows[:25]:
        print(
            f"{row['coin']:>10} {row['fee_mode']} step={row['step_pct']:.2f}% "
            f"geo_risk={row['geo_risk_profit_pct']:.2f}% "
            f"geo_margin={row['geo_margin_profit_pct']:.2f}% "
            f"win={row['positive_window_pct']:.1f}% "
            f"worst={row['worst_window_risk_profit_pct']:.1f}% "
            f"lev={row['max_leverage']} windows={row['windows']}"
        )


if __name__ == "__main__":
    main()
