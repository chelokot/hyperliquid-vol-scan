from __future__ import annotations

import csv
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


API_URL = "https://api.hyperliquid.xyz/info"
OUT_DIR = Path(__file__).resolve().parent / "out"
MAKER_FEE = 0.00015
TAKER_FEE = 0.00045
UNIT_NOTIONAL = 100.0
GRID_STEPS = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03)


@dataclass(frozen=True)
class Market:
    name: str
    max_leverage: int
    day_ntl_vlm: float
    open_interest: float
    funding: float
    mark_px: float
    mid_px: float


@dataclass(frozen=True)
class GridResult:
    step_pct: float
    pnl_maker: float
    pnl_taker: float
    trades: int
    total_notional: float
    max_abs_notional: float
    max_drawdown: float
    pnl_on_peak_margin_pct: float


def post(payload: dict[str, Any]) -> Any:
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            response = requests.post(API_URL, json=payload, timeout=20)
            if response.status_code == 429:
                time.sleep(0.75 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
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
                mark_px=as_float(context.get("markPx")),
                mid_px=as_float(context.get("midPx"), as_float(context.get("markPx"))),
            )
        )
    return markets


def load_candles(coin: str, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    return post(
        {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": "1m",
                "startTime": start_ms,
                "endTime": end_ms,
            },
        }
    )


def load_spread(coin: str) -> float | None:
    book = post({"type": "l2Book", "coin": coin})
    bids, asks = book["levels"]
    if not bids or not asks:
        return None
    bid = float(bids[0]["px"])
    ask = float(asks[0]["px"])
    mid = (bid + ask) / 2
    return (ask - bid) / mid


def grid_backtest(prices: list[float], leverage: int, step: float, fee_rate: float) -> GridResult:
    start_price = prices[0]
    current_level = 0
    cash = 0.0
    quantity = 0.0
    fees = 0.0
    trades = 0
    total_notional = 0.0
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
            total_notional += UNIT_NOTIONAL
            trades += 1
        while current_level > target_level:
            current_level -= 1
            execution_price = level_price(current_level)
            trade_quantity = UNIT_NOTIONAL / execution_price
            cash -= UNIT_NOTIONAL
            quantity += trade_quantity
            fees += UNIT_NOTIONAL * fee_rate
            total_notional += UNIT_NOTIONAL
            trades += 1
        equity = cash + quantity * price - fees
        peak_equity = max(peak_equity, equity)
        max_drawdown = max(max_drawdown, peak_equity - equity)
        max_abs_notional = max(max_abs_notional, abs(quantity * price))

    final_equity = cash + quantity * prices[-1] - fees
    peak_margin = max_abs_notional / leverage if leverage > 0 else max_abs_notional
    pnl_on_peak_margin_pct = final_equity / peak_margin * 100 if peak_margin > 0 else 0.0
    return GridResult(
        step_pct=step * 100,
        pnl_maker=final_equity if fee_rate == MAKER_FEE else 0.0,
        pnl_taker=final_equity if fee_rate == TAKER_FEE else 0.0,
        trades=trades,
        total_notional=total_notional,
        max_abs_notional=max_abs_notional,
        max_drawdown=max_drawdown,
        pnl_on_peak_margin_pct=pnl_on_peak_margin_pct,
    )


def best_grid(prices: list[float], leverage: int) -> tuple[GridResult, GridResult]:
    best_maker: GridResult | None = None
    best_taker: GridResult | None = None
    for step in GRID_STEPS:
        maker = grid_backtest(prices, leverage, step, MAKER_FEE)
        taker = grid_backtest(prices, leverage, step, TAKER_FEE)
        if best_maker is None or maker.pnl_on_peak_margin_pct > best_maker.pnl_on_peak_margin_pct:
            best_maker = maker
        if best_taker is None or taker.pnl_on_peak_margin_pct > best_taker.pnl_on_peak_margin_pct:
            best_taker = taker
    if best_maker is None or best_taker is None:
        raise RuntimeError("No grid result")
    return best_maker, best_taker


def analyze_market(market: Market, start_ms: int, end_ms: int) -> dict[str, Any] | None:
    candles = load_candles(market.name, start_ms, end_ms)
    if len(candles) < 720:
        return None
    candles = sorted(candles, key=lambda candle: int(candle["t"]))
    closes = [float(candle["c"]) for candle in candles]
    highs = [float(candle["h"]) for candle in candles]
    lows = [float(candle["l"]) for candle in candles]
    volumes = [float(candle["v"]) for candle in candles]
    abs_log_move = sum(abs(math.log(closes[index] / closes[index - 1])) for index in range(1, len(closes)))
    net_log_move = math.log(closes[-1] / closes[0])
    range_pct = (max(highs) / min(lows) - 1.0) * 100
    drift_pct = (closes[-1] / closes[0] - 1.0) * 100
    approx_candle_ntl = sum(volume * close for volume, close in zip(volumes, closes, strict=True))
    chop_ratio = abs_log_move / max(abs(net_log_move), 0.0001)
    try:
        spread = load_spread(market.name)
    except Exception:
        spread = None
    best_maker, best_taker = best_grid(closes, market.max_leverage)
    spread_pct = spread * 100 if spread is not None else None
    score = (
        best_maker.pnl_on_peak_margin_pct
        + min(chop_ratio, 250.0) * 0.12
        + min(range_pct, 80.0) * 0.6
        - abs(drift_pct) * 3.0
        - (spread_pct or 0.0) * 80.0
    )
    return {
        "coin": market.name,
        "max_leverage": market.max_leverage,
        "funding_hourly_pct": market.funding * 100,
        "day_ntl_vlm": market.day_ntl_vlm,
        "candle_ntl_vlm": approx_candle_ntl,
        "open_interest": market.open_interest,
        "mark_px": market.mark_px,
        "mid_px": market.mid_px,
        "spread_pct": spread_pct,
        "candles": len(candles),
        "ret_24h_pct": drift_pct,
        "range_24h_pct": range_pct,
        "abs_move_24h_pct": abs_log_move * 100,
        "chop_ratio": chop_ratio,
        "best_maker_step_pct": best_maker.step_pct,
        "best_maker_pnl": best_maker.pnl_maker,
        "best_maker_trades": best_maker.trades,
        "best_maker_notional": best_maker.total_notional,
        "best_maker_max_abs_notional": best_maker.max_abs_notional,
        "best_maker_max_drawdown": best_maker.max_drawdown,
        "best_maker_pnl_on_peak_margin_pct": best_maker.pnl_on_peak_margin_pct,
        "best_taker_step_pct": best_taker.step_pct,
        "best_taker_pnl": best_taker.pnl_taker,
        "best_taker_trades": best_taker.trades,
        "best_taker_notional": best_taker.total_notional,
        "best_taker_max_abs_notional": best_taker.max_abs_notional,
        "best_taker_max_drawdown": best_taker.max_drawdown,
        "best_taker_pnl_on_peak_margin_pct": best_taker.pnl_on_peak_margin_pct,
        "score": score,
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - 24 * 60 * 60 * 1000
    markets = [market for market in load_markets() if market.day_ntl_vlm >= 250_000]
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(analyze_market, market, start_ms, end_ms): market.name
            for market in markets
        }
        for future in as_completed(futures):
            coin = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                print(f"skip {coin}: {exc}")
                continue
            if row is not None:
                rows.append(row)
                print(f"loaded {coin}")
    rows.sort(key=lambda row: row["score"], reverse=True)
    if not rows:
        raise RuntimeError("No markets analyzed")
    output_path = OUT_DIR / f"hyperliquid_scan_{time.strftime('%Y%m%d_%H%M%S', time.gmtime(end_ms / 1000))}.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(output_path)
    for row in rows[:20]:
        print(
            f"{row['coin']:>10} score={row['score']:.1f} ret={row['ret_24h_pct']:.2f}% "
            f"range={row['range_24h_pct']:.2f}% abs={row['abs_move_24h_pct']:.1f}% "
            f"spread={row['spread_pct']:.4f}% lev={row['max_leverage']} "
            f"maker_grid={row['best_maker_pnl_on_peak_margin_pct']:.1f}% "
            f"step={row['best_maker_step_pct']:.2f}% trades={row['best_maker_trades']}"
        )


if __name__ == "__main__":
    main()
