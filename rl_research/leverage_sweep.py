from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

from policy_search import (
    OUT_DIR,
    UNIT_NOTIONAL,
    Candle,
    Market,
    Policy,
    build_windows,
    load_markets,
    simulate_window,
)
from validate_1m_replay import load_rows, row_policy
from validate_binance_history import fetch_month, month_starts


def equity_curve(
    windows: list[list[Candle]],
    market: Market,
    policy: Policy,
    leverage: float,
) -> dict[str, Any]:
    maintenance_rate = 1.0 / (2 * market.max_leverage)
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    worst_day = 0.0
    best_day = 0.0
    liquidated = False
    days = 0
    for window in windows:
        day_start = equity
        for pnl in simulate_window(window, market, policy).trade_pnls:
            trade_return = pnl / UNIT_NOTIONAL
            intra_equity = equity * (1.0 - leverage * (policy.sl_pct + market.taker_slippage))
            if leverage * maintenance_rate >= 1.0 - leverage * (policy.sl_pct + market.taker_slippage) or intra_equity <= 0:
                liquidated = True
                equity = 0.0
                break
            equity *= 1.0 + leverage * trade_return
            if equity <= 0:
                liquidated = True
                equity = 0.0
                break
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, 1.0 - equity / peak)
        days += 1
        if liquidated:
            break
        day_return = equity / day_start - 1.0
        worst_day = min(worst_day, day_return)
        best_day = max(best_day, day_return)
    daily_geo = equity ** (1.0 / days) - 1.0 if equity > 0 and days > 0 else -1.0
    return {
        "leverage": leverage,
        "final_multiple": equity,
        "daily_geo_pct": daily_geo * 100,
        "worst_day_pct": worst_day * 100,
        "best_day_pct": best_day * 100,
        "max_drawdown_pct": max_drawdown * 100,
        "days": days,
        "liquidated": liquidated,
    }


def coin_windows(coin: str, months: int) -> list[list[Candle]]:
    candles: list[Candle] = []
    for month in month_starts(months):
        month_candles = fetch_month(f"{coin}USDT", month)
        if month_candles is not None:
            candles.extend(month_candles)
    return build_windows(candles, 24 * 60, 24 * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--coins", required=True)
    parser.add_argument("--months", type=int, default=3)
    parser.add_argument("--max-coins", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wanted = {coin.strip().upper() for coin in args.coins.split(",") if coin.strip()}
    markets = {market.coin: market for market in load_markets(args.max_coins, None)}
    rows = [row for row in load_rows(Path(args.input), top=10**6) if row["coin"] in wanted]
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        coin = row["coin"]
        market = markets[coin]
        policy = row_policy(row)
        windows = coin_windows(coin, args.months)
        if len(windows) < 30:
            print(f"skip {coin}: only {len(windows)} days of binance data", flush=True)
            continue
        leverages = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
        print(f"\n{coin} (HL max leverage {market.max_leverage}x, {len(windows)} days)", flush=True)
        for leverage in leverages:
            exceeds_exchange_cap = leverage > market.max_leverage
            result = equity_curve(windows, market, policy, leverage)
            result.update({"coin": coin, "exceeds_exchange_cap": exceeds_exchange_cap})
            output_rows.append(result)
            cap_note = " (above HL cap)" if exceeds_exchange_cap else ""
            status = "LIQUIDATED" if result["liquidated"] else f"x{result['final_multiple']:.2f}"
            print(
                f"  L={leverage:>4.1f}{cap_note}: {status:>12} over {result['days']}d | "
                f"daily geo {result['daily_geo_pct']:+.2f}% | worst day {result['worst_day_pct']:+.2f}% | "
                f"max DD {result['max_drawdown_pct']:.1f}%",
                flush=True,
            )
    output_path = OUT_DIR / f"leverage_sweep_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"\n{output_path}")


if __name__ == "__main__":
    main()
