from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from policy_search import MAKER_FEE, TAKER_FEE, UNIT_NOTIONAL, post

OUT_PATH = Path(__file__).resolve().parents[1] / "out" / "rl_research" / "paper_fade_events.jsonl"
POLL_SECONDS = 2.0


@dataclass
class PaperConfig:
    coin: str
    entry_pct: float
    tp_pct: float
    time_stop_minutes: int


CONFIGS = [
    PaperConfig(coin="vntl:OPENAI", entry_pct=0.02, tp_pct=0.02, time_stop_minutes=160),
    PaperConfig(coin="vntl:ANTHROPIC", entry_pct=0.02, tp_pct=0.02, time_stop_minutes=80),
    PaperConfig(coin="vntl:WHEAT", entry_pct=0.015, tp_pct=0.02, time_stop_minutes=80),
]


@dataclass
class PaperState:
    anchor: float
    side: Literal["flat", "long", "short"] = "flat"
    entry_price: float = 0.0
    entry_time: float = 0.0
    seen_trade_ids: set[int] | None = None
    pnl: float = 0.0
    trades: int = 0

    def __post_init__(self) -> None:
        self.seen_trade_ids = set()


def emit(event: dict) -> None:
    event["time_ms"] = int(time.time() * 1000)
    with OUT_PATH.open("a") as handle:
        handle.write(json.dumps(event) + "\n")


def mid_price(coin: str) -> float:
    book = post({"type": "l2Book", "coin": coin})
    bids, asks = book["levels"]
    return (float(bids[0]["px"]) + float(asks[0]["px"])) / 2


def close_trade(config: PaperConfig, state: PaperState, exit_price: float, exit_fee: float, reason: str) -> None:
    quantity = UNIT_NOTIONAL / state.entry_price
    gross = quantity * (exit_price - state.entry_price)
    if state.side == "short":
        gross = -gross
    net = gross - UNIT_NOTIONAL * MAKER_FEE - quantity * exit_price * exit_fee
    state.pnl += net
    state.trades += 1
    emit(
        {
            "type": "close",
            "coin": config.coin,
            "side": state.side,
            "entry": state.entry_price,
            "exit": exit_price,
            "net": net,
            "total_pnl": state.pnl,
            "trades": state.trades,
            "reason": reason,
        }
    )
    state.anchor = exit_price
    state.side = "flat"


def process_coin(config: PaperConfig, state: PaperState) -> None:
    trades = post({"type": "recentTrades", "coin": config.coin})
    fresh = sorted(
        (trade for trade in trades if trade["tid"] not in state.seen_trade_ids),
        key=lambda trade: trade["time"],
    )
    for trade in fresh:
        state.seen_trade_ids.add(trade["tid"])
        price = float(trade["px"])
        if state.side == "flat":
            upper = state.anchor * (1.0 + config.entry_pct)
            lower = state.anchor * (1.0 - config.entry_pct)
            if price < lower:
                state.side = "long"
                state.entry_price = lower
                state.entry_time = time.time()
                emit({"type": "open", "coin": config.coin, "side": "long", "entry": lower, "print": price})
            elif price > upper:
                state.side = "short"
                state.entry_price = upper
                state.entry_time = time.time()
                emit({"type": "open", "coin": config.coin, "side": "short", "entry": upper, "print": price})
        elif state.side == "long" and price > state.entry_price * (1.0 + config.tp_pct):
            close_trade(config, state, state.entry_price * (1.0 + config.tp_pct), MAKER_FEE, "tp")
        elif state.side == "short" and price < state.entry_price * (1.0 - config.tp_pct):
            close_trade(config, state, state.entry_price * (1.0 - config.tp_pct), MAKER_FEE, "tp")
    if state.side != "flat" and time.time() - state.entry_time >= config.time_stop_minutes * 60:
        close_trade(config, state, mid_price(config.coin), TAKER_FEE, "time_stop")
    if len(state.seen_trade_ids) > 50_000:
        state.seen_trade_ids = set(sorted(state.seen_trade_ids)[-10_000:])


def main() -> None:
    states = {config.coin: PaperState(anchor=mid_price(config.coin)) for config in CONFIGS}
    for config in CONFIGS:
        emit({"type": "started", "coin": config.coin, "anchor": states[config.coin].anchor})
    while True:
        for config in CONFIGS:
            try:
                process_coin(config, states[config.coin])
            except Exception as exc:
                emit({"type": "error", "coin": config.coin, "error": str(exc)})
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
