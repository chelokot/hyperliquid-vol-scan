# RL Research

This subproject is for offline research only. It does not read live API keys and does not place orders.

The first model is an interpretable discrete policy trained by cross-entropy policy search. The action space is expressive enough to rediscover the live JTO breakout bot:

```text
mode: breakout | fade
side: both | long | short
entry threshold
take profit
stop loss
time stop
cooldown after closed trade
```

## Execution model

The simulator is deliberately pessimistic:

```text
stop-market fills (entry, SL, time stop, final close) pay taker fee plus slippage = half live spread + 2 bps
maker fills (TP, fade entry) require price to trade strictly through the level
SL is checked before TP inside every bar
a bar that hits SL right after entry closes the trade in the same bar
returns are on notional, never divided by margin, leverage is reported separately
```

## Validation protocol

```text
development period: everything except the last holdout days
  train windows (70%): cross-entropy search objective
  validation windows (30%): part of selection score
selection score: train + validation + stress only, never holdout
holdout (last ~4 days): evaluated once, report-only
null calibration: candidate holdout geo vs N random policies on the same holdout
baseline comparison: live JTO breakout parameters
```

## Candle granularity warning

This is the most important lesson so far. Hyperliquid serves at most ~5000 candles per interval, so 1m history is only ~3.6 days deep. Policies whose TP/SL distances are smaller than the average bar range cannot be evaluated on that bar size: a 15m scan showed LIT at +24%/day which collapsed to -6%/day on the 1m replay of the same period. Therefore:

```text
15m scan      -> candidate generation only, numbers are not trusted
5m re-train   -> development on ~17 days of 5m candles
1m replay     -> final report-only verdict on the last ~3.6 days
```

`record_1m_candles.py` (systemd user unit `agents-hl-1m-recorder.service`) continuously archives 1m candles for the top-80 coins into `out/rl_research/candles_1m/`, so future iterations get a deep 1m history that the public API cannot provide.

## Run

```bash
python rl_research/policy_search.py --max-coins 80 --holdout-days 4 --workers 8
python rl_research/policy_search.py --coins <TOP20> --interval 5m --lookback-days 17 --window-bars 288 --shift-bars 12 --holdout-days 4 --workers 8
python rl_research/validate_1m_replay.py --input <5m_csv> --lookback-days 4 --top 20
```

Fast smoke test:

```bash
python rl_research/policy_search.py --coins JTO,LIT,IO --generations 4 --population 40 --null-samples 50
```

A result is interesting only when the 1m replay stays positive under stress cost, beats the null 95th percentile, and the policy's TP/SL distances are comfortably above the 1m bar range. Anything else is treated as granularity artifact or luck.
