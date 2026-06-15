# hyperliquid-vol-scan

Research + live trading stack for a **stock ⇄ perp dislocation–reversion edge** on
Hyperliquid's builder-dex equity perpetuals.

> ⚠️ **Experimental. Not financial advice.** This trades real money on high
> leverage against an edge whose downside tail is *not fully sampled* by the
> backtest (see [Risks](#risks)). Position sizes here are deliberately tiny
> ($140 of live risk capital). Do not run this with money you can't lose.

---

## The edge in one paragraph

Hyperliquid lists **perpetual futures on US stocks** (NVDA, TSLA, MSTR, …) on
builder-operated dexes (`cash:` and `xyz:`). Each perp has its **own 24/7 order
book** that is only loosely tethered to the underlying equity. Intraday, the perp
price drifts away from the real stock — a **premium/discount that mean-reverts**
over a horizon of seconds-to-minutes. We measure the live stock price (IEX/Finnhub
trades), compute the premium and its microstructure, and a model predicts the
**forward-return quantiles** of the perp. When the predicted reversion exceeds the
round-trip cost, we take the perp side of the dislocation and exit as it closes.

This is a **market-microstructure / statistical-arbitrage** edge, not a
directional bet on stocks.

## Why this only works on Hyperliquid (and not a classical exchange)

The edge **structurally requires a tradable derivative that tracks an asset but
has its own independent price discovery**. Hyperliquid's stock perps are exactly
that: a separate 24/7 book that can dislocate from the underlying.

On a **classical exchange this setup does not exist**:

- The stock *is* the stock — there is no second, independently-priced book of
  "the same NVDA" to revert against.
- Listed equity options / single-stock futures are either far less granular,
  wider-spread, or arbitraged tight by HFT within microseconds.
- The 24/7 + retail-flow + builder-dex microstructure that *creates* the
  dislocation is specific to HL.

**Do not try to port this to Binance/Bybit/CME/IBKR.** The same code pointed at a
classical venue has nothing to revert against and will only pay fees. The edge is
a property of HL's stock-perp design, not of the model.

---

## How it works

```
  IEX/Finnhub stock trades ─┐
                            ├─► per-second bars ─► features (premium, z-scores,
  HL perp trades (builder) ─┘                       reversion stretch, vols, flow…)
                                                          │
                                          neural quantile MLP ensemble (q10…q90)
                                                          │
                                       hysteresis on q25/q50/q75 vs measured cost
                                                          │
                                            target position ─► HL market orders
```

- **Features** (`rl_research/features_v2.py`): one causal, leak-free function
  `compute_features(...)` is the **single source of truth** used by *both* the
  backtest and the live engine — verified byte-identical, so they cannot drift.
  66 features (35 market signals + a per-pair one-hot over 31 pairs): perp/stock
  return ladders, premium + multi-scale premium z-scores,
  premium volatility, mean-reversion "stretch", premium range-position, realized
  vols, order-flow imbalance, print counts, stock staleness, VWAP deviation,
  session fraction, and a per-pair one-hot.
- **Model**: a quantile-regression MLP (pinball loss, monotone quantile head),
  trained on the GPU (PyTorch/ROCm). Production model is a **decision-level
  5-seed ensemble** (each seed votes a position; we average positions, not
  predictions — averaging predictions shrinks the quantile spread and kills the
  entry signal).
- **Costs**: per-pair effective spread is measured from aggressor flips (not the
  Roll estimator, which underestimates on autocorrelated flow), plus taker fee +
  slippage. The entry threshold scales with cost, so wide-spread pairs are traded
  less.
- **Execution** (`rl_research/live_engine.py`): one process, two websockets
  (HL perp trades + Finnhub stock trades), per-second aggregation, ensemble
  inference on CPU (sub-millisecond), position reconciliation via HL market
  orders. **Safe by default** — runs in dry-run unless `--live`.

---

## Results (backtest) and honest caveats

On 80 trading days (Feb–Jun 2026), out-of-time test split, 1-second action delay
and honest costs, the production ensemble backtests at roughly:

| metric | value |
|---|---|
| per-day return, all 31 pairs | **~+1.84%/day** |
| per-day return, top-8 by validation | **~+3.1%/day** |
| worst test day | positive (~+0.7%) |
| positive test days | 11/11 |
| val correlation (q50 vs forward) | ~0.18 |

(5-seed decision-level ensemble; stocks listing on both `cash` and `xyz` dexes
are kept as distinct pairs, which improved the all-pairs return ~+13% and the
top-8 ~+27% over the single-dex set by diversification + per-book optionality.)

**These numbers are real but must be read with the caveats below — they are NOT a
promise of live returns.**

- **Censored tail (the big one):** *every* day in every test window is positive.
  The sample contains **no crash/stress day**, so the true downside is
  unmeasured. This makes backtest-derived leverage dangerous — a single bad day
  not in the sample can wipe a leveraged book.
- **PnL is harness/seed-sensitive:** absolute PnL swings ±0.2–0.4%/day across
  random seeds and ~1.5× across measurement harnesses. `val_corr` is the stable
  metric; always seed-average and compare configs within one harness.
- **Train/serve data skew:** the model trained on **IEX** trades; the live stock
  feed (Finnhub free) is **consolidated**, not IEX. Premium level and
  print-timing features differ — only live A/B vs backtest will reveal the gap.
- **Optimistic fills / capacity:** backtest assumes taker fills at mid+slippage.
  Naive *maker* execution was measured and is **worse** (adverse selection: ~95%
  of resting orders miss, fills are negatively selected). Capacity is tiny.
- **No git-tracked artifacts:** results are reproducible from the scripts, not
  shipped as files.

## Potential profit (and why not to extrapolate)

At $140 with the configured leverage the *modeled* daily edge is a few dollars a
day. **Do not naively compound it** — the censored tail means the realized
distribution has a left side the backtest never saw. The honest framing: this is
a **live experiment to discover the real distribution**, sized so the worst case
(total loss of the $140) is acceptable. Treat any profit as a hypothesis under
test, not income.

---

## History (how this came to be)

1. **Breakout/fade policy search** (CEM over candle policies) — looked great on
   15m candles, **collapsed to negative** under gap-aware fills on 1m replay and
   3 months of Binance data. The "edge" was a fill-price artifact. Hard lesson:
   a TP/SL smaller than the bar range makes candle backtests fantasy.
2. **Pivot to stock ⇄ perp lead-lag / microstructure reversion** — this
   *survived* the full gauntlet (out-of-time, honest costs, gap-aware fills,
   cross-checks). The dislocation is real and reverts.
3. **Feature + model iteration** — XGBoost quantiles → LightGBM → neural quantile
   MLP (best & GPU-fast). Feature sweep found long-memory + multi-scale premium
   structure beats the baseline; a decision-level seed-ensemble removes the seed
   lottery. Both-dex pairs (`cash:` and `xyz:` of the same stock) added as
   distinct pairs for more data and per-book optionality.
4. **Live engine** — single feature path shared with the backtest, dry-run-safe.

## Current stage

- ✅ Backtested with honest validation; production ensemble trained & saved.
- ✅ Live engine built; data feeds + feature parity + inference verified offline.
- ⏳ **Going live** with $140 to measure the real fill quality, data skew, and —
  critically — the **first down day**.

## Plans

- Live paper/real A/B: realized PnL vs backtest, feature-drift vs IEX.
- Measure the real worst-day before any leverage increase.
- Per-stock dex selection (trade the tighter-spread / deeper book).
- Possibly a market-making variant with a dedicated target (the directional
  model cannot be reused for maker — measured).

---

## Repository structure

### Core pipeline (`rl_research/`)
| file | role |
|---|---|
| `features_v2.py` | **single feature path** — `compute_features` + episode builder, the one source of truth for backtest & live |
| `live_features.py` | `FeatureStream` — live ring-buffer that calls the same `compute_features` |
| `nn_train.py` | neural quantile MLP (pinball loss, monotone head), GPU training |
| `train_production.py` | trains the 5-seed **decision-level ensemble**, saves the live bundle + reports |
| `xgb2_train.py` | XGBoost quantile baseline + shared backtest (`portfolio_report`, `quantile_positions`, `step_position`) |
| `lgb_train.py` | LightGBM quantile baseline |
| `feat_explore.py` / `feat_ab*.py` | feature-set sweeps and A/B harnesses |
| `maker_sim.py` | maker-execution simulator (shows naive maker loses to adverse selection) |
| `live_engine.py` | **live trading engine** (HL + Finnhub feeds → ensemble → orders), dry-run by default |

### Data acquisition (`rl_research/`)
| file | role |
|---|---|
| `fetch_range.py` | orchestrates per-day backfill (IEX stock + HL perp) over a date range |
| `backfill_hl_trades.py` | HL perp fills from the S3 node-fills archive (requester-pays) |
| `iex_daily_extract.py` / `iex_tops_stream.py` | IEX HIST (TOPS) stock-trade extraction |

### Other live bots (earlier experiments)
`jto_breakout_bot.py`, `grid_bot.py`, `spcx_leadlag_bot.py`, `scan_hyperliquid.py`, … —
prior strategies on HL; kept for reference. Their `*config*.yaml` are gitignored
(hold an account address); use the `*.example.yaml` templates.

---

## Reproduce

Requires Python 3, an AWS account for the HL archive (requester-pays), a Finnhub
API key for live stock data, and the Hyperliquid Python SDK.

```bash
# 1. backfill tick data (IEX stock + HL perp) for a date range
python rl_research/fetch_range.py 20260302 20260612

# 2. train the production ensemble (builds the feature cache, saves the bundle)
DATES=$(ls out/rl_research/hl_archive/trades | tr '\n' ',')
python rl_research/train_production.py --dates "$DATES" --seeds 5

# 3. live engine — DRY RUN (no orders, logs intended trades)
export FINNHUB_API_KEY=...           # free key from finnhub.io
export HYPERLIQUID_SECRET_KEY=...    # API wallet key (cannot withdraw funds)
cp rl_research/live_engine_config.example.yaml rl_research/live_engine_config.yaml
# edit account_address in the config, then:
python rl_research/live_engine.py --config rl_research/live_engine_config.yaml

# 4. go live (real orders) — only after validating the dry-run
python rl_research/live_engine.py --config rl_research/live_engine_config.yaml --live
```

Funding note: `cash` and `xyz` are **separate builder dexes with separate
margin** — USDC must be deposited into *both* perp accounts to trade pairs on each.

---

## What this is NOT

- **Not** an edge that exists on classical exchanges — see above.
- **Not** safe at scale: capacity is tiny; size beyond a few hundred dollars and
  market impact + the unsampled tail dominate.
- **Not** validated on a down market: the backtest never saw a losing day.
- **Not** financial advice, and **not** a product. A personal research bet.
