#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATES="20260512 20260513 20260514 20260515 20260518 20260519 20260520 20260521 20260522 20260526 20260527 20260528 20260529 20260601 20260602 20260603 20260604 20260605 20260608 20260609 20260610 20260611"
HL_COINS="cash:NVDA,cash:TSLA,cash:META,cash:GOOGL,cash:AMZN,cash:MSFT,cash:HOOD,cash:EWY,xyz:INTC,xyz:CRCL,xyz:MSTR,xyz:CRWV,xyz:MU,xyz:SNDK,xyz:MRVL,xyz:AMD,xyz:PLTR,xyz:COIN,xyz:AAPL,xyz:TSM,xyz:ARM,xyz:AVGO"
for DATE in $DATES; do
  if [ ! -d "out/rl_research/hl_archive/iex/$DATE" ]; then
    echo "=== IEX $DATE"
    python rl_research/iex_daily_extract.py --date "$DATE" --output-dir "out/rl_research/hl_archive/iex/$DATE"
  fi
  if [ ! -f "out/rl_research/hl_archive/trades/$DATE/xyz_MRVL.csv" ]; then
    echo "=== HL $DATE"
    python rl_research/backfill_hl_trades.py --date "$DATE" --hours 11-22 --coins "$HL_COINS"
    rm -rf "out/rl_research/hl_archive/$DATE"
  fi
done
echo fetch done
df -h /var/home | tail -1
