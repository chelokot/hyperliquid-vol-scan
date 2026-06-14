#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATE=20260612
until curl -sf "https://iextrading.com/api/1.0/hist?date=$DATE" > /dev/null; do
  echo "$(date -u '+%H:%M') file not published yet, waiting 30 min"
  sleep 1800
done
echo "file published, fast extraction starts"
python rl_research/iex_daily_extract.py --date "$DATE" --output-dir "out/rl_research/hl_archive/iex/$DATE"
python rl_research/compare_iex_hl.py --date "$DATE" --pairs "SPCX=xyz:SPCX,SNDK=xyz:SNDK,MU=xyz:MU,CRCL=xyz:CRCL"
echo all done
