#!/bin/bash
set -euo pipefail
DEX="$1"
MIN_VOLUME="$2"
cd "$(dirname "$0")/.."
python rl_research/policy_search.py --dex "$DEX" --min-volume "$MIN_VOLUME" --max-coins 30 --lookback-days 50 --holdout-days 4 --workers 8 > "out/rl_research/scan_15m_$DEX.log" 2>&1
CSV15=$(ls -t out/rl_research/policy_search_*.csv | head -1)
TOP=$(python -c "import csv; rows=sorted(csv.DictReader(open('$CSV15')), key=lambda row: float(row['selection_score']), reverse=True); print(','.join(row['coin'] for row in rows[:8]))")
python rl_research/policy_search.py --coins "$TOP" --dex "$DEX" --interval 5m --lookback-days 17 --window-bars 288 --shift-bars 12 --holdout-days 4 --workers 8 > "out/rl_research/retrain_5m_$DEX.log" 2>&1
CSV5=$(ls -t out/rl_research/policy_search_*.csv | head -1)
python rl_research/validate_1m_replay.py --input "$CSV5" --dex "$DEX" --lookback-days 4 --top 8 > "out/rl_research/replay_1m_$DEX.log" 2>&1
echo "done $DEX"
