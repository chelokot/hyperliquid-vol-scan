#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
python rl_research/policy_search.py --max-coins 80 --holdout-days 4 --workers 8 > out/rl_research/wide_scan_15m_v2.log 2>&1
CSV15=$(ls -t out/rl_research/policy_search_*.csv | head -1)
TOP=$(python -c "import csv; rows=sorted(csv.DictReader(open('$CSV15')), key=lambda row: float(row['selection_score']), reverse=True); print(','.join(row['coin'] for row in rows[:20]))")
python rl_research/policy_search.py --coins "$TOP" --interval 5m --lookback-days 17 --window-bars 288 --shift-bars 12 --holdout-days 4 --workers 8 > out/rl_research/retrain_5m_v2.log 2>&1
CSV5=$(ls -t out/rl_research/policy_search_*.csv | head -1)
python rl_research/validate_1m_replay.py --input "$CSV5" --lookback-days 4 --top 20 > out/rl_research/replay_1m_v2.log 2>&1
python rl_research/validate_binance_history.py --input "$CSV5" --top 10 --months 3 > out/rl_research/binance_v2.log 2>&1
echo done
