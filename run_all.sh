#!/usr/bin/env bash
# Reproduce every result in the README, in order.
# Needs data/Data 1 extended.csv, data/player_season_extra.csv and data/injury_reports_pre2021.csv (not in the
# public repo): copy them from Project1_data_package/data/ or rebuild with Project1_data_package/scripts/run_all.sh
# Injury-report text: data/injury_reports.csv is included. To rebuild it from the NBA's PDFs (about 1 hour,
# the CDN rate-limits), run: python3 fetch_injury_reports.py && python3 parse_injury_reports.py
set -euo pipefail
cd "$(dirname "$0")/src"
python3 injury_types.py
python3 train_evaluate.py
python3 bootstrap_by_player.py
python3 miss20_classifier.py
python3 explain.py
python3 backtest.py
python3 make_figures.py
python3 predict_2026.py
python3 build_values.py
python3 build_values_h2h.py
python3 build_values_h2h_14.py
echo "Done. Results in results/, auction boards in docs/index.html (roto) and docs/h2h.html (H2H)"
