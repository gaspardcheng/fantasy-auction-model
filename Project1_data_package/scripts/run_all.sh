#!/usr/bin/env bash
# Runs every step in order. Pass the path to the project's original Data 1.csv for the overlap check.
set -uo pipefail
cd "$(dirname "$0")"
python 01_nba_stats.py
python 02_basketball_reference.py
python 03_injury_report_pdfs.py
python 04_prosportstransactions.py
python 05_kaggle.py || echo "Kaggle skipped"
python 06_build_tables.py
python 07_validate.py "${1:-}"
