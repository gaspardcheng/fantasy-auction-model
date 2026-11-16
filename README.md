# Fantasy Basketball Auction Model

Predict 2026-27 NBA per-game production and games played, turn it into auction dollar values for a
14-team, 9-category rotisserie league, and measure the effect on league standings by replaying a past season.
Games played also uses unstructured data: the free text of the NBA's official injury reports, read with a
transformer encoder.

**Prediction task.** For each player-season *t*, use only seasons *t-1, t-2, t-3* (per-game box score, minutes,
games, age, and the injury reports) to predict season *t* per-game FGM, FGA, FTM, FTA, 3PM, PTS, REB, AST, STL,
BLK, TOV, and games played. Predictions are scored on the fantasy scale: the sum of 9 category z-scores per game,
with FG%/FT% as impact = (pct − league pct) × attempts and TO negated. Season value = (z/game − replacement
z/game) × games / 82, so games played moves value and price but not z/game.

## Reproduce (about 4 minutes, CPU only)

```bash
python3 -m pip install -r requirements.txt
bash run_all.sh          # injury types → train/evaluate → backtest → figures → 2026-27 projections → boards
open docs/index.html     # live auction board
```

| Step | Script | Output |
|---|---|---|
| Injury reports (optional rebuild, ~1 h) | `src/fetch_injury_reports.py`, `src/parse_injury_reports.py` | `data/injury_reports.csv` |
| Injury types, absence, next-season effects | `src/injury_types.py` | `results/injury_*.csv`, `figures/injury_types.png` |
| Model selection + hold-out test | `src/train_evaluate.py` | `results/validation_*.csv`, `test_metrics.csv`, `test_predictions.csv`, `selection.json` |
| Business backtest (1,000 auctions) | `src/backtest.py` | `results/backtest_summary.csv`, `backtest_sims.csv` |
| Figures + bootstrap CIs | `src/make_figures.py` | `results/figures/*.png`, `bootstrap_ci.json` |
| 2026-27 projections (refit on all seasons) | `src/predict_2026.py` | `results/projections_2026_27.csv` |
| Auction values for the board | `src/build_values.py` | `docs/players.js` |

## Data

* `data/Data 1.csv`: per-game stats for every NBA player, 2021-22 to 2025-26 (2,867 player-seasons; NBA Stats export).
* `data/injury_reports.csv`: the NBA's official game-day injury reports, Oct 2021 to the 2026 Finals
  (1,149 PDFs, 96,230 rows of player, status, free-text reason; 3,025 distinct reasons).
* `data/Data 2.xlsx`: third-party 2026-27 projections with Yahoo position eligibility (200 players), included so the
  2026-27 board can be rebuilt. Used only for the 2026-27 blend and positions, never for training or evaluation.
* `data/Data 3.xlsx`: NBA games per team per week, used by the head-to-head boards only.
* Modeling table: 1,387 player-season rows with a prior-season role (≥ 20 GP and ≥ 12 MPG in *t-1*); 353 in the
  2025-26 test season. Players who then missed all of season *t* injured (no games, ≥ 20 days listed Out) are kept
  with 0 games (16 rows, e.g. Haliburton, Lillard, Irving in 2025-26); season *t* reports are used only for that label.

## Injury text (unstructured data)

1. **Classify** each reason into body region and injury type. Keyword rules label 79% of the strings; a logistic
   regression on `all-MiniLM-L6-v2` sentence embeddings, trained on those, labels the rest (typos, rare wording;
   98% 5-fold agreement with the rules).
2. **Episodes**: consecutive Out reports for one body region (gap ≤ 14 days); report days ≈ games missed.
   Median games missed: tear/rupture 36, surgery 34.5, fracture 19, strain 7, sprain 4, soreness 2.
3. **Next season** vs same-age players without injury: Achilles −15.7 games, tear/rupture −7.2, knee −6.5
   [95% CI −10.2, −3.0], chronic soreness/tendinitis −4.9 [−8.3, −1.1]; ankle sprains and hamstring strains ≈ 0.
   No injury type changes per-game value significantly. So the flags (knee, Achilles, tear/surgery, chronic,
   still out at season end, games missed) go into the games-played model only.
4. **Ablation**: pooled embeddings fed to a PyTorch network (`src/text_model.py`) made games played worse.

## Validation design

* **Rolling origin, no leakage.** Validate on 2023-24 (train 2022-23) and 2024-25 (train 2022-23 to 2023-24),
  choose the model, then test once on **2025-26** (train 2022-23 to 2024-25).
* **Benchmarks.** *Naive*: last season repeats. *Marcel*: 5/4/3 GP-weighted average, regressed to the mean,
  age curve (the rule behind most public rankings).
* **Candidates.** Ridge (α ∈ {10, 100, 1000}), Random Forest (min leaf ∈ {5, 15, 30}), HistGradientBoosting
  (4 configs), a Ridge + RF average, and that average with games played from the injury text (network or
  injury-type flags). ML models learn the *change* from last season. Ranked by value/game MAE, ties by games MAE.
* **Selected:** Ensemble Ridge(α=100) + RandomForest(leaf=15) with injury-type games (validation games MAE
  14.97 vs 15.26 without text, 15.83 with the text network; all candidates in `results/validation_summary.csv`).

## Results (hold-out 2025-26)

| Model | Value/game MAE | Value/game Spearman | Games MAE | Season-value Spearman |
|---|---|---|---|---|
| Naive (last season) | 1.874 | 0.731 | 18.67 | 0.664 |
| Marcel (weighted avg) | 1.843 | 0.710 | 17.56 | 0.650 |
| Ensemble Ridge + RF, no injury text | 1.782 | 0.755 | 17.07 | 0.698 |
| Ensemble + injury-text network games | 1.782 | 0.755 | 17.20 | 0.700 |
| **Ensemble + injury-type games** | **1.782** | **0.755** | **16.95** | **0.698** |

Paired bootstrap (2,000 resamples) vs Naive: value MAE −4.9% [95% CI −7.7%, −1.9%], games MAE −9.2% [−14.4%, −3.9%].
vs Marcel: −3.3% [−9.0%, +2.7%] and −3.5% [−8.5%, +1.1%]. No model foresaw the 6 whole-season absences.

**Business backtest.** 1,000 simulated 14-team $200 auctions for 2025-26. Rivals bid Marcel values with ±25%
personal noise, and we bid with the same noise, so a Marcel bidder is an average GM. Rosters are scored on
actual 2025-26 stats with the 8 × 82 games cap.

| Our valuation | Avg roto points (of 126) | Win league | Top 3 |
|---|---|---|---|
| Marcel (= the field) | 68.0 | 6.4% | 22.2% |
| Naive | 73.9 | 10.8% | 38.5% |
| Our model, no injury text | 74.6 | 15.0% | 41.9% |
| **Our model + injury types** | **74.4** | **14.0%** | **40.9%** |
| Perfect foresight (ceiling) | 93.3 | 82.1% | 96.2% |

vs Marcel: +6.4 points, better in 70% of auctions. The injury types do not move roto points (−0.2, s.e. 0.19):
a 0.7% better games forecast is too small to change who gets bought.

## Limitations

One hold-out season. Rookies have no history and fall back to Data 2. The backtest models rivals as noisy copies
of one public ranking, allows $0 bids and does not reserve $1 per open roster spot. Injury episodes are cut at
season end, so absence for late-season injuries is a lower bound. Season-ending injuries after the Finals are
not in the text.

## Repo map

`src/fantasy.py` holds the league rules and scoring, `src/data.py` the features, `src/models.py` the benchmarks
and models, `src/injury_types.py` and `src/text_model.py` the injury text. `docs/index.html` is the 14-team roto auction
board (`roto12.html`, `h2h.html`, `h2h14.html` cover other league formats), servable by GitHub Pages from `/docs`.
