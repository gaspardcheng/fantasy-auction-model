# Fantasy Basketball Auction Model: Methods and Full Results

Predict 2026-27 NBA per-game production and games played, turn it into auction dollar values for a
14-team, 9-category rotisserie league, and measure the effect on league standings by replaying a past season.
Trained on 30 seasons of box scores (1996-97 to 2025-26). Injury risk also uses unstructured data: the free text
of the NBA's official injury reports, read with a transformer encoder.

**Prediction task.** For each player-season *t*, use only what is known on draft day (seasons *t-1, t-2, t-3*:
per-game box score, minutes, games, age, usage and efficiency, team change, and the injury reports) to predict
season *t* per-game FGM, FGA, FTM, FTA, 3PM, PTS, REB, AST, STL, BLK, TOV, and games played. Predictions are
scored on the fantasy scale: the sum of 9 category z-scores per game, with FG%/FT% as impact = (pct − league pct)
× attempts and TO negated. Season value = (z/game − replacement z/game) × games / 82, so games played moves value
and price but not z/game.

## Reproduce (about 70 minutes, CPU only)

```bash
python3 -m pip install -r requirements.txt
bash run_all.sh          # injury types → model selection → bootstrap → risk classifier → backtest → figures → boards
open docs/index.html     # live auction board
```

Most of the time is model selection (22 candidates × 11 fits). The backtest alone takes about a minute.

| Step | Script | Output |
|---|---|---|
| Extended data (optional rebuild, ~1 h) | `Project1_data_package/scripts/` | `Data 1 extended.csv`, `player_season_extra.csv`, `injury_reports_pre2021.csv` |
| Injury reports 2021-26 (optional rebuild, ~1 h) | `src/fetch_injury_reports.py`, `src/parse_injury_reports.py` | `data/injury_reports.csv` |
| Injury types, absence, next-season effects | `src/injury_types.py` | `results/injury_*.csv`, `figures/injury_types.png` |
| Training window + model selection + hold-out test | `src/train_evaluate.py` | `results/window_selection.csv`, `validation_*.csv`, `test_metrics.csv`, `test_predictions.csv`, `row_errors.csv`, `selection.json` |
| Confidence intervals, bootstrap by player | `src/bootstrap_by_player.py` | `results/bootstrap_by_player.csv` |
| "Misses 20+ games" risk classifier | `src/miss20_classifier.py` | `results/miss20_classifier.json`, `miss20_rules.txt`, `figures/miss20_calibration.png` |
| What drives the forecast | `src/explain.py` | `results/explain.json` |
| Business backtest (3 rival rooms × 1,000 auctions) | `src/backtest.py` | `results/backtest_summary.csv`, `backtest_sims.csv` |
| Figures | `src/make_figures.py` | `results/figures/*.png`, `bootstrap_ci.json` |
| 2026-27 projections (refit on all seasons) | `src/predict_2026.py` | `results/projections_2026_27.csv` |
| Auction values for the boards | `src/build_values*.py` | `docs/players*.js` |

## Data

* `data/Data 1 extended.csv`: per-game stats for every NBA player, 1996-97 to 2025-26
  (14,569 player-seasons; NBA Stats). Matches the original `data/Data 1.csv` (2021-22 to 2025-26, 2,867
  player-seasons) on all 2,867 rows.
* `data/player_season_extra.csv`: usage %, true shooting %, assist %, rebound %, pace, height,
  team at first and last game, and each team's actual games played.
* `data/injury_reports.csv` and `data/injury_reports_pre2021.csv`: the NBA's official game-day injury
  reports, 17 Dec 2018 to the 2026 Finals (1,689 reports, 136,206 rows of player, status and free-text
  reason; 4,600 distinct reasons). There are no official reports before 2018-19.
* `data/Data 2.xlsx`: third-party 2026-27 projections with Yahoo position eligibility (200 players). Used only for
  the 2026-27 blend, positions and next season's teams, never for training or evaluation.
* `data/Data 3.xlsx`: NBA games per team per week, used by the head-to-head boards only.

The three extended files come from free public sources (`Project1_data_package/docs/SOURCES.md`);
`Project1_data_package/scripts/` rebuilds them.

**Modeling table.** 8,509 player-seasons for target seasons 1999-00 to 2025-26 with a prior-season role
(≥ 20 GP and ≥ 12 MPG in *t-1*); 353 in the 2025-26 test season. Rules:

* Games played are on an 82-game scale (1998-99, 2011-12, 2019-20 and 2020-21 were short).
* Team change uses the team at the first game of *t*, known by draft day; Data 1's team is the final team and
  would leak mid-season trades. Relocated franchises count as the same team.
* Players who missed all of season *t* injured (no games, ≥ 20 days listed Out) are kept with 0 games
  (e.g. Haliburton, Lillard, Irving, VanVleet in 2025-26). This is only detectable from 2018-19 on.
* Injury inputs exist only from 2019-20 (the first full season of reports). Before that they are marked missing
  with an indicator, not set to zero.

## Injury text (unstructured data)

1. **Classify** each reason into body region and injury type. Non-injury strings ("Personal reasons") are dropped.
   Keyword rules label 81% of strings, after fuzzy matching fixes typos in rare words ("Sprian" → sprain).
   A logistic regression on `all-MiniLM-L6-v2` sentence embeddings, trained on those, labels the rest.
   It agrees with the rules 98% in 5-fold CV, but that is measured on the strings the rules can label; the
   encoder-only strings (about 8% of rows) are harder and need a hand-labelled check.
2. **Episodes**: consecutive Out reports for one body region (gap ≤ 14 days); report days ≈ games missed
   (6,712 episodes). Median games missed: surgery 34, tear/rupture 26, fracture 19, strain 7, sprain 4, soreness 2.
3. **Next season** vs same-age players without injury, training seasons only: knee −9.0 games [95% CI −12.4, −5.7],
   surgery −5.6 [−10.6, −0.6], chronic soreness/tendinitis −4.5 [−7.5, −1.5]; ankle and hamstring ≈ 0.
   No injury type changes per-game value significantly.
4. **In the forecast** the text adds little once 30 seasons of history are used: games MAE 14.48 → 14.52 in
   validation with the injury-type flags, 14.46 with a PyTorch network on pooled embeddings
   (`src/text_model.py`). A "misses 20+ games" classifier reaches test AUC 0.654 with the text and 0.648 without.

## Validation design

* **Rolling origin, no leakage.** Ten folds: validate on each season from 2015-16 to 2024-25, trained on every
  season before it. Choose the model on these folds only, then test once on **2025-26**.
* **Training window**, chosen with Ridge on the same folds: every season since 1999-00 (value MAE 1.468, against
  1.491 with the last 3 seasons).
* **Benchmarks.** *Naive*: last season repeats. *Marcel*: 5/4/3 GP-weighted average, regressed to the mean,
  age curve (the rule behind most public rankings).
* **Candidates (22).** Ridge (α ∈ {10, 100, 1000}); elastic net/LASSO with CV penalty; gradient boosting with
  early stopping (4 configs); random forest (3); multi-output neural network (2); nearest-neighbour comparables
  (k ∈ {25, 100}); Ridge + RF average, with games from injury types or an injury-text network; Ridge with games
  from a miss-20 classifier or from boosting; the 2 benchmarks. ML models learn the *change* from last season.
  Ranked by value/game MAE, ties by games MAE. Per-fold table: `results/validation_by_fold.csv`.
* **Selected:** Ridge (α = 10) for per-game stats, gradient boosting for games played (validation value MAE 1.464,
  games MAE 14.39). Ridge beat both benchmarks in all 10 folds. Linear models beat boosting, forests and the
  network on per-game stats by 2–3%; boosting is best on games played.

## Results (hold-out 2025-26, 353 players)

| Model | Value/game MAE | Value/game Spearman | Games MAE | Season-value Spearman |
|---|---|---|---|---|
| Naive (last season) | 1.874 | 0.731 | 18.67 | 0.664 |
| Marcel (weighted avg) | 1.843 | 0.711 | 17.56 | 0.651 |
| Ensemble Ridge + RF | 1.731 | 0.761 | 16.41 | 0.703 |
| Neural network (128 units) | 1.693 | 0.773 | 16.45 | 0.709 |
| **Ridge + boosting games (selected)** | **1.712** | **0.757** | **16.52** | **0.696** |

Bootstrap by player (2,000 resamples; players repeat across seasons, so they are resampled with all their seasons):

| Comparison | Value/game MAE | Games MAE |
|---|---|---|
| vs Marcel, 2025-26 test | −7.1% [−12.2%, −2.0%] | −6.0% [−9.9%, −1.7%] |
| vs Naive, 2025-26 test | −8.6% [−13.2%, −3.9%] | −11.5% [−16.3%, −6.6%] |
| vs Marcel, 10 validation folds pooled* | −10.0% [−11.8%, −8.1%] | −4.0% [−5.6%, −2.5%] |
| vs Ensemble Ridge + RF, 2025-26 test | −1.1% [−3.1%, +1.1%] | +0.7% [−0.9%, +2.3%] |

\* Same folds used for model selection, so slightly optimistic. The top models are statistically tied; the
network was best on the test but 11th of 22 in validation, so it was not selected.

**Business backtest.** 1,000 simulated 14-team $200 auctions for 2025-26 in each of three rival rooms (all rivals
on Marcel, all on Naive, half and half). Rivals bid with ±25% personal noise and we bid with the same noise, so a
GM bidding the room's rule is an average GM (66.9 and 66.8 points against the expected 67.5). Auction rules: whole
dollars, $1 minimum, $1 kept per open spot, second-highest bid + $1, random tie-breaks, unsold if nobody bids $1.
Rosters are scored on actual 2025-26 stats with the 8 × 82 games cap. Rival noise, nominations and tie-breaks are
paired across methods and rooms.

| Our valuation (rivals on Marcel) | Avg roto points (of 126) | Win league | Top 3 |
|---|---|---|---|
| Marcel (= the field) | 66.9 | 8.2% | 22.0% |
| Naive | 74.2 | 12.3% | 38.0% |
| Ensemble Ridge + RF | 77.1 | 21.1% | 50.4% |
| **Our model (selected)** | **76.2** | **17.9%** | **49.4%** |
| Perfect foresight (ceiling) | 96.9 | 91.9% | 99.6% |

| Rivals bid | Our points | vs a Marcel GM | vs a Naive GM | Win league |
|---|---|---|---|---|
| Marcel | 76.2 | +9.3 (s.e. 0.4) | +2.0 (0.4) | 17.9% |
| Naive | 77.7 | +6.0 (0.5) | +11.0 (0.5) | 31.8% |
| Half and half | 74.7 | +8.9 (0.4) | +7.0 (0.4) | 21.8% |

The model adds 7 to 11 roto points over a GM who bids like the room. It and the Ridge + RF ensemble are within a
point of each other in every room.

## Limitations

The auction gain rests on one replayed season (forecast accuracy is tested on 11). Rookies have no history and fall
back to Data 2. Rivals are noisy copies of one or two public rules, and nominations are not strategic. Injury
reports start in Dec 2018, so earlier whole-season absences are not detected. Injury episodes are cut at season
end, so absence for late-season injuries is a lower bound, and injuries after the Finals are not in the text. The
2026-27 board blends our model 50/50 with Data 2, a judgment call.

## Repo map

`src/fantasy.py` holds the league rules and scoring, `src/data.py` the features, `src/models.py` the benchmarks
and models, `src/injury_types.py` and `src/text_model.py` the injury text. `docs/index.html` is the 14-team roto
auction board (`roto12.html`, `h2h.html`, `h2h14.html` cover other league formats), servable by GitHub Pages from
`/docs`. `Project1_data_package/` rebuilds the extended data.
