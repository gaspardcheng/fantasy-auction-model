# Project 1 data package (15.071, fantasy auction model)

Extends our original `Data 1.csv` (5 seasons, 2021-22 to 2025-26) to 30 seasons, plus three more seasons of official injury reports. Everything comes from free public sources and passes validation against `Data 1.csv` (2,867/2,867 rows match).

## data/
| File | Rows | Use |
|---|---|---|
| `Data 1 extended.csv` | 14,569 | Same 22 columns as `Data 1.csv` (+ `SOURCE`), 1996-97 to 2025-26. A drop-in replacement. |
| `player_season_extra.csv` | 14,569 | Joins on SEASON + PLAYER_ID. Advanced stats (USG_PCT, TS_PCT, PACE, PIE, ratings), height/weight/draft, TEAM_FIRST_GAME, TEAM_LAST_GAME, N_TEAMS, TEAM_GAMES_LAST, TEAM_GAMES_MAX, SCHEDULED_GAMES. |
| `injury_reports_pre2021.csv` | 39,976 | Official NBA injury reports, 2018-19 to 2020-21. Same columns as the repo's `data/injury_reports.csv` (date, season, player, status, reason); append the two. |

## Rules when using it
1. **Injury features before 2018-19 are MISSING, not zero.** There are no injury reports before then. The 2018-19 reports start on 17 Dec 2018.
2. **Scale games played by the team's actual games** (`TEAM_GAMES_LAST`), not by 82. 1998-99, 2011-12, 2019-20 (64 to 75 games) and 2020-21 were short.
3. **Use `TEAM_FIRST_GAME` for team changes.** It is known by draft day. The team in Data 1 is the final team, which leaks mid-season trades.
4. **Reasons in 2018-19 are worded differently** ("Right ankle sprain" rather than "Right Ankle; Sprain"). Rerun `injury_types.py` with sentence-transformers installed, then check how many come out "Unspecified".
5. **Do not publish these CSVs in the public repo** until NBA data redistribution terms are confirmed. Publish `scripts/` instead.

## docs/
- `validation_report.md`: all checks PASS.
- `download_report.md`: the full download log, with each checkpoint, what failed and why, and the parser fixes.
- `SOURCES.md`: sources and terms of use. prosportstransactions was blocked (403) and Kaggle was skipped (no token).

## scripts/
The download, build and validation scripts that reproduce `data/`. Run `run_all.sh "<path>/Data 1.csv"`. Raw downloads (~240 MB) are not included; the scripts re-fetch them and can resume after an interruption.
