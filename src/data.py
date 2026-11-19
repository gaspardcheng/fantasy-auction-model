"""Build the modeling table from Data 1 extended (per-game stats, 1996-97 to 2025-26) and its companion
table player_season_extra.csv (advanced stats, team at first / last game, team games played).

One row = (player, target season t). Inputs come only from seasons before t:
lag1 = t-1 (required), lag2 = t-2, lag3 = t-3 (optional; filled with the latest available season + flag).
Outcomes: the player's per-game stats in season t and games played in season t.

Games played are on an 82-game scale: GP x 82 / the team's actual games (TEAM_GAMES_LAST), capped at 82.
1998-99 (50 games), 2011-12 (66), 2019-20 (64 to 75) and 2020-21 (72) were short seasons.

Team change ("moved"): the player's team at the first game of season t (TEAM_FIRST_GAME, known by draft
day) differs from his team at the last game of t-1. Relocated franchises count as the same team.
The team in Data 1 is the player's final team, which leaks mid-season trades, so it is not used.

Players who missed ALL of season t with an injury (no row in Data 1, but listed injured on >= 20 official
injury reports during t) are kept with games played = 0. Without them, a GM's worst outcome, paying for
someone who never plays, would be invisible to the evaluation. Other absentees (retired, overseas, unsigned)
are dropped. Season t's injury reports are only used for this outcome label, never as an input.
Official injury reports exist from 2018-19 (starting 17 Dec 2018), so this label, and every injury input,
is only available from then on; before that, injury inputs are MISSING (see injury_known), not zero.
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HISTORY = DATA / "Data 1 extended.csv"
EXTRA = DATA / "player_season_extra.csv"
REPORT_FILES = [DATA / "injury_reports_pre2021.csv", DATA / "injury_reports.csv"]
for _f in (HISTORY, EXTRA):
    if not _f.exists():
        raise FileNotFoundError(f"{_f} is missing. These files are not in the public repo (NBA data terms "
                                f"unconfirmed); copy them from Project1_data_package/data/ or rebuild them "
                                f"with Project1_data_package/scripts/run_all.sh")

SEASONS = sorted(pd.read_csv(HISTORY, usecols=["SEASON"])["SEASON"].unique())
SIDX = {s: i for i, s in enumerate(SEASONS)}
FIRST_TARGET = SEASONS[3]          # first target season with three real lag seasons (1999-00)
TEST_SEASON = SEASONS[-1]          # hold-out season, touched once
INJURY_FIRST = "2018-19"           # first season with official injury reports (from 17 Dec 2018)
INJURY_FULL = "2019-20"            # first season with reports covering the whole season
GAMES = 82

STATS = ["GP", "MIN", "FGM", "FGA", "FTM", "FTA", "FG3M", "FG3A", "PTS", "REB", "AST", "STL", "BLK", "TOV"]
ROLE = ["USG_PCT", "TS_PCT", "AST_PCT", "REB_PCT", "PACE"]       # role / efficiency signals, lagged like STATS
TARGETS = ["FGM", "FGA", "FTM", "FTA", "FG3M", "PTS", "REB", "AST", "STL", "BLK", "TOV"]
HEALTH = ("Injury/Illness", "Return to Competition", "Concussion Protocol", "Rest", "Health and Safety")
MISSED_SEASON_DAYS = 20
# relocated franchises -> current abbreviation, so a relocation is not read as a team change
FRANCHISE = {"SEA": "OKC", "VAN": "MEM", "NJN": "BKN", "NOH": "NOP", "NOK": "NOP", "CHH": "NOP"}


def load_history() -> pd.DataFrame:
    h = pd.read_csv(HISTORY)
    ex = pd.read_csv(EXTRA).drop(columns="PLAYER_NAME")
    h = h.merge(ex, on=["SEASON", "PLAYER_ID"], how="left", validate="1:1")
    h["GP_RAW"] = h["GP"]
    h["GP"] = (h["GP"] * GAMES / h["TEAM_GAMES_LAST"]).clip(upper=GAMES)
    for c in ("TEAM_FIRST_GAME", "TEAM_LAST_GAME"):
        h[c] = h[c].replace(FRANCHISE)
    h["s"] = h["SEASON"].map(SIDX)
    return h


def norm_name(s):                                              # also used to match Data 2 and report names
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower().replace("-", " "))
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def injury_known(target_s, lag=1):
    """True where season target_s - lag has a full season of official injury reports. Injury inputs for
    other seasons are missing: models get them as 0 plus this indicator, never as a plain 0."""
    return (np.asarray(target_s) - lag) >= SIDX[INJURY_FULL]


def load_injury_reports() -> pd.DataFrame:
    """Health-related rows of the official injury reports, with NBA player ids and season index `s`."""
    rep = pd.concat([pd.read_csv(f) for f in REPORT_FILES if f.exists()], ignore_index=True)
    rep = rep[rep["reason"].fillna("").str.startswith(HEALTH)].copy()
    h = load_history()
    h = h[h["s"] >= SIDX[INJURY_FIRST] - 3]                    # name matching: recent players only
    by_season = {(norm_name(n), s): pid for n, s, pid in zip(h["PLAYER_NAME"], h["SEASON"], h["PLAYER_ID"])}
    any_season = {norm_name(n): pid for n, pid in zip(h["PLAYER_NAME"], h["PLAYER_ID"])}
    key = rep["player"].map(norm_name)
    rep["PLAYER_ID"] = [by_season.get((k, s), any_season.get(k)) for k, s in zip(key, rep["season"])]
    rep = rep.dropna(subset=["PLAYER_ID"])
    rep["PLAYER_ID"] = rep["PLAYER_ID"].astype(int)
    rep["s"] = rep["season"].map(SIDX)
    return rep


def lag_features(h: pd.DataFrame, target_s: int) -> pd.DataFrame:
    """Features for predicting season index `target_s` (may be len(SEASONS) = next season; then `moved` is 0
    and the caller sets it from next season's rosters)."""
    by = {k: g.set_index("PLAYER_ID") for k, g in h.groupby("s")}
    base = by.get(target_s - 1)
    if base is None:
        return pd.DataFrame()
    X = pd.DataFrame(index=base.index)
    X["PLAYER_NAME"] = base["PLAYER_NAME"]
    X["age"] = base["AGE"] + 1
    X["age2"] = (X["age"] - 27) ** 2
    prev = None
    for lag in (1, 2, 3):
        g = by.get(target_s - lag)
        cols = pd.DataFrame(index=X.index)
        has = pd.Series(False, index=X.index)
        if g is not None:
            g = g.reindex(X.index)
            has = g["GP"].notna()
            for c in STATS + ROLE:
                cols[c] = g[c]
        else:
            for c in STATS + ROLE:
                cols[c] = np.nan
        if prev is not None:                       # missing older season -> repeat the newer one
            cols = cols.fillna(prev)
        X[f"has_lag{lag}"] = has.astype(int)
        for c in STATS + ROLE:
            X[f"{c}_l{lag}"] = cols[c]
        prev = cols
    X["n_seasons"] = X[["has_lag1", "has_lag2", "has_lag3"]].sum(axis=1)
    X["min_trend"] = X["MIN_l1"] - X["MIN_l2"]
    X["gp_mean"] = X[["GP_l1", "GP_l2", "GP_l3"]].mean(axis=1)
    X["height"] = base["PLAYER_HEIGHT_INCHES"].fillna(h["PLAYER_HEIGHT_INCHES"].median())
    X["traded_l1"] = (base["N_TEAMS"] > 1).astype(int)
    nxt = by.get(target_s)
    first = nxt["TEAM_FIRST_GAME"].reindex(X.index) if nxt is not None else pd.Series(np.nan, index=X.index)
    X["moved"] = (first.notna() & (first != base["TEAM_LAST_GAME"])).astype(int)
    X["target_s"] = target_s
    return X.reset_index()


def build_panel() -> pd.DataFrame:
    h = load_history()
    rep = load_injury_reports()
    rows = []
    for t in range(SIDX[FIRST_TARGET], len(SEASONS)):
        X = lag_features(h, t)
        y = h[h["s"] == t].set_index("PLAYER_ID")[["GP", "MIN"] + TARGETS].add_prefix("y_")
        rows.append(X.merge(y, left_on="PLAYER_ID", right_index=True, how="inner"))
        if t >= SIDX[INJURY_FIRST]:                            # missed the whole season injured -> 0 games
            out = rep[(rep["s"] == t) & (rep["status"] == "Out")].groupby("PLAYER_ID")["date"].nunique()
            missed = X[~X["PLAYER_ID"].isin(y.index) & X["PLAYER_ID"].isin(out[out >= MISSED_SEASON_DAYS].index)]
            rows.append(missed.assign(**{c: 0.0 for c in y.columns}))
    panel = pd.concat(rows, ignore_index=True)
    panel["season"] = panel["target_s"].map(dict(enumerate(SEASONS)))
    return panel


def feature_cols(panel):
    return [c for c in panel.columns if c not in ("PLAYER_ID", "PLAYER_NAME", "season", "target_s")
            and not c.startswith("y_")]


if __name__ == "__main__":
    p = build_panel()
    print(p.shape)
    print(p.groupby("season").size())
