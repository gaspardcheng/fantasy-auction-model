"""Build the modeling table from Data 1 (per-game stats, 2021-22 to 2025-26).

One row = (player, target season t). Inputs come only from seasons before t:
lag1 = t-1 (required), lag2 = t-2, lag3 = t-3 (optional; filled with the latest available season + flag).
Outcomes: the player's per-game stats in season t and games played in season t.
Players who missed ALL of season t with an injury (no row in Data 1, but listed injured on >= 20 official
injury reports during t) are kept with games played = 0. Without them, a GM's worst outcome, paying for
someone who never plays, would be invisible to the evaluation. Other absentees (retired, overseas, unsigned)
are dropped. Season t's injury reports are only used for this outcome label, never as an input.
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
STATS = ["GP", "MIN", "FGM", "FGA", "FTM", "FTA", "FG3M", "FG3A", "PTS", "REB", "AST", "STL", "BLK", "TOV"]
TARGETS = ["FGM", "FGA", "FTM", "FTA", "FG3M", "PTS", "REB", "AST", "STL", "BLK", "TOV"]
REPORTS = DATA / "injury_reports.csv"
HEALTH = ("Injury/Illness", "Return to Competition", "Concussion Protocol", "Rest", "Health and Safety")
MISSED_SEASON_DAYS = 20


def load_history() -> pd.DataFrame:
    h = pd.read_csv(DATA / "Data 1.csv")
    h = h[h["SEASON"].isin(SEASONS)].copy()
    h["s"] = h["SEASON"].map({s: i for i, s in enumerate(SEASONS)})
    return h


def norm_name(s):                                              # also used to match Data 2 and report names
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower().replace("-", " "))
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def load_injury_reports() -> pd.DataFrame:
    """Health-related rows of the official injury reports, with NBA player ids and season index `s`."""
    rep = pd.read_csv(REPORTS)
    rep = rep[rep["reason"].fillna("").str.startswith(HEALTH)].copy()
    h = load_history()
    by_season = {(norm_name(n), s): pid for n, s, pid in zip(h["PLAYER_NAME"], h["SEASON"], h["PLAYER_ID"])}
    any_season = {norm_name(n): pid for n, pid in zip(h["PLAYER_NAME"], h["PLAYER_ID"])}
    key = rep["player"].map(norm_name)
    rep["PLAYER_ID"] = [by_season.get((k, s), any_season.get(k)) for k, s in zip(key, rep["season"])]
    rep = rep.dropna(subset=["PLAYER_ID"])
    rep["PLAYER_ID"] = rep["PLAYER_ID"].astype(int)
    rep["s"] = rep["season"].map({s: i for i, s in enumerate(SEASONS)})
    return rep


def lag_features(h: pd.DataFrame, target_s: int) -> pd.DataFrame:
    """Features for predicting season index `target_s` (may be len(SEASONS) = next season)."""
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
            for c in STATS:
                cols[c] = g[c]
        else:
            for c in STATS:
                cols[c] = np.nan
        if prev is not None:                       # missing older season -> repeat the newer one
            cols = cols.fillna(prev)
        X[f"has_lag{lag}"] = has.astype(int)
        for c in STATS:
            X[f"{c}_l{lag}"] = cols[c]
        prev = cols
    X["n_seasons"] = X[["has_lag1", "has_lag2", "has_lag3"]].sum(axis=1)
    X["min_trend"] = X["MIN_l1"] - X["MIN_l2"]
    X["gp_mean"] = X[["GP_l1", "GP_l2", "GP_l3"]].mean(axis=1)
    X["target_s"] = target_s
    return X.reset_index()


def build_panel() -> pd.DataFrame:
    h = load_history()
    rep = load_injury_reports() if REPORTS.exists() else None
    rows = []
    for t in range(1, len(SEASONS)):
        X = lag_features(h, t)
        y = h[h["s"] == t].set_index("PLAYER_ID")[["GP", "MIN"] + TARGETS].add_prefix("y_")
        rows.append(X.merge(y, left_on="PLAYER_ID", right_index=True, how="inner"))
        if rep is not None:                                    # missed the whole season injured -> 0 games
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
