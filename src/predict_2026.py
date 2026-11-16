"""Fit the selected model on all seasons and project 2026-27; blend with the external projections (Data 2).

Output: results/projections_2026_27.csv (one row per player in Data 2), used by build_values.py.
Blend: 50% our model + 50% Data 2 for per-game stats and games played, when both exist.
Rookies / players without NBA history keep the Data 2 projection.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import SEASONS, TARGETS, build_panel, lag_features, load_history
from models import CANDIDATES, eligible
from build_values import load_projections, norm_name

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
W_MODEL = 0.5
MAP = {"FG3M": "3PM", "TOV": "TO"}          # our names -> Data 2 names


def main():
    sel = json.load(open(RES / "selection.json"))["selected_model"]
    panel = build_panel()
    hist = load_history()
    X_next = lag_features(hist, len(SEASONS))                 # features for 2026-27
    X_next = X_next[X_next["GP_l1"] >= 1]
    pred = CANDIDATES[sel](eligible(panel), X_next)
    pred["key"] = X_next["PLAYER_NAME"].map(norm_name).values

    proj = load_projections()
    proj["key"] = proj["name"].map(norm_name)
    m = proj.merge(pred, on="key", how="left", suffixes=("", "_model"))
    m["has_model"] = m["GP_model"].notna()
    for t in TARGETS + ["GP"]:
        d2 = MAP.get(t, t)
        m[f"model_{t}"] = m[f"{t}_model"] if f"{t}_model" in m else m[t]
        m[d2] = np.where(m["has_model"], W_MODEL * m[f"model_{t}"] + (1 - W_MODEL) * m[d2], m[d2])
    keep = ["name", "pos", "team", "ADP", "GP", "MPG", "FGM", "FGA", "FTM", "FTA", "3PM", "PTS", "REB",
            "AST", "STL", "BLK", "TO", "has_model"] + [f"model_{t}" for t in TARGETS + ["GP"]]
    m[keep].to_csv(RES / "projections_2026_27.csv", index=False)
    print(f"{m['has_model'].sum()} of {len(m)} players blended with model ({sel}); rest use Data 2 only")


if __name__ == "__main__":
    main()
