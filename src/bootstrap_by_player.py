"""Confidence intervals for our model's error against the benchmarks, bootstrapped by player.

A player appears once per season but in several seasons, so pooled seasons are not independent rows.
The bootstrap resamples PLAYERS (with all their player-seasons) 2,000 times and recomputes
(our MAE / benchmark MAE - 1), for value per game and games played.

Scopes:
  * test 2025-26:         the untouched hold-out season (one row per player, so this is a plain bootstrap).
  * validation 10 folds:  2015-16 to 2024-25, pooled. These folds were used to CHOOSE the model, so this
                          figure is slightly optimistic; it is reported for its larger sample, not as a test.
Output: results/bootstrap_by_player.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
N_BOOT = 2000
BENCHMARKS = ["Marcel (weighted avg)", "Naive (last season)", "Ensemble Ridge+RF"]


def cluster_boot(d, ours, base, col, rng):
    """d: rows of one scope. Returns point estimate and 95% CI of mean(ours)/mean(base) - 1, by player."""
    w = d.pivot_table(index=["PLAYER_ID", "season"], columns="model", values=col)[[ours, base]].dropna()
    g = w.groupby(level="PLAYER_ID").sum()                      # per-player error sums
    n = w.groupby(level="PLAYER_ID").size().reindex(g.index).to_numpy()
    a, b = g[ours].to_numpy(), g[base].to_numpy()
    idx = rng.integers(0, len(g), (N_BOOT, len(g)))
    ratio = a[idx].sum(1) / b[idx].sum(1) - 1
    return {"pct": (a.sum() / b.sum() - 1) * 100, "lo": np.percentile(ratio, 2.5) * 100,
            "hi": np.percentile(ratio, 97.5) * 100, "players": len(g), "player_seasons": int(n.sum()),
            "ours_mae": a.sum() / n.sum(), "base_mae": b.sum() / n.sum()}


def main():
    e = pd.read_csv(RES / "row_errors.csv")
    ours = json.load(open(RES / "selection.json"))["selected_model"]
    rng = np.random.default_rng(0)
    rows = []
    for split in ("test", "validation"):
        d = e[e["split"] == split]
        for base in BENCHMARKS:
            if base == ours or base not in set(d["model"]):
                continue
            for col, metric in (("z_err", "value per game"), ("gp_err", "games played")):
                rows.append({"scope": split, "ours": ours, "vs": base, "metric": metric,
                             **cluster_boot(d, ours, base, col, rng)})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "bootstrap_by_player.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"Error of {ours} relative to each benchmark (%, 95% CI by player, {N_BOOT} resamples):")
    print(out.drop(columns="ours").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
