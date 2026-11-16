"""Model selection (rolling-origin validation) and final hold-out test.

Folds (target season):  validate 2023-24 (train 2022-23), validate 2024-25 (train 2022-23..2023-24)
Hold-out test:           2025-26 (train 2022-23..2024-25) -- touched once, after selection.
Primary metric: fantasy value per game (sum of 9 category z-scores) -- MAE and Spearman rank correlation.
Ties on it (models that share the per-game stats model and differ only in games played) are broken by
games-played MAE. Every "Ensemble Ridge+RF" variant goes to the hold-out test as an ablation of the
injury-report text.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import fantasy as F
from data import TARGETS, build_panel
from models import BASELINES, CANDIDATES, eligible

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
VAL = ["2023-24", "2024-25"]
TEST = "2025-26"


def actual_frame(df):
    a = pd.DataFrame({t: df[f"y_{t}"] for t in TARGETS}, index=df.index)
    a["GP"] = df["y_GP"]
    return a


def evaluate(pred, df):
    """Score predictions against the actual season on the fantasy scale of that season."""
    act = actual_frame(df)
    played = act["GP"] >= 15                                  # per-game accuracy needs a real sample
    ref = F.fit_reference(act[played])
    zp, za = F.z_scores(pred, ref), F.z_scores(act, ref)
    val_a, _ = F.season_value(za["Z"], act["GP"])
    val_p, _ = F.season_value(zp["Z"], pred["GP"])
    top_a = set(val_a.nlargest(F.POOL).index)
    top_p = set(val_p.nlargest(F.POOL).index)
    out = {
        "Z_MAE": float((zp["Z"] - za["Z"])[played].abs().mean()),
        "Z_Spearman": float(spearmanr(zp["Z"][played], za["Z"][played]).statistic),
        "GP_MAE": float((pred["GP"] - act["GP"]).abs().mean()),
        "Value_Spearman": float(spearmanr(val_p, val_a).statistic),
        "Top140_hit": len(top_a & top_p) / F.POOL,
    }
    for c in F.CATS:
        out[f"r_{c}"] = float(np.corrcoef(zp[c][played], za[c][played])[0, 1])
    return out


def run_fold(name, fn, panel, season):
    train = eligible(panel[panel["season"] < season])
    test = eligible(panel[panel["season"] == season])
    pred = fn(train, test)
    return evaluate(pred, test), pred, test


def main():
    RES.mkdir(exist_ok=True)
    panel = build_panel()
    print("rows per target season (eligible):", eligible(panel).groupby("season").size().to_dict())

    # ---------- model selection on validation folds
    rows = []
    for name, fn in CANDIDATES.items():
        for s in VAL:
            m, _, _ = run_fold(name, fn, panel, s)
            rows.append({"model": name, "fold": s, **m})
            print(f"{name:28s} {s}  Z_MAE={m['Z_MAE']:.3f}  rho={m['Z_Spearman']:.3f}  GP_MAE={m['GP_MAE']:.1f}")
    cv = pd.DataFrame(rows)
    cv.to_csv(RES / "validation_folds.csv", index=False)
    summ = cv.groupby("model").mean(numeric_only=True).round({"Z_MAE": 9}).sort_values(["Z_MAE", "GP_MAE"])
    summ.to_csv(RES / "validation_summary.csv")
    ml = summ.drop(index=BASELINES)
    best = ml.index[0]
    best_family = {b.split(" ")[0] for b in ml.index}
    finalists = BASELINES + [ml[ml.index.str.startswith(f)].index[0] for f in sorted(best_family)]
    finalists += [m for m in ml.index if m.startswith("Ensemble Ridge+RF") and m not in finalists]
    print("\nValidation summary:\n", summ[["Z_MAE", "Z_Spearman", "GP_MAE", "Value_Spearman", "Top140_hit"]].round(3))
    print("selected:", best)

    # ---------- hold-out test (once), best configuration of each family + baselines
    test_rows, preds = [], {}
    for name in finalists:
        m, pred, test = run_fold(name, CANDIDATES[name], panel, TEST)
        test_rows.append({"model": name, **m})
        preds[name] = pred
    test_df = pd.DataFrame(test_rows).set_index("model")
    test_df.to_csv(RES / "test_metrics.csv")
    print("\nHold-out 2025-26:\n", test_df[["Z_MAE", "Z_Spearman", "GP_MAE", "Value_Spearman", "Top140_hit"]].round(3))

    # save test-season predictions for the business backtest
    keep = test[["PLAYER_ID", "PLAYER_NAME"]].copy()
    act = actual_frame(test)
    for name, p in preds.items():
        p.add_prefix(f"{name}|").pipe(lambda d: keep.__setitem__(list(d.columns), d.values))
    keep[[f"actual|{c}" for c in act.columns]] = act.values
    keep.to_csv(RES / "test_predictions.csv", index=False)
    json.dump({"selected_model": best, "finalists": finalists}, open(RES / "selection.json", "w"), indent=1)


if __name__ == "__main__":
    main()
