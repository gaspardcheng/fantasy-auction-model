"""Model selection (rolling-origin validation) and final hold-out test, on 1999-00 to 2025-26 target seasons.

Training window: the N target seasons before the season being predicted. N is chosen first, with Ridge,
on the validation folds only (N = 3 is about what the original 5-season data allowed).
Folds (target season):  validate 2015-16, 2016-17, ..., 2024-25 (10 folds), each trained on the N seasons before it.
Hold-out test:           2025-26 (trained on the N seasons before it) -- touched once, after selection.
Primary metric: fantasy value per game (sum of 9 category z-scores) -- MAE and Spearman rank correlation,
averaged over folds; the per-fold table is in results/validation_by_fold.csv.
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
from data import SEASONS, SIDX, TARGETS, TEST_SEASON, build_panel
from models import BASELINES, CANDIDATES, eligible, window

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TEST = TEST_SEASON
VAL = SEASONS[SIDX["2015-16"]:SIDX[TEST]]
WINDOWS = [3, 5, 10, 15, None]            # training seasons tried (None = every season since 1999-00)
WINDOW_MODEL = "Ridge a=100"


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


def run_fold(name, fn, panel, season, n_seasons):
    train = eligible(window(panel, season, n_seasons))
    test = eligible(panel[panel["season"] == season])
    pred = fn(train, test)
    return evaluate(pred, test), pred, test


def main():
    RES.mkdir(exist_ok=True)
    panel = build_panel()
    print("rows per target season (eligible):", eligible(panel).groupby("season").size().to_dict())

    # ---------- training window, chosen with one model on the validation folds
    wrows = []
    for n in WINDOWS:
        for s in VAL:
            m, _, _ = run_fold(WINDOW_MODEL, CANDIDATES[WINDOW_MODEL], panel, s, n)
            wrows.append({"train_seasons": n or 0, "fold": s, **m})
    wins = pd.DataFrame(wrows).groupby("train_seasons")[["Z_MAE", "GP_MAE", "Value_Spearman"]].mean()
    wins.to_csv(RES / "window_selection.csv")
    n_best = int(wins["Z_MAE"].idxmin()) or None
    print(f"training window ({WINDOW_MODEL}, 0 = all seasons):\n{wins.round(4)}\nchosen: {n_best or 'all'} seasons")

    # ---------- model selection on validation folds
    rows = []
    for name, fn in CANDIDATES.items():
        for s in VAL:
            m, _, _ = run_fold(name, fn, panel, s, n_best)
            rows.append({"model": name, "fold": s, **m})
            print(f"{name:28s} {s}  Z_MAE={m['Z_MAE']:.3f}  rho={m['Z_Spearman']:.3f}  GP_MAE={m['GP_MAE']:.1f}")
    cv = pd.DataFrame(rows)
    cv.to_csv(RES / "validation_folds.csv", index=False)
    cv.pivot(index="model", columns="fold", values="Z_MAE").round(4).to_csv(RES / "validation_by_fold.csv")
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
        m, pred, test = run_fold(name, CANDIDATES[name], panel, TEST, n_best)
        test_rows.append({"model": name, **m})
        preds[name] = pred
    test_df = pd.DataFrame(test_rows).set_index("model")
    test_df.to_csv(RES / "test_metrics.csv")
    print(f"\nHold-out {TEST}:\n", test_df[["Z_MAE", "Z_Spearman", "GP_MAE", "Value_Spearman", "Top140_hit"]].round(3))

    # save test-season predictions for the business backtest
    keep = test[["PLAYER_ID", "PLAYER_NAME"]].copy()
    act = actual_frame(test)
    for name, p in preds.items():
        p.add_prefix(f"{name}|").pipe(lambda d: keep.__setitem__(list(d.columns), d.values))
    keep[[f"actual|{c}" for c in act.columns]] = act.values
    keep.to_csv(RES / "test_predictions.csv", index=False)
    json.dump({"selected_model": best, "finalists": finalists, "train_seasons": n_best,
               "validation_folds": VAL, "test_season": TEST}, open(RES / "selection.json", "w"), indent=1)


if __name__ == "__main__":
    main()
