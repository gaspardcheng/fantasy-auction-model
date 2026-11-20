"""Injury risk as a probability: will a player miss 20+ of 82 games next season?

Two readable classifiers on the same inputs (age, games and minutes in the last three seasons, and the
injury-report flags of last season's main injury):
  * logistic regression (coefficients = odds multipliers per standard deviation)
  * a shallow decision tree (depth 3, at least 60 players per leaf) whose leaves are plain-language risk rules
Trained on target seasons 2020-21 to 2024-25 (the seasons with a full previous year of injury reports),
tested once on 2025-26. A history-only logistic model, trained on every season since 1999-00, shows what
the injury text adds.
Reported: AUC, Brier score, calibration by risk band, and the threshold chosen on training data
(cross-validated predictions, maximising sensitivity + specificity).
Output: results/miss20_classifier.json, results/miss20_rules.txt, results/figures/miss20_calibration.png
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from data import TEST_SEASON, build_panel, injury_known
from injury_types import injury_flags
from models import MISS_GAMES, eligible

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
HIST = ["age", "GP_l1", "GP_l2", "GP_l3", "gp_mean", "MIN_l1", "n_seasons"]
INJ = ["inj_knee", "inj_achilles", "inj_major", "inj_chronic", "inj_carried", "inj_games"]


def threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def scores(y, p, thr):
    pred = p >= thr
    return {"AUC": roc_auc_score(y, p), "Brier": brier_score_loss(y, p), "threshold": thr,
            "flagged_share": pred.mean(), "precision": y[pred].mean() if pred.any() else np.nan,
            "recall": pred[y].mean()}


def main():
    panel = eligible(build_panel())
    panel = panel.assign(miss=(panel["y_GP"] <= 82 - MISS_GAMES))
    inj = injury_flags(panel[injury_known(panel["target_s"])])
    train, test = inj[inj["season"] != TEST_SEASON], inj[inj["season"] == TEST_SEASON]
    y_tr, y_te = train["miss"].to_numpy(), test["miss"].to_numpy()
    groups = train["PLAYER_ID"]
    out, probs = {"base_rate_train": y_tr.mean(), "base_rate_test": y_te.mean(),
                  "n_train": len(train), "n_test": len(test)}, {}

    models = {
        "Logistic, history + injury text": (make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)), HIST + INJ),
        "Tree depth 3, history + injury text": (DecisionTreeClassifier(max_depth=3, min_samples_leaf=60, random_state=0), HIST + INJ),
        "Logistic, history only": (make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)), HIST),
    }
    for name, (m, cols) in models.items():
        cv_p = cross_val_predict(m, train[cols], y_tr, cv=GroupKFold(5), groups=groups, method="predict_proba")[:, 1]
        thr = threshold(y_tr, cv_p)
        m.fit(train[cols], y_tr)
        p = m.predict_proba(test[cols])[:, 1]
        probs[name] = p
        out[name] = {"cv_train": scores(y_tr, cv_p, thr), "test": scores(y_te, p, thr)}
    # history only, trained on every season since 1999-00 (no injury reports needed)
    allh = panel[panel["season"] != TEST_SEASON]
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(allh[HIST], allh["miss"])
    p = m.predict_proba(test[HIST])[:, 1]
    out["Logistic, history only, all seasons"] = {"test": scores(y_te, p, threshold(allh["miss"], m.predict_proba(allh[HIST])[:, 1]))}

    logit = models["Logistic, history + injury text"][0]
    coefs = pd.Series(logit[-1].coef_[0], index=HIST + INJ)
    out["odds_ratio_per_sd"] = np.exp(coefs).round(2).to_dict()
    tree = models["Tree depth 3, history + injury text"][0]
    rules = export_text(tree, feature_names=HIST + INJ, show_weights=True, decimals=1)
    (RES / "miss20_rules.txt").write_text(rules)

    p = probs["Logistic, history + injury text"]
    bands = pd.qcut(p, 5, labels=False, duplicates="drop")
    cal = pd.DataFrame({"band": bands, "pred": p, "actual": y_te}).groupby("band").agg(
        n=("pred", "size"), predicted=("pred", "mean"), actual=("actual", "mean"))
    out["calibration_test"] = cal.round(3).to_dict(orient="index")
    json.dump(out, open(RES / "miss20_classifier.json", "w"), indent=1, default=float)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, cal[["predicted", "actual"]].to_numpy().max() + 0.05] * 1, [0, cal[["predicted", "actual"]].to_numpy().max() + 0.05], "--", color="#888", lw=1)
    ax.plot(cal["predicted"], cal["actual"], "o-", color="#1f4e8c")
    for _, r in cal.iterrows():
        ax.annotate(f"n={int(r['n'])}", (r["predicted"], r["actual"]), textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.set_xlabel("Predicted chance of missing 20+ games"); ax.set_ylabel(f"Actual share, {TEST_SEASON}")
    ax.set_title(f"Calibration on {TEST_SEASON} (AUC {out['Logistic, history + injury text']['test']['AUC']:.2f})", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); (RES / "figures").mkdir(exist_ok=True); fig.savefig(RES / "figures" / "miss20_calibration.png", dpi=150)

    pd.set_option("display.width", 200)
    print(f"base rate: train {out['base_rate_train']:.1%} (n={len(train)}), test {out['base_rate_test']:.1%} (n={len(test)})")
    print(pd.DataFrame({k: v["test"] for k, v in out.items() if isinstance(v, dict) and "test" in v}).T.round(3))
    print("\nodds ratio per SD:", out["odds_ratio_per_sd"])
    print("\ncalibration (test):\n", cal.round(3))
    print("\ntree rules:\n", rules)


if __name__ == "__main__":
    main()
