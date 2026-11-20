"""What drives the forecast, in readable terms (training seasons only, scored on the 2025-26 hold-out).

  * Elastic net (penalty chosen by cross-validation): how many inputs it keeps, and the largest
    standardized coefficients for the change in points per game.
  * Games played from gradient boosting (the selected games model): permutation importance on 2025-26,
    i.e. how much the games error rises when one input is shuffled.
Output: results/explain.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from data import TARGETS, TEST_SEASON, build_panel, feature_cols
from models import ENet, eligible

RES = Path(__file__).resolve().parents[1] / "results"


def main():
    panel = eligible(build_panel())
    train, test = panel[panel["season"] < TEST_SEASON], panel[panel["season"] == TEST_SEASON]
    X = feature_cols(panel)
    tr = train[train["y_GP"] >= 15]
    Y = tr[[f"y_{t}" for t in TARGETS]].to_numpy() - tr[[f"{t}_l1" for t in TARGETS]].to_numpy()
    en = ENet().fit(tr[X], Y).m
    cv = en[-1]
    coef = pd.DataFrame(cv.coef_.T, index=X, columns=TARGETS)
    kept = coef.index[(coef.abs() > 1e-8).any(axis=1)]
    pts = coef["PTS"][coef["PTS"].abs() > 1e-8].sort_values(key=abs, ascending=False)

    gbm = HistGradientBoostingRegressor(learning_rate=0.03, max_iter=2000, max_leaf_nodes=7, min_samples_leaf=50,
                                        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                        n_iter_no_change=30, random_state=0)
    gbm.fit(train[X], train["y_GP"] - train["GP_l1"])
    imp = permutation_importance(gbm, test[X], test["y_GP"] - test["GP_l1"], scoring="neg_mean_absolute_error",
                                 n_repeats=20, random_state=0)
    gp_imp = pd.Series(imp.importances_mean, index=X).sort_values(ascending=False)

    out = {"enet_alpha": float(cv.alpha_), "enet_l1_ratio": float(cv.l1_ratio_), "n_inputs": len(X),
           "enet_kept": len(kept), "enet_dropped": sorted(set(X) - set(kept)),
           "enet_pts_top": pts.head(10).round(3).to_dict(),
           "gbm_trees": int(gbm.n_iter_),
           "gbm_games_importance_top": gp_imp.head(10).round(3).to_dict()}
    json.dump(out, open(RES / "explain.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
