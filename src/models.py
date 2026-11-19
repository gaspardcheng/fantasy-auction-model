"""Benchmarks and ML models. Each returns predicted per-game stats + games played for the rows of `test`."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import TARGETS, feature_cols

OUT = TARGETS + ["GP"]


def eligible(df):
    """Draft-relevant players: real role last season."""
    return df[(df["GP_l1"] >= 20) & (df["MIN_l1"] >= 12)]


def window(panel, season, n_seasons):
    """Training rows for target `season`: the `n_seasons` target seasons just before it (None = all)."""
    s = panel.loc[panel["season"] == season, "target_s"]
    t = s.iloc[0] if len(s) else panel["target_s"].max() + 1
    lo = -np.inf if n_seasons is None else t - n_seasons
    return panel[(panel["target_s"] < t) & (panel["target_s"] >= lo)]


# ------------------------------------------------------------------ benchmarks
def naive(train, test):
    """Business rule 1: 'last season repeats' (what most casual GMs do)."""
    p = pd.DataFrame({t: test[f"{t}_l1"] for t in TARGETS}, index=test.index)
    p["GP"] = test["GP_l1"]
    return p


def marcel(train, test):
    """Business rule 2: Marcel-style projection used by public fantasy sites.
    5/4/3 season weights (x games played), regressed to the league mean with 100 games of ballast,
    small age curve (+0.6%/yr under 28, -0.3%/yr over 28). Games = weighted GP regressed to 60."""
    w = {1: 5, 2: 4, 3: 3}
    gw = {l: w[l] * test[f"GP_l{l}"] * test[f"has_lag{l}"].clip(lower=0) for l in w}
    gw[1] = w[1] * test["GP_l1"]
    den = sum(gw.values())
    mean = eligible(train)[[f"{t}_l1" for t in TARGETS]].mean()
    age = np.where(test["age"] < 28, 1 + 0.006 * (28 - test["age"]), 1 - 0.003 * (test["age"] - 28))
    p = pd.DataFrame(index=test.index)
    for t in TARGETS:
        num = sum(gw[l] * test[f"{t}_l{l}"] for l in w)
        p[t] = (num + 100 * mean[f"{t}_l1"]) / (den + 100) * age
    gp = sum(w[l] * test[f"GP_l{l}"] for l in w) / 12
    p["GP"] = 0.75 * gp + 0.25 * 60
    return p


# ------------------------------------------------------------------ ML models
def _fit_predict(make, train, test):
    """Models learn the CHANGE from last season (y_t - y_{t-1}); predictions add last season back.
    This keeps every model anchored to the strong 'last season' baseline and lets it learn
    systematic adjustments (age curve, minutes trend, regression of outlier seasons)."""
    X_cols = feature_cols(train)
    lag_s = [f"{t}_l1" for t in TARGETS]
    tr_stats = train[train["y_GP"] >= 15]
    p = pd.DataFrame(index=test.index)
    m = make()
    m.fit(tr_stats[X_cols], tr_stats[[f"y_{t}" for t in TARGETS]].to_numpy() - tr_stats[lag_s].to_numpy())
    p[TARGETS] = np.clip(m.predict(test[X_cols]) + test[lag_s].to_numpy(), 0, None)
    g = make()
    g.fit(train[X_cols], (train["y_GP"] - train["GP_l1"]).to_numpy())
    p["GP"] = np.clip(g.predict(test[X_cols]).reshape(-1) + test["GP_l1"].to_numpy(), 0, 82)
    return p


class PerTarget:
    """Fit one HistGradientBoosting model per target column."""
    def __init__(self, **kw):
        self.kw = kw

    def fit(self, X, Y):
        Y = pd.DataFrame(Y)
        self.models = [HistGradientBoostingRegressor(random_state=0, **self.kw).fit(X, Y.iloc[:, j])
                       for j in range(Y.shape[1])]
        return self

    def predict(self, X):
        out = np.column_stack([m.predict(X) for m in self.models])
        return out if out.shape[1] > 1 else out.ravel()


def ridge(alpha):
    return lambda tr, te: _fit_predict(
        lambda: make_pipeline(StandardScaler(), Ridge(alpha=alpha)), tr, te)


def gbm(lr, leaves, min_leaf):
    return lambda tr, te: _fit_predict(
        lambda: PerTarget(learning_rate=lr, max_iter=300, max_leaf_nodes=leaves,
                          min_samples_leaf=min_leaf, l2_regularization=1.0), tr, te)


def forest(min_leaf):
    return lambda tr, te: _fit_predict(
        lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=min_leaf, max_features=0.5,
                                      n_jobs=-1, random_state=0), tr, te)


def games_from(stats_fn, games_fn):
    """Per-game stats from `stats_fn`, games played from `games_fn` (both fitted on the same training rows)."""
    def f(tr, te):
        p = stats_fn(tr, te)
        g = games_fn(tr, te)
        p["GP"] = g["GP"] if isinstance(g, pd.DataFrame) else g
        return p
    return f


def injury_types(fn):
    """Same model, with last season's injury type (from the injury-report text, see injury_types.py) as
    extra inputs. Used for games played only: injury type predicted availability, not per-game output."""
    def f(tr, te):
        from injury_types import injury_flags
        return fn(injury_flags(tr), injury_flags(te))
    return f


def injury_text_nn(tr, te):
    """Games played from the injury-text network (text_model.py): pooled transformer embeddings."""
    from text_model import nn_games
    return nn_games(tr, te, text="anchors", epochs=150)


def ensemble(*fns):
    """Average of several models' predictions."""
    def f(tr, te):
        ps = [fn(tr, te) for fn in fns]
        return sum(ps) / len(ps)
    return f


CANDIDATES = {
    "Naive (last season)": naive,
    "Marcel (weighted avg)": marcel,
    **{f"Ridge a={a}": ridge(a) for a in (10, 100, 1000)},
    **{f"GBM lr={lr} leaves={lv}": gbm(lr, lv, 20) for lr in (0.03, 0.08) for lv in (7, 15)},
    **{f"RandomForest leaf={m}": forest(m) for m in (5, 15, 30)},
    "Ensemble Ridge+RF": ensemble(ridge(100), forest(15)),
    # unstructured data: games played from the injury-report text (per-game stats unchanged, so these tie
    # with "Ensemble Ridge+RF" on per-game error and are separated by games-played error)
    "Ensemble Ridge+RF + injury-text NN games": games_from(ensemble(ridge(100), forest(15)), injury_text_nn),
    "Ensemble Ridge+RF + injury-type games": games_from(ensemble(ridge(100), forest(15)),
                                                         injury_types(ensemble(ridge(100), forest(15)))),
}
BASELINES = ["Naive (last season)", "Marcel (weighted avg)"]
