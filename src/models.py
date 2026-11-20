"""Benchmarks and ML models. Each returns predicted per-game stats + games played for the rows of `test`."""
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LogisticRegression, MultiTaskElasticNetCV, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
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


class ENet:
    """Elastic net with the penalty (alpha and L1 share) chosen by 5-fold cross-validation inside the
    training rows. Multi-output: one sparsity pattern shared by all 11 stats, so the kept features read as
    one pricing rule."""
    L1 = [0.2, 0.5, 0.8, 1.0]                                 # 1.0 = LASSO

    def fit(self, X, Y):
        Y = np.asarray(Y)
        cv = (MultiTaskElasticNetCV if Y.ndim > 1 else ElasticNetCV)(l1_ratio=self.L1, alphas=30, cv=5,
                                                                    max_iter=5000, n_jobs=-1)
        self.m = make_pipeline(StandardScaler(), cv).fit(X, Y)
        return self

    def predict(self, X):
        return self.m.predict(X)


def enet(tr, te):
    return _fit_predict(ENet, tr, te)


def gbm_es(lr, leaves, min_leaf):
    """Gradient boosting with early stopping: the number of trees is chosen on a 10% inner validation split."""
    return lambda tr, te: _fit_predict(
        lambda: PerTarget(learning_rate=lr, max_iter=2000, max_leaf_nodes=leaves, min_samples_leaf=min_leaf,
                          l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                          n_iter_no_change=30), tr, te)


class MLP:
    """Multi-output neural network (all 11 stats jointly, or games), early stopping on a 10% inner split.
    Inputs and targets standardized; average of 3 seeds."""
    def __init__(self, hidden=(64, 32)):
        self.hidden = hidden

    def fit(self, X, Y):
        self.ms = [TransformedTargetRegressor(
            make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=self.hidden, alpha=1e-2, learning_rate_init=1e-3,
                                                         early_stopping=True, n_iter_no_change=20,
                                                         max_iter=500, random_state=seed)),
            transformer=StandardScaler()).fit(X, np.asarray(Y)) for seed in range(3)]
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.ms], axis=0)


def mlp(hidden):
    return lambda tr, te: _fit_predict(lambda: MLP(hidden), tr, te)


COMP_FEATURES = ["age", "MIN_l1", "GP_l1", "gp_mean", "min_trend", "USG_PCT_l1", "TS_PCT_l1", "AST_PCT_l1",
                 "REB_PCT_l1", "height"] + [f"{t}_l1" for t in TARGETS]


class Comparables:
    """Nearest-neighbour comparables: the k most similar past player-seasons (age, role, last season's
    line) and what happened to them next, distance-weighted."""
    def __init__(self, k):
        self.k = k

    def fit(self, X, Y):
        self.m = make_pipeline(StandardScaler(), KNeighborsRegressor(self.k, weights="distance"))
        self.m.fit(X[COMP_FEATURES], Y)
        return self

    def predict(self, X):
        return self.m.predict(X[COMP_FEATURES])


def comparables(k):
    return lambda tr, te: _fit_predict(lambda: Comparables(k), tr, te)


MISS_GAMES = 20


def miss20_games(stats_fn):
    """Games played from a "misses 20+ games" classifier (logistic regression on history, age and the
    injury-report flags): expected games = P(miss) x mean games of past missers + (1 - P) x mean games
    of past non-missers, each mean fitted by Ridge within its group."""
    def f(tr, te):
        from injury_types import injury_flags
        p = stats_fn(tr, te)
        tr, te = injury_flags(tr), injury_flags(te)
        X_cols = feature_cols(tr)
        miss = (tr["y_GP"] <= 82 - MISS_GAMES).to_numpy()
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(tr[X_cols], miss)
        prob = clf.predict_proba(te[X_cols])[:, 1]
        part = [make_pipeline(StandardScaler(), Ridge(100)).fit(tr.loc[g, X_cols], tr.loc[g, "y_GP"])
                for g in (miss, ~miss)]
        gp = prob * part[0].predict(te[X_cols]) + (1 - prob) * part[1].predict(te[X_cols])
        p["GP"] = np.clip(gp, 0, 82)
        return p
    return f


CANDIDATES = {
    "Naive (last season)": naive,
    "Marcel (weighted avg)": marcel,
    **{f"Ridge a={a}": ridge(a) for a in (10, 100, 1000)},
    "ElasticNet (CV penalty)": enet,
    **{f"GBM lr={lr} leaves={lv} minleaf={ml}": gbm_es(lr, lv, ml)
       for lr, lv, ml in ((0.03, 7, 50), (0.03, 15, 50), (0.1, 7, 50), (0.03, 7, 150))},
    **{f"RandomForest leaf={m}": forest(m) for m in (5, 15, 30)},
    **{f"MLP {'x'.join(map(str, h))}": mlp(h) for h in ((64, 32), (128,))},
    **{f"Comparables k={k}": comparables(k) for k in (25, 100)},
    "Ensemble Ridge+RF": ensemble(ridge(100), forest(15)),
    # games played from the injury reports (per-game stats unchanged, so these tie with their stats model on
    # per-game error and are separated by games-played error)
    "Ensemble Ridge+RF + injury-text NN games": games_from(ensemble(ridge(100), forest(15)), injury_text_nn),
    "Ensemble Ridge+RF + injury-type games": games_from(ensemble(ridge(100), forest(15)),
                                                         injury_types(ensemble(ridge(100), forest(15)))),
    "Ridge a=10 + miss-20 classifier games": miss20_games(ridge(10)),
    "Ridge a=10 + GBM games": games_from(ridge(10), gbm_es(0.03, 7, 50)),
}
BASELINES = ["Naive (last season)", "Marcel (weighted avg)"]
