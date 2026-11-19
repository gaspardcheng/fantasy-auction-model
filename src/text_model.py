"""Deep learning on unstructured injury text -> games played next season.

Text: the free-text "Reason" of every NBA official injury report a player appeared on in a season
(data/injury_reports.csv, built by fetch_injury_reports.py + parse_injury_reports.py), e.g.
"Left Knee; Surgery", "Right Achilles Tendon; Rupture", "Left Foot; Fifth metatarsal fractuce". There are
~2,700 distinct strings with typos and synonyms, so we read them with a pretrained sentence encoder
(all-MiniLM-L6-v2, a 6-layer transformer) instead of hand-written keyword rules.

Per player-season s we pool the reason embeddings, weighted by how many reports listed the player Out:
  * "all":  every injury that season
  * "late": injuries listed from 3 weeks before the regular season ends through the Finals, the ones
            still open when the offseason starts
plus simple counts (days out, late days out, distinct injuries).

The network (PyTorch) maps each pooled embedding through one shared learned projection (384 -> k, an
"injury severity" space), concatenates it with the tabular history, and predicts next season's change in
games played. Rows for target season t use text from t-1 and t-2 only, so there is no leakage.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from data import injury_known, load_injury_reports

ROOT = Path(__file__).resolve().parents[1]
EMB_CACHE = ROOT / "data" / "reason_embeddings.csv"
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
REG_SEASON_END = {"2018-19": "2019-04-10", "2019-20": "2020-08-14", "2020-21": "2021-05-16",
                  "2021-22": "2022-04-10", "2022-23": "2023-04-09", "2023-24": "2024-04-14",
                  "2024-25": "2025-04-13", "2025-26": "2026-04-12"}
STATUS_W = {"Out": 1.0, "Doubtful": 1.0, "Questionable": 0.5, "Probable": 0.1, "Available": 0.1}
POOLS = ["all", "late"]
COUNTS = ["days_out", "late_out", "n_injuries"]
TAB = ["GP_l1", "GP_l2", "GP_l3", "gp_mean", "age", "age2", "MIN_l1", "min_trend", "n_seasons", "has_lag2", "has_lag3"]


def clean_reason(r: str) -> str:
    return r.replace("Injury/Illness - ", "").strip()


def embed(texts):
    """Sentence embeddings for each distinct reason string, cached on disk."""
    texts = sorted(set(texts))
    cache = pd.read_csv(EMB_CACHE, index_col=0) if EMB_CACHE.exists() else pd.DataFrame()
    todo = [t for t in texts if t not in cache.index]
    if todo:
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer(ENCODER, device="cpu")
        new = pd.DataFrame(enc.encode(todo, batch_size=128, normalize_embeddings=True), index=todo)
        new.columns = [str(c) for c in new.columns]
        cache = pd.concat([cache, new])
        cache.to_csv(EMB_CACHE)
    return cache.loc[texts]


@lru_cache(maxsize=1)
def season_text_features() -> pd.DataFrame:
    """One row per (PLAYER_ID, season index): pooled embeddings + counts from that season's reports."""
    rep = load_injury_reports()
    rep["text"] = rep["reason"].map(clean_reason)
    rep["w"] = rep["status"].map(STATUS_W).fillna(0.1)
    rep["late"] = rep["date"] >= rep["season"].map(
        lambda s: (pd.Timestamp(REG_SEASON_END[s]) - pd.Timedelta(days=21)).date().isoformat())
    E = embed(rep["text"])
    out = []
    for (pid, s), g in rep.groupby(["PLAYER_ID", "s"]):
        row = {"PLAYER_ID": pid, "s": s,
               "days_out": g.loc[g["w"] >= 1, "date"].nunique(),
               "late_out": g.loc[(g["w"] >= 1) & g["late"], "date"].nunique(),
               "n_injuries": g["text"].nunique()}
        for pool, gg in (("all", g), ("late", g[g["late"]])):
            if len(gg):
                w = gg.groupby("text")["w"].sum()
                row[pool] = (E.loc[w.index].to_numpy() * w.to_numpy()[:, None]).sum(0) / w.sum()
            else:
                row[pool] = np.zeros(E.shape[1])
        out.append(row)
    return pd.DataFrame(out).set_index(["PLAYER_ID", "s"])


def text_inputs(df: pd.DataFrame):
    """Arrays for the rows of a panel slice: embeddings (rows, 4 pools, dim) and counts (rows, 6)."""
    T = season_text_features()
    dim = len(T["all"].iloc[0])
    embs, cnts = [], []
    for lag in (1, 2):
        idx = list(zip(df["PLAYER_ID"], df["target_s"] - lag))
        sub = T.reindex(idx)
        for pool in POOLS:
            embs.append(np.stack([v if isinstance(v, np.ndarray) else np.zeros(dim) for v in sub[pool]]))
        cnts.append(sub[COUNTS].fillna(0).to_numpy(float))
    return np.stack(embs, axis=1).astype(np.float32), np.log1p(np.hstack(cnts)).astype(np.float32)


# Zero-shot severity probes: similarity of a player's pooled injury text to reference phrases.
ANCHORS = ["Achilles tendon rupture, season-ending surgery", "Torn ACL, knee reconstruction surgery",
           "Bone fracture", "Surgery recovery", "Chronic back injury", "Muscle strain",
           "Ankle sprain", "Minor soreness", "Illness", "Rest, injury management"]


def reduce_text(E_tr, E_te, how, k):
    """(rows, pools, dim) embeddings -> (rows, pools * k) features, fitted on training rows only."""
    if how == "anchors":
        A = embed(ANCHORS).loc[ANCHORS].to_numpy(np.float32)            # (k, dim)
        return (E_tr @ A.T).reshape(len(E_tr), -1), (E_te @ A.T).reshape(len(E_te), -1)
    flat = E_tr.reshape(-1, E_tr.shape[2])
    flat = flat[np.abs(flat).sum(1) > 0]                                 # pools that have text
    mu = flat.mean(0)
    V = np.linalg.svd(flat - mu, full_matrices=False)[2][:k]             # PCA directions
    proj = lambda E: np.where(np.abs(E).sum(2, keepdims=True) > 0, (E - mu) @ V.T, 0).reshape(len(E), -1)
    return proj(E_tr), proj(E_te)


class GamesNet(nn.Module):
    def __init__(self, n_tab, dim=384, k=8, hidden=32, learned=True, n_pools=4, dropout=0.2):
        super().__init__()
        self.learned = learned
        self.proj = nn.Linear(dim, k)                          # shared across pools and lags
        n_in = n_tab + (n_pools * k if learned else 0)
        self.mlp = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, tab, emb):
        x = tab
        if self.learned:
            x = torch.cat([tab, torch.tanh(self.proj(emb)).flatten(1)], dim=1)
        return self.mlp(x).squeeze(1)


def nn_games(train, test, counts=True, text="anchors", k=8, hidden=32, dropout=0.2, wd=1e-2, epochs=400,
             seeds=5):
    """Predicted games played for `test` rows (average of `seeds` networks).
    text: None (no embeddings), "anchors" (zero-shot severity probes), "pca" (k components) or
    "learned" (a k-dim projection trained end to end with the network)."""
    Etr, Ctr = text_inputs(train)
    Ete, Cte = text_inputs(test)
    known = lambda d: np.column_stack([injury_known(d["target_s"], lag) for lag in (1, 2)])
    Xtr = np.hstack([train[TAB].to_numpy(np.float32), known(train)])   # text is missing (zeros) where not known
    Xte = np.hstack([test[TAB].to_numpy(np.float32), known(test)])
    if counts:
        Xtr, Xte = np.hstack([Xtr, Ctr]), np.hstack([Xte, Cte])
    if text in ("anchors", "pca"):
        Rtr, Rte = reduce_text(Etr, Ete, text, k)
        Xtr, Xte = np.hstack([Xtr, Rtr]), np.hstack([Xte, Rte])
    mu, sd = Xtr.mean(0), Xtr.std(0)
    keep = sd > 1e-6                                           # e.g. has_lag3 is all 0 when training on 1 season
    Xtr, Xte = ((Xtr - mu) / np.where(keep, sd, 1))[:, keep], ((Xte - mu) / np.where(keep, sd, 1))[:, keep]
    y = ((train["y_GP"] - train["GP_l1"]) / 82).to_numpy(np.float32)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    preds = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        net = GamesNet(Xtr.shape[1], dim=Etr.shape[2], k=k, hidden=hidden, learned=text == "learned",
                       dropout=dropout)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=wd)
        for _ in range(epochs):                                # full batch: the table is small
            net.train()
            opt.zero_grad()
            loss = nn.functional.huber_loss(net(t(Xtr), t(Etr)), t(y), delta=0.1)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            preds.append(net(t(Xte), t(Ete)).numpy())
    gp = np.mean(preds, axis=0) * 82 + test["GP_l1"].to_numpy()
    return pd.Series(np.clip(gp, 0, 82), index=test.index)


def with_text_games(stats_fn, blend=None, **kw):
    """Candidate model: per-game stats from `stats_fn`, games played from the injury-text network.
    blend=w averages the network with stats_fn's own games forecast (weight w on the network)."""
    def f(train, test):
        p = stats_fn(train, test)
        g = nn_games(train, test, **kw)
        p["GP"] = g if blend is None else blend * g + (1 - blend) * p["GP"]
        return p
    return f
