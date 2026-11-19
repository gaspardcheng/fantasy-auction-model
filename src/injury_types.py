"""Injury types from the official injury-report text: how long each keeps players out, and whether it
carries into the next season's performance.

1. Classify every free-text reason ("Left Knee; MCL sprain", "Right Achilles Tendon; Rupture") into a body
   region and an injury type. Keyword rules label the clean strings. For the rest (typos, rare wording),
   a classifier on the transformer embeddings used in text_model.py, trained on the rule-labelled strings,
   assigns the label; its cross-validated accuracy on the rule-labelled strings is reported.
2. Build injury episodes: consecutive reports listing a player Out (or Doubtful) for the same body region,
   broken by gaps of more than 14 days. Report days approximate games missed, since a team's players
   are listed only on its game days.
3. Absence by type: games missed per episode.
4. Next season by type: each player-season's main injury (the episode with most games missed, >= 5) versus
   what happened in season t: games played, minutes and per-game fantasy value, compared with players
   of the same age who had no significant injury.

Outputs: results/injury_reasons.csv, injury_episodes.csv, injury_absence_by_type.csv,
injury_next_season.csv, figures/injury_types.png
"""
import re
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import fantasy as F
from data import TEST_SEASON, build_panel, injury_known, load_injury_reports
from models import eligible
from text_model import REG_SEASON_END, clean_reason, embed

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
GAP_DAYS = 14
MAIN_MIN_GAMES = 5

# body region -> keywords (first match wins, so specific regions come before general ones)
REGIONS = {
    "Achilles": ["achilles", "achillies"],
    "Knee": ["knee", "acl", "mcl", "pcl", "lcl", "meniscus", "patella"],
    "Ankle": ["ankle"],
    "Foot / toe": ["foot", "toe", "heel", "plantar", "midfoot", "metatarsal", "navicular"],
    "Lower leg": ["calf", "tibia", "fibula", "shin", "lower leg", "leg"],
    "Hamstring / thigh": ["hamstring", "quad", "thigh"],
    "Hip / groin": ["hip", "groin", "adductor", "pelvi", "glute"],
    "Back / core": ["back", "lumbar", "spine", "oblique", "rib", "costal", "abdom", "append", "core", "neck",
                    "cervical", "chest", "sternum"],
    "Shoulder / arm": ["shoulder", "elbow", "arm", "bicep", "tricep", "clavicle", "ac joint", "ac sprain", "forearm",
                       "thoracic outlet", "tos"],
    "Hand / wrist": ["hand", "wrist", "thumb", "finger", "digit", "pinky"],
    "Head / face": ["concussion", "head", "nasal", "nose", "face", "facial", "eye", "jaw", "orbital", "mask",
                    "tooth", "teeth", "lip", "ear"],
    "Illness": ["illness", "covid", "flu", "virus", "respiratory", "health and safety", "throat", "lung",
                "gastro", "migraine", "dehydration"],
}
# injury type, most severe first: an episode takes the most severe type seen while it lasts
TYPES = {
    "Surgery": ["surgery", "repair", "reconstruction", "-scope", "arthroscop", "appendectomy", "debridement",
                "procedure"],
    "Tear / rupture": ["tear", "torn", "rupture", "dislocat", "sublux"],
    "Fracture": ["fracture", "fractuce", "break", "broken", "stress reaction"],
    "Sprain": ["sprain"],
    "Strain": ["strain"],
    "Contusion": ["contusion", "bruise"],
    "Soreness / tendinitis": ["sore", "tight", "inflam", "tendin", "tendon", "spasm", "pain", "discomfort",
                              "impingement", "irritat", "effusion", "bursitis", "stiff", "swelling"],
    "Management": ["management", "mgmt", "recovery", "maintenance", "reconditioning", "rehab", "rest"],
    "Illness": ["illness", "covid", "flu", "virus", "health and safety"],
}
SEVERITY = list(TYPES)
# reasons filed under Injury/Illness that are not injuries (whole string); dropped before classifying
NON_INJURY = re.compile(r"^(n/a; )?(medical assessment|personal( reasons)?|family reasons|not with team|rest)$", re.I)


def _match(t, table):
    for label, keys in table.items():
        if any(re.search(r"\b" + re.escape(k) if not k.startswith("-") else re.escape(k[1:]), t) for k in keys):
            return label
    return None


@lru_cache(maxsize=None)
def _vocab(names):
    table = {"region": REGIONS, "type": TYPES}[names]
    return sorted({k for keys in table.values() for k in keys if k.isalpha() and len(k) >= 4})


FUZZY_HITS = {}
KNOWN_WORDS = set()          # words that are not typos: in 3+ distinct reason strings, or a keyword


def set_known_words(texts):
    counts = pd.Series([w for t in set(texts) for w in set(re.findall(r"[a-z]+", t.lower()))]).value_counts()
    KNOWN_WORDS.clear()
    KNOWN_WORDS.update(counts[counts >= 3].index)
    KNOWN_WORDS.update(_vocab("region") + _vocab("type"))


def rule(text, table):
    """Keyword rules; if no keyword matches, retry with typos fixed by fuzzy matching: a rare word (in fewer
    than 3 distinct strings, not itself a keyword) of 4+ letters within 80% similarity of a keyword is
    replaced by it ("groun" -> "groin", "ilness" -> "illness", "sprian" -> "sprain")."""
    t = text.lower()
    label = _match(t, table)
    if label is None:
        vocab = _vocab("region" if table is REGIONS else "type")
        words = re.findall(r"[a-z]+", t)
        fixed = [w if len(w) < 4 or w in KNOWN_WORDS else (get_close_matches(w, vocab, n=1, cutoff=0.8) or [w])[0]
                 for w in words]
        if fixed != words:
            label = _match(" ".join(fixed), table)
            if label is not None:
                FUZZY_HITS[text] = " ".join(fixed)
    return label


def encoder_classifier(texts, labels):
    """Logistic regression on sentence embeddings, trained on the rule-labelled strings.
    Returns predictions for every text and the 5-fold cross-validated accuracy on the labelled ones."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    E = embed(texts).loc[texts].to_numpy()
    known = labels.notna().to_numpy()
    clf = LogisticRegression(C=10, max_iter=3000)
    acc = cross_val_score(clf, E[known], labels[known], cv=5).mean()
    clf.fit(E[known], labels[known])
    return pd.Series(clf.predict(E), index=texts), acc


def classify_reasons(rep):
    texts = sorted(set(rep["text"]))
    set_known_words(texts)
    out = pd.DataFrame(index=pd.Index(texts, name="text"))
    acc = {}
    for col, table in (("region", REGIONS), ("type", TYPES)):
        out[f"{col}_rule"] = [rule(t, table) for t in texts]
        out[f"{col}_encoder"], acc[col] = encoder_classifier(texts, out[f"{col}_rule"])
        out[col] = out[f"{col}_rule"].fillna(out[f"{col}_encoder"])
    nature = out.index.to_series().str.split(";", n=1).str[1].fillna("").str.strip()
    out.loc[out["type_rule"].isna() & nature.isin(["", "Injury", "Out"]).to_numpy(), "type"] = "Unspecified"
    out.loc[out["region"] == "Illness", "type"] = "Illness"
    out["source"] = np.where(out["region_rule"].notna() & out["type_rule"].notna(), "rules", "encoder")
    return out, acc


def episodes(rep):
    """Runs of Out/Doubtful reports for the same player, season and body region."""
    out = rep[rep["status"].isin(["Out", "Doubtful"])].copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["PLAYER_ID", "s", "region", "date"])
    new = (out.groupby(["PLAYER_ID", "s", "region"])["date"].diff().dt.days.fillna(999) > GAP_DAYS)
    out["ep"] = new.cumsum()
    sev = {t: i for i, t in enumerate(SEVERITY)}
    ep = out.groupby("ep").agg(PLAYER_ID=("PLAYER_ID", "first"), player=("player", "first"), s=("s", "first"),
                               season=("season", "first"), region=("region", "first"),
                               start=("date", "min"), end=("date", "max"), games=("date", "nunique"),
                               type=("type", lambda x: min(x, key=lambda t: sev.get(t, 99))),
                               example=("text", lambda x: x.value_counts().index[0]))
    ep["into_offseason"] = ep["end"] >= ep["season"].map(lambda s: pd.Timestamp(REG_SEASON_END[s]) - pd.Timedelta(days=7))
    return ep.reset_index(drop=True)


def labelled_reports():
    """Injury rows (rest and COVID protocols removed) with body region and injury type."""
    rep = load_injury_reports()
    rep = rep[~rep["reason"].str.startswith(("Rest", "Health and Safety"))].copy()
    rep["text"] = rep["reason"].map(clean_reason)
    rep = rep[~rep["text"].str.match(NON_INJURY)]
    lab, acc = classify_reasons(rep)
    return rep.merge(lab[["region", "type"]], left_on="text", right_index=True), lab, acc


@lru_cache(maxsize=1)
def main_injuries() -> pd.DataFrame:
    """Each player-season's main injury: the episode with the most games missed (>= MAIN_MIN_GAMES)."""
    ep = episodes(labelled_reports()[0])
    return (ep[ep["games"] >= MAIN_MIN_GAMES].sort_values("games")
            .groupby(["PLAYER_ID", "s"]).tail(1).set_index(["PLAYER_ID", "s"]))


def injury_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Season t-1 injury features for panel rows: the injury types that predicted fewer games next season."""
    m = main_injuries().reindex(list(zip(df["PLAYER_ID"], df["target_s"] - 1)))
    out = df.copy()
    out["inj_known"] = injury_known(df["target_s"]).astype(int)   # flags below are 0 where unknown
    out["inj_knee"] = (m["region"] == "Knee").to_numpy(int)
    out["inj_achilles"] = (m["region"] == "Achilles").to_numpy(int)
    out["inj_major"] = m["type"].isin(["Tear / rupture", "Surgery"]).to_numpy(int)
    out["inj_chronic"] = (m["type"] == "Soreness / tendinitis").to_numpy(int)
    out["inj_carried"] = m["into_offseason"].fillna(False).astype(bool).to_numpy(int)
    out["inj_games"] = m["games"].fillna(0).to_numpy()
    return out


def season_z(h_rows):
    """Per-game fantasy value (sum of 9 category z-scores) on each season's own scale."""
    df = h_rows.rename(columns={})
    ref = F.fit_reference(df[df["GP"] >= 15])
    return F.z_scores(df, ref)["Z"]


def next_season_effects(panel):
    """Main injury of season t-1 vs outcomes in season t, relative to same-age players without injury.
    Uses only seasons with a full year of injury reports, and never the hold-out season, so the injury
    flags used by the models (knee, Achilles, major, chronic) are chosen on training seasons only."""
    rows = eligible(panel)
    rows = rows[injury_known(rows["target_s"]) & (rows["season"] != TEST_SEASON)].copy()
    rows["s_prev"] = rows["target_s"] - 1
    main = main_injuries()
    key = list(zip(rows["PLAYER_ID"], rows["s_prev"]))
    m = main.reindex(key)
    rows["region"] = m["region"].fillna("No significant injury").to_numpy()
    rows["type"] = m["type"].fillna("No significant injury").to_numpy()
    rows["games_missed_prev"] = m["games"].fillna(0).to_numpy()
    rows["into_offseason"] = m["into_offseason"].fillna(False).astype(bool).to_numpy()

    # per-game value in t-1 and t, each on its own season's scale
    stats = ["FGM", "FGA", "FTM", "FTA", "FG3M", "PTS", "REB", "AST", "STL", "BLK", "TOV"]
    prev = rows[[f"{c}_l1" for c in stats]].set_axis(stats, axis=1).assign(GP=rows["GP_l1"])
    cur = rows[[f"y_{c}" for c in stats]].set_axis(stats, axis=1).assign(GP=rows["y_GP"])
    rows["z_prev"] = np.nan
    rows["z_next"] = np.nan
    for s, idx in rows.groupby("target_s").groups.items():
        rows.loc[idx, "z_prev"] = season_z(prev.loc[idx])
        played = cur.loc[idx][cur.loc[idx, "GP"] >= 15]
        rows.loc[played.index, "z_next"] = season_z(played)
    rows["dz"] = rows["z_next"] - rows["z_prev"]
    rows["dmin"] = rows["y_MIN"].where(rows["y_GP"] >= 15) - rows["MIN_l1"]

    # age-expected change from players with no significant injury (linear in age)
    ctrl = rows[(rows["type"] == "No significant injury")]
    for col in ("dz", "dmin", "y_GP"):
        c = ctrl.dropna(subset=[col])
        b = np.polyfit(c["age"], c[col], 1)
        rows[f"excess_{col}"] = rows[col] - np.polyval(b, rows["age"])
    return rows


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x.dropna())
    if len(x) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def summarize(rows, by):
    out = []
    for g, d in rows.groupby(by):
        lo, hi = boot_ci(d["excess_dz"])
        glo, ghi = boot_ci(d["excess_y_GP"])
        out.append({by: g, "n": len(d), "games_missed_prev": d["games_missed_prev"].mean(),
                    "next_GP": d["y_GP"].mean(), "excess_GP": d["excess_y_GP"].mean(),
                    "excess_GP_lo": glo, "excess_GP_hi": ghi,
                    "excess_min": d["excess_dmin"].mean(), "excess_z": d["excess_dz"].mean(),
                    "excess_z_lo": lo, "excess_z_hi": hi, "n_played": d["dz"].notna().sum()})
    return pd.DataFrame(out).sort_values("excess_z")


def plot(by_region, by_type):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = ((by_type, "type", "excess_GP", "Games next season, by injury type", "games vs same-age uninjured"),
              (by_region, "region", "excess_GP", "Games next season, by body region", "games vs same-age uninjured"),
              (by_type, "type", "excess_z", "Value per game next season, by type", "z per game vs same-age uninjured"))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    for ax, (d, col, m, title, xlabel) in zip(axes, panels):
        d = d[d["n"] >= 8].sort_values(m)
        y = np.arange(len(d))
        base = (d[col] == "No significant injury").to_numpy()
        ax.errorbar(d[m], y, xerr=[d[m] - d[f"{m}_lo"], d[f"{m}_hi"] - d[m]], fmt="o", color="#1f4e8c",
                    ecolor="#9fb3cf", capsize=3)
        ax.scatter(d[m][base], y[base], color="#c4511a", zorder=3)
        ax.axvline(0, color="#888", lw=1, ls="--")
        ax.set_yticks(y, [f"{r} (n={n})" for r, n in zip(d[col], d["n"])], fontsize=9)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Main injury last season vs next season (95% bootstrap CI). Injury type predicts games, not per-game play.",
                 fontsize=12)
    fig.tight_layout()
    (RES / "figures").mkdir(exist_ok=True)
    fig.savefig(RES / "figures" / "injury_types.png", dpi=150)


def main():
    rep, lab, acc = labelled_reports()
    lab.to_csv(RES / "injury_reasons.csv")
    print(f"{len(lab)} distinct reasons; rules label {lab['source'].eq('rules').mean():.0%} fully, the encoder "
          f"classifier fills the rest (5-fold accuracy vs rules: region {acc['region']:.0%}, type {acc['type']:.0%})")

    ep = episodes(rep)
    ep.to_csv(RES / "injury_episodes.csv", index=False)
    absence = (ep.groupby(["region", "type"])["games"].agg(episodes="size", median="median", mean="mean",
                                                          p90=lambda x: x.quantile(0.9))
               .reset_index().sort_values("median", ascending=False))
    absence.to_csv(RES / "injury_absence_by_type.csv", index=False)
    print(f"\n{len(ep)} injury episodes. Games missed per episode by region:")
    print(ep.groupby("region")["games"].agg(episodes="size", median="median", mean="mean",
                                            p90=lambda x: x.quantile(0.9)).sort_values("median", ascending=False).round(1))
    print("\nby type:")
    print(ep.groupby("type")["games"].agg(episodes="size", median="median", mean="mean",
                                          p90=lambda x: x.quantile(0.9)).sort_values("median", ascending=False).round(1))

    rows = next_season_effects(build_panel())
    by_region, by_type = summarize(rows, "region"), summarize(rows, "type")
    off = summarize(rows.assign(carried=np.where(rows["type"] == "No significant injury", "No significant injury",
                                                 np.where(rows["into_offseason"], "Still out at season end",
                                                          "Healed before season end"))), "carried")
    pd.concat([by_region.assign(group="region").rename(columns={"region": "label"}),
               by_type.assign(group="type").rename(columns={"type": "label"}),
               off.assign(group="carried").rename(columns={"carried": "label"})]).to_csv(
        RES / "injury_next_season.csv", index=False)
    pd.set_option("display.width", 200)
    cols = ["n", "games_missed_prev", "next_GP", "excess_GP", "excess_min", "excess_z", "excess_z_lo", "excess_z_hi"]
    for name, d in (("body region", by_region), ("injury type", by_type), ("timing", off)):
        print(f"\nNext season by main injury ({name}) of the previous season:")
        print(d.set_index(d.columns[0])[cols].round(2))
    plot(by_region, by_type)


if __name__ == "__main__":
    main()
