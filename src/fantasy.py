"""League rules and fantasy scoring shared by every script.

League: 14 teams, 9-cat rotisserie (FG%, FT%, 3PM, PTS, REB, AST, STL, BLK, TO reversed),
10 roster spots, 8 active slots x 82-game cap, unlimited free-agent pickups.
"""
import numpy as np
import pandas as pd

TEAMS = 14
BUDGET = 200
ROSTER_SPOTS = 10
ACTIVE_SLOTS = 8
GAMES_CAP = 82
POOL = TEAMS * ROSTER_SPOTS                      # 140 players get drafted
CATS = ["FG", "FT", "3PM", "PTS", "REB", "AST", "STL", "BLK", "TO"]
COUNT = ["FGM", "FGA", "FTM", "FTA", "FG3M", "PTS", "REB", "AST", "STL", "BLK", "TOV"]


def raw_cats(df, fg_pct, ft_pct):
    """Per-game category contributions. FG/FT use impact = makes - league% * attempts."""
    return pd.DataFrame({
        "FG": df["FGM"] - fg_pct * df["FGA"],
        "FT": df["FTM"] - ft_pct * df["FTA"],
        "3PM": df["FG3M"], "PTS": df["PTS"], "REB": df["REB"], "AST": df["AST"],
        "STL": df["STL"], "BLK": df["BLK"], "TO": -df["TOV"],
    }, index=df.index)


def fit_reference(df, n_pool=POOL, iters=5):
    """Mean/sd of each category over the draftable pool (iterated so the pool is self-consistent)."""
    mask = np.ones(len(df), bool)
    for _ in range(iters):
        pool = df[mask]
        ref = {"fg": pool["FGM"].sum() / pool["FGA"].sum(), "ft": pool["FTM"].sum() / pool["FTA"].sum()}
        r = raw_cats(df, ref["fg"], ref["ft"])
        ref["mu"], ref["sd"] = r[mask].mean(), r[mask].std()
        z = ((r - ref["mu"]) / ref["sd"]).sum(axis=1)
        mask = (z.rank(ascending=False, method="first") <= n_pool).to_numpy()
    return ref


def z_scores(df, ref):
    r = raw_cats(df, ref["fg"], ref["ft"])
    z = (r - ref["mu"]) / ref["sd"]
    z["Z"] = z[CATS].sum(axis=1)
    return z


def season_value(z_pg, gp, n_pool=POOL):
    """Value over replacement for a season: (z/game - replacement z/game) * games/82.
    Missed games are streamed with replacement-level free agents, so only the gap is lost."""
    ranks = pd.Series(z_pg).rank(ascending=False, method="first")
    repl = pd.Series(z_pg)[(ranks > n_pool) & (ranks <= n_pool + 15)].mean()
    return (z_pg - repl) * gp / 82, repl


def dollars(value, money=TEAMS * BUDGET, n_pool=POOL):
    v = pd.Series(value)
    top = v.rank(ascending=False, method="first") <= n_pool
    pos = v.clip(lower=0)
    return pd.Series(np.where(top, pos / pos[top].sum() * money, 0.0), index=v.index)


def team_totals(per_game, gp, z_pg, repl_line):
    """Season category totals for one roster under the 8 x 82 games cap.
    Best players (by z/game) use games first; unused games are filled by a replacement player."""
    order = np.argsort(-z_pg)
    left = ACTIVE_SLOTS * GAMES_CAP
    tot = np.zeros(len(COUNT))
    for i in order:
        g = min(gp[i], left)
        left -= g
        tot += per_game[i] * g
    tot += repl_line * left
    d = dict(zip(COUNT, tot))
    return {"FG": d["FGM"] / d["FGA"], "FT": d["FTM"] / d["FTA"], "3PM": d["FG3M"], "PTS": d["PTS"],
            "REB": d["REB"], "AST": d["AST"], "STL": d["STL"], "BLK": d["BLK"], "TO": d["TOV"]}


def roto_points(totals: pd.DataFrame) -> pd.DataFrame:
    """Rank teams in each category: best = 14 points, worst = 1 (TO: fewest is best). Ties share points."""
    pts = pd.DataFrame(index=totals.index)
    for c in CATS:
        pts[c] = totals[c].rank(ascending=(c != "TO"), method="average")
    pts["total"] = pts[CATS].sum(axis=1)
    return pts
