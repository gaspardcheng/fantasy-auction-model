"""
Head-to-head (H2H) category league: auction values + build analysis for the H2H board.

League: 12 teams, $200 auction, 12 roster spots (PG, SG, G, SF, PF, F, C, C, UTIL, UTIL, BN, BN),
9 categories scored each week (Mon-Sun) as W / L / T per category. Weeks 1-18 regular season,
top 8 by (category wins - losses) make the playoffs in weeks 19-21.

Inputs:
  results/projections_2026_27.csv (or data/Data 2.xlsx)  per-game projections + positions
  data/Data 1.csv                                         past games played (health)
  data/Data 3.xlsx                                        NBA games per team per week

Output:
  docs/players_h2h.js   window.H2H_DATA = {...}, loaded by docs/h2h.html
  results/h2h_builds.csv  every 5-, 6-, 7- and 9-category build, scored against a league-average team

Weekly model
  A player contributes (team games that week) x (availability) x (per-game line). Availability =
  expected games / 82. A team's weekly total in a category is approximately normal:
    mean = sum over players of games * avail * per-game mean
    var  = sum over players of games * (avail * DISP * per-game mean + avail * (1 - avail) * per-game mean^2)
  (game-to-game noise + the chance he sits). FG% / FT% use binomial shooting noise on the week's attempts.
  P(win category) = Phi((my mean - opp mean) / sqrt(my var + opp var)); TO reversed.
  Expected weekly W-L = sum over categories of (2P - 1).
"""
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from build_values import BLENDED, DATA, load_history, load_projections, norm_name, per_game_z

# ---------------------------------------------------------------- league settings
TEAMS = 12
BUDGET = 200
SLOTS = ["PG", "SG", "G", "SF", "PF", "F", "C", "C", "UTIL", "UTIL", "BN", "BN"]
ROSTER_SPOTS = len(SLOTS)
POOL = TEAMS * ROSTER_SPOTS                 # 144 players get drafted
REG_WEEKS = 18
PLAYOFF_WEEKS = [19, 20, 21]
PLAYOFF_TEAMS = 8
CATS = ["FG", "FT", "3PM", "PTS", "REB", "AST", "STL", "BLK", "TO"]
STATK = ["FGM", "FGA", "FTM", "FTA", "3PM", "PTS", "REB", "AST", "STL", "BLK", "TO"]
# per-game variance / mean of each counting stat (game-to-game noise; Poisson = 1, points are lumpier)
DISP = {"3PM": 1.0, "PTS": 2.5, "REB": 1.2, "AST": 1.1, "STL": 1.0, "BLK": 1.2, "TO": 1.0}
SHOOT_K = 1.2                               # streakiness multiplier on binomial FG% / FT% noise
HEALTH_WEIGHT_HISTORY = 0.4

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "players_h2h.js"
BUILDS = ROOT / "results" / "h2h_builds.csv"

ABBR = {"ATL": "Atlanta Hawks", "BKN": "Brooklyn Nets", "BOS": "Boston Celtics", "CHA": "Charlotte Hornets",
        "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers", "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets",
        "DET": "Detroit Pistons", "GS": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
        "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies", "MIA": "Miami Heat",
        "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves", "NO": "New Orleans Pelicans",
        "NY": "New York Knicks", "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic",
        "PHI": "Philadelphia 76ers", "PHO": "Phoenix Suns", "POR": "Portland Trail Blazers",
        "SA": "San Antonio Spurs", "SAC": "Sacramento Kings", "TOR": "Toronto Raptors", "UTA": "Utah Jazz",
        "WAS": "Washington Wizards"}


def load_schedule() -> pd.DataFrame:
    s = pd.read_excel(DATA / "Data 3.xlsx").set_index("TEAM")
    weeks = [f"W{w}" for w in range(1, PLAYOFF_WEEKS[-1] + 1)]
    full = s[weeks].astype(int)
    out = pd.DataFrame({a: full.loc[n].to_numpy() for a, n in ABBR.items()}).T
    out.columns = weeks
    return out


def load_players() -> pd.DataFrame:
    use_ml = BLENDED.exists()
    df = pd.read_csv(BLENDED) if use_ml else load_projections()
    df["key"] = df["name"].map(norm_name)
    df = df.merge(load_history(), on="key", how="left")
    proj_rate = df["GP"] / 82
    w = np.where((df["hist_seasons"].fillna(0) >= 2) & (not use_ml), HEALTH_WEIGHT_HISTORY, 0.0)
    df["exp_gp"] = (82 * ((1 - w) * proj_rate + w * df["hist_gp_rate"].fillna(proj_rate))).round(1)
    df["avail"] = (df["exp_gp"] / 82).clip(upper=1.0)
    return df


# ---------------------------------------------------------------- weekly team model (numpy)
def player_terms(df):
    """Per scheduled game: expected stat line (mu) and its variance (var), plus shooting variance."""
    m = df[STATK].to_numpy(float)
    a = df["avail"].to_numpy(float)[:, None]
    disp = np.array([DISP.get(k, 0.0) for k in STATK])
    mu = a * m
    var = a * disp * m + a * (1 - a) * m ** 2
    fgp = np.divide(m[:, 0], m[:, 1], out=np.zeros(len(m)), where=m[:, 1] > 0)
    ftp = np.divide(m[:, 2], m[:, 3], out=np.zeros(len(m)), where=m[:, 3] > 0)
    shoot = a[:, 0, None] * np.c_[m[:, 1] * fgp * (1 - fgp), m[:, 3] * ftp * (1 - ftp)]
    return mu, var, shoot


def team_week(idx, games, mu, var, shoot, scale=1.0):
    """Weekly aggregates for a set of players: mean (W x STATK), var (W x STATK), shooting var (W x 2)."""
    g = games[idx]                                   # players x weeks
    return (scale * g.T @ mu[idx], scale * g.T @ var[idx], scale * g.T @ shoot[idx])


def phi(x):
    """Standard normal CDF (same polynomial approximation as the web board)."""
    x = np.asarray(x, float)
    t = 1 / (1 + 0.2316419 * np.abs(x))
    d = 0.3989423 * np.exp(-x * x / 2)
    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    return np.where(x > 0, 1 - p, p)


def win_probs(me, opp):
    """P(I win each category each week). me/opp = (mean, var, shoot). Returns W x 9."""
    (m1, v1, s1), (m2, v2, s2) = me, opp
    k = {s: i for i, s in enumerate(STATK)}
    out = []
    for c in CATS:
        if c in ("FG", "FT"):
            mk, at, j = (k["FGM"], k["FGA"], 0) if c == "FG" else (k["FTM"], k["FTA"], 1)
            p1, p2 = m1[:, mk] / m1[:, at], m2[:, mk] / m2[:, at]
            sd = np.sqrt(SHOOT_K * (s1[:, j] / m1[:, at] ** 2 + s2[:, j] / m2[:, at] ** 2))
            out.append(phi((p1 - p2) / sd))
        else:
            d = m1[:, k[c]] - m2[:, k[c]]
            if c == "TO":
                d = -d
            out.append(phi(d / np.sqrt(v1[:, k[c]] + v2[:, k[c]])))
    return np.c_[tuple(out)]


# ---------------------------------------------------------------- build analysis
def best_roster(score, cost, budget=BUDGET, n=ROSTER_SPOTS):
    """0/1 knapsack with exactly n players: maximise sum(score) with sum(cost) <= budget."""
    P = len(score)
    NEG = -1e18
    dp = np.full((n + 1, budget + 1), NEG)
    dp[0, 0] = 0.0
    take = np.zeros((P, n + 1, budget + 1), bool)
    for i in range(P):
        c, s = int(cost[i]), score[i]
        for k in range(n, 0, -1):
            cand = np.full(budget + 1, NEG)
            cand[c:] = dp[k - 1, :budget + 1 - c] + s
            better = cand > dp[k]
            dp[k][better] = cand[better]
            take[i, k] = better
    b = int(np.argmax(dp[n]))
    k, chosen = n, []
    for i in range(P - 1, -1, -1):
        if k and take[i, k, b]:
            chosen.append(i)
            b -= int(cost[i])
            k -= 1
    return chosen[::-1]


def analyse_builds(df, games, mu, var, shoot):
    reg = slice(0, REG_WEEKS)
    po = slice(PLAYOFF_WEEKS[0] - 1, PLAYOFF_WEEKS[-1])
    drafted = np.flatnonzero(df["rank"].to_numpy() <= POOL)
    field = team_week(drafted, games, mu, var, shoot, scale=1 / TEAMS)   # league-average opponent
    cost = np.maximum(1, np.ceil(df["base_price"].to_numpy())).astype(int)
    cand = np.flatnonzero(df["rank"].to_numpy() <= POOL + 25)
    # value over replacement in each category: every game played adds stats, so an empty roster spot
    # is worth a free agent, not an average player
    zc = df[[f"z_{c}" for c in CATS]].to_numpy()
    rk = df["rank"].to_numpy()
    repl_c = zc[(rk > POOL) & (rk <= POOL + 15)].mean(axis=0)
    season_z = (zc - repl_c) * (df["reg_games"].to_numpy()[:, None] / 62)
    rows = []
    for k in (9, 7, 6, 5):
        for targets in itertools.combinations(range(len(CATS)), k):
            score = season_z[cand][:, list(targets)].sum(axis=1)
            pick = cand[best_roster(score, cost[cand])]
            p = win_probs(team_week(pick, games, mu, var, shoot), field)
            wl = 2 * p - 1
            rows.append({
                "build": "Balanced (all 9)" if k == 9 else "Punt " + ", ".join(CATS[i] for i in range(9) if i not in targets),
                "targets": [CATS[i] for i in targets], "n_targets": k,
                "wl_week": wl[reg].sum(axis=1).mean(), "wl_season": wl[reg].sum(),
                "wl_playoff_week": wl[po].sum(axis=1).mean(),
                "cats_won_week": (p[reg] > 0.5).sum(axis=1).mean(),
                **{f"p_{c}": p[reg, j].mean() for j, c in enumerate(CATS)},
                "cost": int(cost[pick].sum()), "players": [df["name"].iat[i] for i in pick],
            })
    return pd.DataFrame(rows).sort_values("wl_week", ascending=False).reset_index(drop=True)


def main():
    df = load_players()
    sched = load_schedule()
    df = df[df["team"].isin(sched.index)].reset_index(drop=True)
    games = sched.loc[df["team"]].to_numpy(float)                 # players x 21 weeks (team games)
    df["reg_games"] = (games[:, :REG_WEEKS].sum(axis=1) * df["avail"]).round(1)
    df["po_games"] = (games[:, PLAYOFF_WEEKS[0] - 1:].sum(axis=1) * df["avail"]).round(1)
    df["po_team_games"] = games[:, PLAYOFF_WEEKS[0] - 1:].sum(axis=1).astype(int)

    # ---- per-game z-scores against the 144 players who get drafted
    pool_mask = np.ones(len(df), bool)
    for _ in range(5):
        z, sd, mu_z, fg_pct, ft_pct = per_game_z(df, pool_mask)
        zsum = z[CATS].sum(axis=1)
        pool_mask = zsum.rank(ascending=False, method="first").le(POOL).to_numpy()
    df[[f"z_{c}" for c in CATS]] = z[CATS].round(3).to_numpy()
    df["z_pg"] = zsum.round(3)

    # ---- regular-season value over replacement -> market dollars ($1 minimum per roster spot)
    ranks = zsum.rank(ascending=False, method="first")
    repl_z = float(zsum[(ranks > POOL) & (ranks <= POOL + 15)].mean())
    avg_reg = games[:, :REG_WEEKS].sum(axis=1).mean()
    df["value"] = ((df["z_pg"] - repl_z) * df["reg_games"] / avg_reg).round(3)
    top = df["value"].rank(ascending=False, method="first").le(POOL)
    pos = df["value"].clip(lower=0)
    df["base_price"] = np.where(top, 1 + pos / pos[top].sum() * (TEAMS * BUDGET - POOL), 0).round(1)
    df = df.sort_values("value", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    games = sched.loc[df["team"]].to_numpy(float)

    mu, var, shoot = player_terms(df)
    builds = analyse_builds(df, games, mu, var, shoot)
    BUILDS.parent.mkdir(exist_ok=True)
    builds.to_csv(BUILDS, index=False)

    cols = ["rank", "name", "pos", "team", "ADP", "exp_gp", "avail", "reg_games", "po_games", "po_team_games",
            "MPG", *STATK, *[f"z_{c}" for c in CATS], "z_pg", "value", "base_price"]
    data = {
        "settings": {"teams": TEAMS, "budget": BUDGET, "slots": SLOTS, "roster_spots": ROSTER_SPOTS,
                     "reg_weeks": REG_WEEKS, "playoff_weeks": PLAYOFF_WEEKS, "playoff_teams": PLAYOFF_TEAMS, "cats": CATS,
                     "disp": DISP, "shoot_k": SHOOT_K},
        "league": {"fg_pct": fg_pct, "ft_pct": ft_pct, "replacement_z": repl_z,
                   "schedule": {t: sched.loc[t].tolist() for t in sched.index}},
        "builds": builds.head(25).drop(columns=["targets"]).round(3).to_dict(orient="records")
                  + builds[builds["n_targets"] == 9].round(3).drop(columns=["targets"]).to_dict(orient="records"),
        "players": df[cols].replace({np.nan: None}).to_dict(orient="records"),
    }
    OUT.write_text("window.H2H_DATA = " + json.dumps(data, ensure_ascii=False, default=float) + ";\n", encoding="utf-8")

    print(f"{len(df)} players, pool={POOL}, replacement z/game={repl_z:.2f}")
    print(df[["rank", "name", "pos", "team", "reg_games", "po_team_games", "z_pg", "base_price"]].head(20).to_string(index=False))
    show = builds[["build", "wl_week", "wl_playoff_week", "cats_won_week", "cost"]]
    print("\nBest builds (expected weekly category W-L vs a league-average team):")
    print(show.head(12).to_string(index=False))
    print(show[builds["n_targets"] == 9].to_string(index=False, header=False))
    for k in (5, 6, 7):
        b = builds[builds["n_targets"] == k]
        print(f"{k}-category builds: best {b['wl_week'].max():+.2f}/wk, median {b['wl_week'].median():+.2f}/wk")
    print(f"\nwrote {OUT} and {BUILDS}")


if __name__ == "__main__":
    main()
