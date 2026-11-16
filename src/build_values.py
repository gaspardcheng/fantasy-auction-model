"""
Fantasy basketball auction valuation (14-team, 9-cat rotisserie, games-played caps).

Inputs:
  data/Data 1.csv   - per-game stats per player for the past 5 seasons (NBA stats export)
  data/Data 2.xlsx  - 2026-27 per-game projections (one row per player, POS = Yahoo eligibility)
  results/projections_2026_27.csv - (optional) Data 2 blended with our ML model (src/predict_2026.py)

Output:
  docs/players.js  - window.FANTASY_DATA = {...}, loaded by the auction board (docs/index.html)

Method
  1. Per-game category scores. FG% and FT% use "impact" = (player% - pool%) * attempts,
     so a high percentage on high volume counts most. TO is negated (fewer is better).
  2. Z-score each category against the pool of draftable players (14 teams x 10 roster spots),
     iterating so the pool mean/std are computed on the players who actually get drafted.
  3. Health: expected games = blended GP projection (ML model learned from past games played);
     without the ML file, a blend of projected GP and the past-5-season GP rate.
  4. Value over replacement (VORP). With an 82-game cap per slot and unlimited free-agent
     pickups, a missed game is filled by a replacement-level free agent, so
        season value = (z per game - replacement z per game) * expected games / 82
  5. Auction dollars: league money (14 x 200) split in proportion to VORP among the draftable pool.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- league settings
TEAMS = 14
BUDGET = 200
ROSTER_SPOTS = 10          # 8 active (PG, SG, SF, PF, C, UTIL x3) + 2 bench; IL is empty at the auction
ACTIVE_SLOTS = 8
GAMES_CAP = 82             # per active slot
CATS = ["FG", "FT", "3PM", "PTS", "REB", "AST", "STL", "BLK", "TO"]
HEALTH_WEIGHT_HISTORY = 0.4  # share of expected GP that comes from past seasons (rest = projection)
SEASON_GAMES = {"2021-22": 82, "2022-23": 82, "2023-24": 82, "2024-25": 82, "2025-26": 82}

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BLENDED = ROOT / "results" / "projections_2026_27.csv"
OUT = ROOT / "docs" / "players.js"


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower().replace("-", " "))
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def load_projections() -> pd.DataFrame:
    raw = pd.read_excel(DATA / "Data 2.xlsx")
    df = raw[raw["PLAYER"].notna() & (raw["PLAYER"] != "PLAYER")].copy()  # drop repeated header rows

    def split_pair(col):
        nums = df[col].astype(str).str.extract(r"\(([\d.]+)/([\d.]+)\)").astype(float)
        return nums[0], nums[1]

    df["FGM"], df["FGA"] = split_pair("FGM/FGA")
    df["FTM"], df["FTA"] = split_pair("FTM/FTA")
    num = ["GP", "MPG", "3PM", "PTS", "TREB", "AST", "STL", "BLK", "TO"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df["ADP"] = pd.to_numeric(df["ADP"], errors="coerce")
    df = df.rename(columns={"PLAYER": "name", "POS": "pos", "TEAM": "team", "TREB": "REB"})
    df["pos"] = df["pos"].astype(str).str.replace(" ", "")
    df = df.dropna(subset=["GP", "FGA", "FTA", "PTS"]).reset_index(drop=True)
    return df[["name", "pos", "team", "ADP", "GP", "MPG", "FGM", "FGA", "FTM", "FTA",
               "3PM", "PTS", "REB", "AST", "STL", "BLK", "TO"]]


def load_history() -> pd.DataFrame:
    h = pd.read_csv(DATA / "Data 1.csv")
    h["key"] = h["PLAYER_NAME"].map(norm_name)
    h = h.groupby(["key", "SEASON"], as_index=False)["GP"].max()  # one row per player-season
    h["rate"] = h["GP"] / h["SEASON"].map(SEASON_GAMES).fillna(82)
    # recent seasons count more: weights 1..5 from oldest to newest
    order = {s: i + 1 for i, s in enumerate(sorted(SEASON_GAMES))}
    h["w"] = h["SEASON"].map(order).fillna(1)
    g = h.groupby("key")[["SEASON", "GP", "rate", "w"]].apply(
        lambda x: pd.Series({
            "hist_gp_rate": np.average(x["rate"], weights=x["w"]),
            "hist_seasons": len(x),
            "hist_gp": " / ".join(f"{s[2:]}:{int(gp)}" for s, gp in zip(x["SEASON"], x["GP"])),
        }))
    return g.reset_index()


def per_game_z(df: pd.DataFrame, pool_mask: np.ndarray) -> pd.DataFrame:
    pool = df[pool_mask]
    fg_pct = pool["FGM"].sum() / pool["FGA"].sum()
    ft_pct = pool["FTM"].sum() / pool["FTA"].sum()
    raw = pd.DataFrame({
        "FG": df["FGM"] - fg_pct * df["FGA"],      # impact = (FG% - league FG%) * FGA
        "FT": df["FTM"] - ft_pct * df["FTA"],
        "3PM": df["3PM"], "PTS": df["PTS"], "REB": df["REB"], "AST": df["AST"],
        "STL": df["STL"], "BLK": df["BLK"], "TO": -df["TO"],
    })
    mu, sd = raw[pool_mask].mean(), raw[pool_mask].std()
    return (raw - mu) / sd, sd, mu, fg_pct, ft_pct


def main():
    use_ml = BLENDED.exists()
    df = pd.read_csv(BLENDED) if use_ml else load_projections()
    hist = load_history()
    df["key"] = df["name"].map(norm_name)
    df = df.merge(hist, on="key", how="left")

    # ---- health: expected games played
    proj_rate = df["GP"] / 82
    has_hist = df["hist_seasons"].fillna(0) >= 2
    w = np.where(has_hist & (not use_ml), HEALTH_WEIGHT_HISTORY, 0.0)
    df["exp_gp"] = (82 * ((1 - w) * proj_rate + w * df["hist_gp_rate"].fillna(proj_rate))).round(1)

    # ---- iterate z-scores on the draftable pool
    n_pool = TEAMS * ROSTER_SPOTS
    pool_mask = np.ones(len(df), dtype=bool)
    for _ in range(5):
        z, sd, mu, fg_pct, ft_pct = per_game_z(df, pool_mask)
        zsum = z[CATS].sum(axis=1)
        pool_mask = zsum.rank(ascending=False, method="first").le(n_pool).to_numpy()

    df[[f"z_{c}" for c in CATS]] = z[CATS].round(3).to_numpy()
    df["z_pg"] = zsum.round(3)

    # ---- replacement level: average of the best ~15 undrafted players (the free-agent pool)
    ranks = zsum.rank(ascending=False, method="first")
    repl_mask = (ranks > n_pool) & (ranks <= n_pool + 15)
    repl_z = float(zsum[repl_mask].mean()) if repl_mask.any() else float(zsum.min())
    repl_stats = df.loc[repl_mask, ["FGM", "FGA", "FTM", "FTA", "3PM", "PTS", "REB", "AST", "STL", "BLK", "TO"]].mean()

    # ---- season value over replacement (health-adjusted)
    df["vorp"] = ((df["z_pg"] - repl_z) * df["exp_gp"] / 82).round(3)
    pos_vorp = df["vorp"].clip(lower=0)
    top = df["vorp"].rank(ascending=False, method="first").le(n_pool)
    total = pos_vorp[top].sum()
    league_money = TEAMS * BUDGET
    df["base_price"] = np.where(top, pos_vorp / total * league_money, 0).round(1)
    df = df.sort_values("vorp", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    # ---- spread of a team's category totals (used by the board to turn totals into expected roto points)
    rng = np.random.default_rng(7)
    drafted = df[df["rank"] <= n_pool]
    totals = []
    for _ in range(3000):
        t = drafted.sample(ROSTER_SPOTS, random_state=int(rng.integers(1e9)))
        totals.append(team_totals(t, repl_stats))
    sigma = pd.DataFrame(totals).std().to_dict()

    cols = ["rank", "name", "pos", "team", "ADP", "GP", "exp_gp", "hist_gp", "MPG",
            "FGM", "FGA", "FTM", "FTA", "3PM", "PTS", "REB", "AST", "STL", "BLK", "TO",
            *[f"z_{c}" for c in CATS], "z_pg", "vorp", "base_price"]
    out = df[cols].replace({np.nan: None})
    data = {
        "settings": {"teams": TEAMS, "budget": BUDGET, "roster_spots": ROSTER_SPOTS,
                     "active_slots": ACTIVE_SLOTS, "games_cap": GAMES_CAP, "cats": CATS},
        "league": {"fg_pct": fg_pct, "ft_pct": ft_pct, "z_mean": mu.to_dict(), "z_sd": sd.to_dict(),
                   "replacement_z": repl_z, "replacement_stats": repl_stats.to_dict(), "team_sigma": sigma},
        "players": out.to_dict(orient="records"),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("window.FANTASY_DATA = " + json.dumps(data, ensure_ascii=False, default=float) + ";\n",
                   encoding="utf-8")

    print(f"{len(df)} players, pool={n_pool}, replacement z/game={repl_z:.2f}")
    print(f"Matched to history: {df['hist_seasons'].notna().sum() if 'hist_seasons' in df else 'n/a'}")
    print(df[["rank", "name", "pos", "exp_gp", "z_pg", "vorp", "base_price"]].head(25).to_string(index=False))
    print(f"\nwrote {OUT}")


def team_totals(players: pd.DataFrame, repl: pd.Series) -> dict:
    """Season category totals for a roster under the games cap (same logic as the web board)."""
    cap = ACTIVE_SLOTS * GAMES_CAP
    p = players.sort_values("z_pg", ascending=False)
    games_left = cap
    tot = {k: 0.0 for k in ["FGM", "FGA", "FTM", "FTA", "3PM", "PTS", "REB", "AST", "STL", "BLK", "TO"]}
    for _, r in p.iterrows():
        g = min(r["exp_gp"], games_left)
        games_left -= g
        for k in tot:
            tot[k] += r[k] * g
    for k in tot:                       # empty games are streamed with replacement-level free agents
        tot[k] += repl[k] * games_left
    return {"FG": tot["FGM"] / tot["FGA"], "FT": tot["FTM"] / tot["FTA"], "3PM": tot["3PM"],
            "PTS": tot["PTS"], "REB": tot["REB"], "AST": tot["AST"], "STL": tot["STL"],
            "BLK": tot["BLK"], "TO": tot["TO"]}


if __name__ == "__main__":
    main()
