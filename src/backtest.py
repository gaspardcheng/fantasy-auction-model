"""Business-impact backtest: replay the 2025-26 auction with each projection method and score the
season with what actually happened.

Each simulation is a full 14-team, $200, 10-player auction:
  * The 13 rival GMs value players with the public benchmark (Marcel-style projections, converted to
    dollars) times their own personal opinion noise (lognormal, sd 25%).
  * We bid up to OUR dollar values, computed from the projection method being tested, with the same
    amount of personal noise as rivals. So if our method = the benchmark, we are an average GM
    (expected ~67.5 points); any gain comes from better information.
  * Players are nominated roughly in order of benchmark value. Highest willingness wins and pays the
    second-highest willingness + $1 (English auction). Willingness is capped by a team's remaining
    budget and scaled by live inflation (money left / value left). Unspent money is lost.
  * Every roster is scored with ACTUAL 2025-26 per-game stats and games, under the 8 x 82 games cap,
    with unused games streamed from replacement-level free agents; teams ranked 14..1 per category.
Rival noise and nomination order are re-used across methods (paired comparison).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import fantasy as F

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
N_SIMS = 1000
RIVAL_NOISE = 0.25
STAT_COLS = ["FGM", "FGA", "FTM", "FTA", "FG3M", "PTS", "REB", "AST", "STL", "BLK", "TOV"]


def frame(df, prefix):
    out = df[[f"{prefix}|{c}" for c in STAT_COLS + ["GP"]]].copy()
    out.columns = STAT_COLS + ["GP"]
    return out


def dollar_values(pred):
    """Auction dollars a GM would compute from a set of projections (z-scores -> value over replacement)."""
    ref = F.fit_reference(pred)
    z = F.z_scores(pred, ref)["Z"]
    v, _ = F.season_value(z, pred["GP"])
    return F.dollars(v).to_numpy()


def auction(values, order):
    """values: (teams, players) dollar valuations. Returns list of player indices per team."""
    T = values.shape[0]
    budget = np.full(T, float(F.BUDGET))
    open_ = np.full(T, F.ROSTER_SPOTS)
    rosters = [[] for _ in range(T)]
    value_left = values[:, order].sum(axis=1)                 # each team's view of value still on the board
    for p in order:
        active = open_ > 0
        if not active.any():
            break
        infl = np.where(value_left > 0, budget.sum() / np.maximum(value_left, 1e-9), 1.0)
        will = np.where(active, np.minimum(values[:, p] * infl, budget), -1.0)
        value_left -= values[:, p]
        top2 = np.argsort(-will)[:2]
        w = top2[0]
        price = 0.0 if will[w] <= 0 else min(will[w], will[top2[1]] + 1) if active.sum() > 1 else 0.0
        price = max(price, 0.0)
        rosters[w].append(p)
        budget[w] -= round(price)
        open_[w] -= 1
    return rosters


def main():
    df = pd.read_csv(RES / "test_predictions.csv")
    sel = json.load(open(RES / "selection.json"))
    methods = {"Naive (last season)": "Naive (last season)", "Marcel (weighted avg)": "Marcel (weighted avg)",
               "Our model (no injury text)": "Ensemble Ridge+RF",
               "Our model": sel["selected_model"], "Perfect foresight": "actual"}
    act = frame(df, "actual")
    ref_a = F.fit_reference(act[act["GP"] >= 15])
    z_act = F.z_scores(act, ref_a)["Z"].to_numpy()
    v_act, _ = F.season_value(pd.Series(z_act), act["GP"])
    repl_idx = np.argsort(-v_act.to_numpy())[F.POOL:F.POOL + 20]
    repl_line = act.iloc[repl_idx][STAT_COLS].mean().to_numpy()
    per_game, gp = act[STAT_COLS].to_numpy(), act["GP"].to_numpy()

    dollars = {m: dollar_values(frame(df, c)) for m, c in methods.items()}
    field = dollars["Marcel (weighted avg)"]
    n = len(df)

    rng = np.random.default_rng(42)
    rows = []
    for s in range(N_SIMS):
        rivals = field[None, :] * rng.lognormal(0, RIVAL_NOISE, (F.TEAMS - 1, n))
        own = rng.lognormal(0, RIVAL_NOISE, n)       # we are as opinionated as everyone else (symmetric game)
        order = np.argsort(-(field * rng.lognormal(0, 0.3, n)))       # nomination order
        for m, mine in dollars.items():
            rosters = auction(np.vstack([(mine * own)[None, :], rivals]), order)
            tot = pd.DataFrame([F.team_totals(per_game[t], gp[t], z_act[t], repl_line) for t in rosters])
            pts = F.roto_points(tot)
            rows.append({"sim": s, "method": m, "points": pts.loc[0, "total"],
                         "rank": int((pts["total"] > pts.loc[0, "total"]).sum() + 1),
                         **{c: pts.loc[0, c] for c in F.CATS}})
        if s % 200 == 0:
            print("sim", s)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "backtest_sims.csv", index=False)
    summ = out.groupby("method").agg(points=("points", "mean"), points_sd=("points", "std"),
                                     win=("rank", lambda r: (r == 1).mean()),
                                     top3=("rank", lambda r: (r <= 3).mean()),
                                     avg_rank=("rank", "mean")).loc[list(methods)]
    piv = out.pivot(index="sim", columns="method", values="points")
    for b in ["Naive (last season)", "Marcel (weighted avg)", "Our model (no injury text)"]:
        d = piv["Our model"] - piv[b]
        summ.loc["Our model", f"gain_vs_{b.split()[0]}"] = d.mean()
        summ.loc["Our model", f"share_sims_better_than_{b.split()[0]}"] = (d > 0).mean()
    summ.to_csv(RES / "backtest_summary.csv")
    pd.set_option("display.width", 200)
    print(summ.round(3))


if __name__ == "__main__":
    main()
