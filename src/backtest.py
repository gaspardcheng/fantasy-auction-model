"""Business-impact backtest: replay the 2025-26 auction with each projection method and score the
season with what actually happened.

Each simulation is a full 14-team, $200, 10-player auction:
  * The 13 rival GMs value players with a public benchmark converted to dollars, times their own personal
    opinion noise (lognormal, sd 25%). Three rival "rooms" are run, because the result depends on what
    rivals use: every rival on Marcel, every rival on Naive (last season), or half and half.
  * We bid up to OUR dollar values, computed from the projection method being tested, with the same
    amount of personal noise as rivals. So if our method = the room's benchmark, we are an average GM
    (expected 9 categories x 7.5 = 67.5 points); any gain comes from better information.
  * Players are nominated roughly in order of the room's benchmark value. Bids are whole dollars, at least
    $1, and each team keeps $1 for every other open roster spot. The highest bidder wins and pays the
    second-highest bid + $1 (English auction); ties are broken at random; a player nobody bids $1 on goes
    unsold. Willingness is scaled by live inflation (money left / value left). Unspent money is lost.
  * Every roster is scored with ACTUAL 2025-26 per-game stats and games, under the 8 x 82 games cap,
    with unused games (and any empty roster spot) streamed from replacement-level free agents;
    teams ranked 14..1 per category.
Rival noise, nomination noise and tie-breaks are re-used across methods and rooms (paired comparison).
Usage: python3 backtest.py [n_sims]   (default 1000)
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


def auction(values, order, rng):
    """values: (teams, players) dollar valuations. Returns list of player indices per team.
    Rules: bids are whole dollars, at least $1, and a team always keeps $1 for each other open roster spot.
    The highest bidder wins and pays the second-highest bid + $1 (English auction); ties at the top are
    broken at random. A player nobody will pay $1 for goes unsold."""
    T = values.shape[0]
    budget = np.full(T, float(F.BUDGET))
    open_ = np.full(T, F.ROSTER_SPOTS)
    rosters = [[] for _ in range(T)]
    value_left = values[:, order].sum(axis=1)                 # each team's view of value still on the board
    for p in order:
        if not (open_ > 0).any():
            break
        infl = np.where(value_left > 0, budget.sum() / np.maximum(value_left, 1e-9), 1.0)
        max_bid = np.where(open_ > 0, budget - (open_ - 1), 0.0)   # $1 held back per other open spot
        will = np.floor(np.minimum(values[:, p] * infl, max_bid))
        value_left -= values[:, p]
        top = will.max()
        if top < 1:
            continue                                          # unsold
        w = rng.choice(np.flatnonzero(will == top))
        others = np.delete(will, w)
        second = others.max() if len(others) else 0.0
        price = max(1.0, min(top, second + 1)) if second >= 1 else 1.0
        rosters[w].append(p)
        budget[w] -= price
        open_[w] -= 1
    return rosters


def rooms(dollars):
    """Rival rooms: every rival on Marcel, every rival on Naive, or half and half (alternating seats)."""
    m, nv = dollars["Marcel (weighted avg)"], dollars["Naive (last season)"]
    mixed = np.vstack([m if i % 2 == 0 else nv for i in range(F.TEAMS - 1)])
    return {"Marcel": np.tile(m, (F.TEAMS - 1, 1)), "Naive": np.tile(nv, (F.TEAMS - 1, 1)), "Mixed": mixed}


def main(n_sims=N_SIMS):
    df = pd.read_csv(RES / "test_predictions.csv")
    sel = json.load(open(RES / "selection.json"))
    methods = {"Naive (last season)": "Naive (last season)", "Marcel (weighted avg)": "Marcel (weighted avg)",
               "Ensemble Ridge+RF": "Ensemble Ridge+RF",
               "Our model": sel["selected_model"], "Perfect foresight": "actual"}
    act = frame(df, "actual")
    ref_a = F.fit_reference(act[act["GP"] >= 15])
    z_act = F.z_scores(act, ref_a)["Z"].to_numpy()
    v_act, _ = F.season_value(pd.Series(z_act), act["GP"])
    repl_idx = np.argsort(-v_act.to_numpy())[F.POOL:F.POOL + 20]
    repl_line = act.iloc[repl_idx][STAT_COLS].mean().to_numpy()
    per_game, gp = act[STAT_COLS].to_numpy(), act["GP"].to_numpy()

    dollars = {m: dollar_values(frame(df, c)) for m, c in methods.items()}
    room_vals = rooms(dollars)
    n = len(df)

    rng = np.random.default_rng(42)
    rows = []
    for s in range(n_sims):
        # one set of draws per simulation, shared by every method and every room (paired comparison)
        rival_noise = rng.lognormal(0, RIVAL_NOISE, (F.TEAMS - 1, n))
        own = rng.lognormal(0, RIVAL_NOISE, n)       # we are as opinionated as everyone else (symmetric game)
        nom_noise = rng.lognormal(0, 0.3, n)
        tie_seed = rng.integers(2**32)
        for room, base in room_vals.items():
            rivals = base * rival_noise
            order = np.argsort(-(base.mean(axis=0) * nom_noise))     # nomination order: roughly by room value
            for m, mine in dollars.items():
                rosters = auction(np.vstack([(mine * own)[None, :], rivals]), order, np.random.default_rng(tie_seed))
                tot = pd.DataFrame([F.team_totals(per_game[t], gp[t], z_act[t], repl_line) for t in rosters])
                pts = F.roto_points(tot)
                rows.append({"sim": s, "room": room, "method": m, "points": pts.loc[0, "total"],
                             "rank": int((pts["total"] > pts.loc[0, "total"]).sum() + 1),
                             "roster": len(rosters[0]), **{c: pts.loc[0, c] for c in F.CATS}})
        if s % 100 == 0:
            print("sim", s, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "backtest_sims.csv", index=False)
    summ = out.groupby(["room", "method"]).agg(
        points=("points", "mean"), points_sd=("points", "std"), win=("rank", lambda r: (r == 1).mean()),
        top3=("rank", lambda r: (r <= 3).mean()), avg_rank=("rank", "mean"), roster=("roster", "mean"))
    for room in room_vals:
        piv = out[out["room"] == room].pivot(index="sim", columns="method", values="points")
        for b in ["Naive (last season)", "Marcel (weighted avg)", "Ensemble Ridge+RF"]:
            d = piv["Our model"] - piv[b]
            se = d.std() / np.sqrt(len(d))
            summ.loc[(room, "Our model"), f"gain_vs_{b.split()[0]}"] = d.mean()
            summ.loc[(room, "Our model"), f"gain_vs_{b.split()[0]}_se"] = se
            summ.loc[(room, "Our model"), f"share_sims_better_than_{b.split()[0]}"] = (d > 0).mean()
    summ = summ.reindex([(r, m) for r in room_vals for m in methods])
    summ.to_csv(RES / "backtest_summary.csv")
    pd.set_option("display.width", 250)
    print(f"{n_sims} paired simulations per room. Sanity check: a GM using the same values as the room "
          f"should average {len(F.CATS) * (F.TEAMS + 1) / 2:.1f} points.")
    print(summ.round(3))


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_SIMS)
