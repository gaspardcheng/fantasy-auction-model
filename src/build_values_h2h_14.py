"""
H2H auction values + build analysis for the 14-team H2H board (docs/h2h14.html).

Same method as build_values_h2h.py, with this league:
  14 teams, $200, 12 roster spots = PG, SG, G, SF, PF, F, C, C, UTIL, UTIL, BN, BN.
  Every rostered player costs at least $1. Draftable pool = 14 x 12 = 168 players.
  9 categories scored weekly; weeks 1-18 regular season, top 8 make the playoffs (weeks 19-21).

Output: docs/players_h2h_14.js and results/h2h_builds_14.csv, leaving the 12-team H2H files untouched.
"""
import build_values_h2h as bv

bv.TEAMS = 14
bv.POOL = bv.TEAMS * bv.ROSTER_SPOTS
bv.OUT = bv.ROOT / "docs" / "players_h2h_14.js"
bv.BUILDS = bv.ROOT / "results" / "h2h_builds_14.csv"

if __name__ == "__main__":
    bv.main()
