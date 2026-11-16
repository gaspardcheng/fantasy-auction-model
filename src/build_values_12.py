"""
Auction values for the 12-team roto board (docs/roto12.html).

Same method as build_values.py, with the 12-team league:
  12 teams, $200, 12 roster spots = PG, SG, G, SF, PF, F, C, C, UTIL, UTIL (10 active) + BN, BN.
  Draftable pool = 12 x 12 = 144 players; 10 active slots x 82-game cap.

Output: docs/players_12.js (window.FANTASY_DATA), leaving docs/players.js (14-team board) untouched.
"""
import build_values as bv

SLOTS = ["PG", "SG", "G", "SF", "PF", "F", "C", "C", "UTIL", "UTIL", "BN", "BN"]

bv.TEAMS = 12
bv.ROSTER_SPOTS = len(SLOTS)
bv.ACTIVE_SLOTS = 10
bv.OUT = bv.ROOT / "docs" / "players_12.js"

if __name__ == "__main__":
    bv.main()
