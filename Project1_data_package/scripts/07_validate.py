"""Checks before anything is handed back. Usage: python 07_validate.py "<path to the project's data/Data 1.csv>"
Writes output/validation_report.md. Every FAIL must be explained in the final report.
"""
import sys
import pandas as pd
from common import OUT, SEASONS, DATA1_COLS

ext = pd.read_csv(OUT / "Data 1 extended.csv")
lines = ["# Validation report", ""]
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    lines.append(f"- {'PASS' if cond else 'FAIL'}: {name} {detail}")


check("all 22 Data 1 columns present", all(c in ext.columns for c in DATA1_COLS))
counts = ext.groupby("SEASON").size()
check("every season present", set(SEASONS) <= set(counts.index), f"missing: {sorted(set(SEASONS) - set(counts.index))}")
check("400-750 players per season", counts.between(400, 750).all(), counts.to_dict())
check("no duplicate player-seasons", not ext.duplicated(["SEASON", "PLAYER_ID"]).any())
check("PLAYER_ID filled", ext["PLAYER_ID"].notna().mean() > 0.99, f"{ext['PLAYER_ID'].isna().sum()} missing")
check("PTS within 0..40 per game", ext["PTS"].between(0, 40).all())

if len(sys.argv) > 1 and sys.argv[1]:                                    # overlap with the project's own Data 1
    d1 = pd.read_csv(sys.argv[1])
    m = d1.merge(ext, on=["SEASON", "PLAYER_ID"], suffixes=("", "_new"))
    check("overlap seasons: same players", len(m) >= 0.98 * len(d1), f"{len(m)}/{len(d1)} matched")
    for c in ["GP", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FGA", "FTA"]:
        diff = (m[c] - m[f"{c}_new"]).abs()
        check(f"overlap {c} matches", (diff <= 0.11).mean() > 0.99, f"max diff {diff.max():.2f}")

try:
    x = pd.read_csv(OUT / "player_season_extra.csv")
    check("extra table joins 1:1", len(x) == len(ext))
    if "N_TEAMS" in x:
        lines.append(f"- INFO: share of player-seasons with 2+ teams: {(x['N_TEAMS'] > 1).mean():.1%}")
except FileNotFoundError:
    check("player_season_extra.csv exists", False)

lines += ["", "Overall: " + ("PASS" if ok else "FAIL - see lines above")]
(OUT / "validation_report.md").write_text("\n".join(lines))
print("\n".join(lines))
