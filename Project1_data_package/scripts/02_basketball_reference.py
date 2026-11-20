"""Source B (backup + cross-check): Basketball-Reference season tables.

Use when nba_api fails for a season, and to cross-check A1. Also gives every team a traded player played for
(the per-team rows under a 'TOT'/'2TM' line).
Pages per season (year = season end, e.g. 2019 for 2018-19): NBA_<year>_per_game.html, NBA_<year>_advanced.html
Sports-Reference asks bots to stay under 20 requests per minute; we wait 4 s between requests (15/min) and cache
everything, so a full run of 60 pages takes about 4-5 minutes and never repeats a request.
"""
import sys
from io import StringIO
import pandas as pd
from common import RAW, SEASONS, polite_get, log

D = RAW / "bref"
URL = "https://www.basketball-reference.com/leagues/NBA_{y}_{page}.html"


def fetch(season, page):
    y = int(season[:4]) + 1
    out = D / f"{page}_{season}.csv"
    if out.exists():
        return
    html = polite_get(URL.format(y=y, page=page), D / "html", min_interval=4.0)
    if html is None:
        log("bref", f"{page} {season}", "failed"); return
    html = html.replace("<!--", "").replace("-->", "")         # some tables sit inside HTML comments
    tables = pd.read_html(StringIO(html))
    df = max(tables, key=len)
    df = df[df["Player"] != "Player"].copy()                    # drop repeated header rows
    df.insert(0, "SEASON", season)
    D.mkdir(parents=True, exist_ok=True); df.to_csv(out, index=False)
    log("bref", f"{page} {season}", "ok", len(df))
    print(f"bref {page:8s} {season} {len(df)} rows")


if __name__ == "__main__":
    for s in (sys.argv[1:] or SEASONS):
        for p in ("per_game", "advanced"):
            fetch(s, p)
