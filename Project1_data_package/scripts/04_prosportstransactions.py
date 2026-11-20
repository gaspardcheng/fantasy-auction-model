"""Source D: injury / injured-list transactions from prosportstransactions.com (free, public, back to the 1940s).

One row per transaction: date, team, player acquired (activated / returned), player relinquished (placed on IL /
out with injury), notes (free text: 'placed on IL with sprained left ankle'). This is the only free source that
covers injuries before the league's PDFs, and it also covers OFFSEASON injuries the PDFs miss.
Search query: both 'Movement to/from injured/inactive list' and 'Missed games due to injury' boxes, NBA, one
season window at a time, 25 rows per page. 3 s between pages, cached, resumable.
Output: raw/pst/pst_<season>.csv and output/pst_injuries.csv (all seasons).
"""
import sys
from io import StringIO
import pandas as pd
from common import RAW, OUT, SEASONS, polite_get, log

D = RAW / "pst"
URL = "https://www.prosportstransactions.com/basketball/Search/SearchResults.php"


def season_window(s):
    y = int(s[:4])
    return f"{y}-07-01", f"{y + 1}-06-30"            # July..June so offseason injuries belong to the coming season


def fetch_season(s):
    out = D / f"pst_{s}.csv"
    if out.exists():
        return pd.read_csv(out)
    begin, end = season_window(s)
    frames, start = [], 0
    while True:
        params = {"Player": "", "Team": "", "BeginDate": begin, "EndDate": end, "ILChkBx": "yes",
                  "InjuriesChkBx": "yes", "Submit": "Search", "start": start}
        html = polite_get(URL, D / "html", min_interval=3.0, params=params)
        if html is None:
            log("pst", f"{s} start={start}", "failed"); break
        try:
            t = [x for x in pd.read_html(StringIO(html), header=0) if "Notes" in x.columns]
        except ValueError:
            t = []
        if not t or t[0].empty:
            break
        frames.append(t[0]); start += 25
        if len(t[0]) < 25:
            break
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "season", s)
    D.mkdir(parents=True, exist_ok=True); df.to_csv(out, index=False)
    log("pst", s, "ok", len(df)); print(f"pst {s}: {len(df)} rows")
    return df


if __name__ == "__main__":
    allp = pd.concat([fetch_season(s) for s in (sys.argv[1:] or SEASONS)], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True); allp.to_csv(OUT / "pst_injuries.csv", index=False)
