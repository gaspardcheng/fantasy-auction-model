"""Download the NBA's official game-day injury reports (PDF), one per day, 2021-22 to 2025-26.

Source: https://ak-static.cms.nba.com/referee/injury/Injury-Report_<date>_<time>.pdf (public, published by
the league for every game day since 2021-22). We take the 5 PM ET report, falling back to nearby hours.
PDFs are cached in data/injury_reports/ so the script can be re-run safely.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "injury_reports"
URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{d}_{t}.pdf"
# regular season start to the end of the Finals (playoff injuries happen before next season, so no leakage)
SEASONS = {
    "2021-22": (date(2021, 10, 19), date(2022, 6, 20)),
    "2022-23": (date(2022, 10, 18), date(2023, 6, 15)),
    "2023-24": (date(2023, 10, 24), date(2024, 6, 20)),
    "2024-25": (date(2024, 10, 22), date(2025, 6, 25)),
    "2025-26": (date(2025, 10, 21), date(2026, 6, 25)),
}
HOURS = ["05", "06", "04", "07", "01", "08", "12"]


def fetch_day(d: date):
    out = CACHE / f"{d.isoformat()}.pdf"
    if out.exists():
        return d, "cached"
    new = d >= date(2025, 8, 1)                     # file naming changed in 2025-26
    for h in HOURS:
        for t in ((f"{h}_00PM", f"{h}_30PM") if new else (f"{h}PM",)):
            time.sleep(0.5)                         # the CDN blocks fast bursts
            try:
                r = requests.get(URL.format(d=d.isoformat(), t=t), timeout=30)
            except requests.RequestException:
                continue
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                out.write_bytes(r.content)
                return d, t
    return d, "none"


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    days = [s + timedelta(n) for s, e in SEASONS.values() for n in range((e - s).days + 1)]
    with ThreadPoolExecutor(2) as ex:
        res = list(ex.map(fetch_day, days))
    got = sum(r != "none" for _, r in res)
    print(f"{got}/{len(days)} days with a report, cached in {CACHE}")


if __name__ == "__main__":
    main()
