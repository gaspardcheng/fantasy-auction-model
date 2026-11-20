"""Source C: the NBA's official game-day injury report PDFs for seasons BEFORE 2021-22.

The project already has 2021-22 .. 2025-26. This script first PROBES whether older seasons exist on the same
public CDN (10 sample dates per season). Only seasons that answer get a full download.
URL pattern (unchanged from the repo's fetch_injury_reports.py):
  https://ak-static.cms.nba.com/referee/injury/Injury-Report_<YYYY-MM-DD>_<HH>PM.pdf
Then parse with scripts/parse_injury_reports_from_repo.py (point its CACHE/OUT at raw/injury_pdfs and
output/injury_reports_pre2021.csv) so rows match data/injury_reports.csv: date, season, player, status, reason.
"""
import random, time
from datetime import date, timedelta
import requests
from common import RAW, log

D = RAW / "injury_pdfs"
URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{d}_{t}.pdf"
OLD = {   # regular season start .. end of Finals
    "2017-18": (date(2017, 10, 17), date(2018, 6, 8)),
    "2018-19": (date(2018, 10, 16), date(2019, 6, 13)),
    "2019-20": (date(2019, 10, 22), date(2020, 10, 11)),   # includes the 2020 bubble restart
    "2020-21": (date(2020, 12, 22), date(2021, 7, 20)),
}
HOURS = ["05PM", "06PM", "04PM", "07PM", "01PM", "08PM", "12PM", "05_00PM", "05_30PM"]


def fetch(d):
    out = D / f"{d.isoformat()}.pdf"
    if out.exists():
        return True
    for t in HOURS:
        time.sleep(0.6)
        try:
            r = requests.get(URL.format(d=d.isoformat(), t=t), timeout=30)
        except requests.RequestException:
            continue
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            out.write_bytes(r.content); return True
    return False


def main():
    D.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    for season, (a, b) in OLD.items():
        days = [a + timedelta(n) for n in range((b - a).days + 1)]
        probe = random.sample(days[:150], 10)                   # regular-season sample dates
        hits = sum(fetch(d) for d in probe)
        log("injury_pdfs", f"probe {season}", "ok" if hits else "absent", hits)
        print(f"{season}: {hits}/10 probe dates found")
        if hits >= 3:
            got = sum(fetch(d) for d in days)
            log("injury_pdfs", season, "ok", got)
            print(f"  downloaded {got}/{len(days)} days")


if __name__ == "__main__":
    main()
