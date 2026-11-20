"""Shared settings, polite HTTP with caching, and logging for every fetch script."""
import json, time, hashlib
from pathlib import Path
import requests

PKG = Path(__file__).resolve().parents[1]
RAW = PKG / "raw"            # untouched downloads (cache; safe to re-run)
OUT = PKG / "output"         # cleaned tables for the modeling repo
LOG = PKG / "output" / "download_log.jsonl"

# nba.com league dashboards go back to 1996-97. Pull everything; the modeling team chooses the window.
FIRST, LAST = 1996, 2025     # start years -> seasons 1996-97 ... 2025-26
SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(FIRST, LAST + 1)]
SCHEDULED_GAMES = {"1998-99": 50, "2011-12": 66, "2019-20": 72, "2020-21": 72}   # default 82

DATA1_COLS = ["SEASON", "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE", "GP", "MIN", "FGM", "FGA",
              "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT", "PTS", "REB", "AST", "STL", "BLK", "TOV"]

UA = "Mozilla/5.0 (academic project, MIT 15.071; contact via course staff)"


def log(source, item, status, rows=None, note=""):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "source": source, "item": item,
                            "status": status, "rows": rows, "note": note}) + "\n")


def polite_get(url, cache_dir, min_interval, params=None, tries=4):
    """GET with an on-disk cache, a fixed minimum gap between live requests, and exponential backoff.
    Returns text, or None after repeated failure. Never bypasses logins, captchas or blocks."""
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
    path = cache_dir / f"{key}.html"
    if path.exists():
        return path.read_text(errors="ignore")
    wait = min_interval
    for _ in range(tries):
        time.sleep(wait)
        try:
            r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=60)
        except requests.RequestException:
            wait *= 2; continue
        if r.status_code == 200:
            path.write_text(r.text); return r.text
        if r.status_code in (403, 429):          # blocked or rate-limited: back off hard, then give up
            wait = max(wait * 4, 60)
        else:
            wait *= 2
    return None
