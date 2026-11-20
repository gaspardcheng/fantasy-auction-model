"""Source A (primary): NBA.com stats via the free, open-source `nba_api` package.

Same source and column names as the project's `Data 1.csv`, so the extended table drops straight into the pipeline.
Per season, 1996-97 .. 2025-26, regular season:
  A1 per-game box score   (LeagueDashPlayerStats, Base, PerGame)      -> raw/nba/base_<season>.csv
  A2 advanced             (LeagueDashPlayerStats, Advanced, PerGame)  -> raw/nba/adv_<season>.csv   (USG%, TS%, PACE, PIE ...)
  A3 every player game    (PlayerGameLogs)                            -> raw/nba/gamelogs_<season>.csv (team on each date)
  A4 bio                  (LeagueDashPlayerBioStats)                  -> raw/nba/bio_<season>.csv   (height, weight, draft)
Re-runnable: finished files are skipped. ~4 calls per season, 2.5 s apart plus backoff.
"""
import sys, time
import pandas as pd
from common import RAW, SEASONS, log

from nba_api.stats.endpoints import leaguedashplayerstats, playergamelogs, leaguedashplayerbiostats

D = RAW / "nba"; D.mkdir(parents=True, exist_ok=True)
PAUSE = 2.5


def call(fn, name, season, tries=5):
    out = D / f"{name}_{season}.csv"
    if out.exists():
        return
    wait = PAUSE
    for k in range(tries):
        time.sleep(wait)
        try:
            df = fn().get_data_frames()[0]
            df.insert(0, "SEASON", season)
            df.to_csv(out, index=False)
            log("nba_api", f"{name} {season}", "ok", len(df))
            print(f"{name:9s} {season} {len(df):6d} rows")
            return
        except Exception as e:                       # timeouts are common on stats.nba.com: back off, retry
            wait = min(wait * 3, 120)
            last = repr(e)[:200]
    log("nba_api", f"{name} {season}", "failed", note=last)
    print(f"FAILED {name} {season}: {last}", file=sys.stderr)


def main(seasons=SEASONS):
    for s in seasons:
        call(lambda: leaguedashplayerstats.LeagueDashPlayerStats(season=s, per_mode_detailed="PerGame",
             measure_type_detailed_defense="Base", season_type_all_star="Regular Season", timeout=60), "base", s)
        call(lambda: leaguedashplayerstats.LeagueDashPlayerStats(season=s, per_mode_detailed="PerGame",
             measure_type_detailed_defense="Advanced", season_type_all_star="Regular Season", timeout=60), "adv", s)
        call(lambda: playergamelogs.PlayerGameLogs(season_nullable=s, season_type_nullable="Regular Season",
             timeout=90), "gamelogs", s)
        call(lambda: leaguedashplayerbiostats.LeagueDashPlayerBioStats(season=s,
             season_type_all_star="Regular Season", timeout=60), "bio", s)


if __name__ == "__main__":
    main(sys.argv[1:] or SEASONS)
