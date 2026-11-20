"""Combine the raw downloads into clean tables the modeling repo can use. Never edits the project's own data.

output/Data 1 extended.csv     same 22 columns as data/Data 1.csv, every season downloaded (1996-97 .. 2025-26)
output/player_season_extra.csv  per player-season: advanced stats, bio, team movement from game logs
output/injury_reports_pre2021.csv  (if source C found PDFs; produced by the parser, see 03_injury_report_pdfs.py)
output/pst_injuries.csv        (source D)
Fallback: a season missing from nba_api is filled from Basketball-Reference per-game (one row per player:
the TOT/2TM row when traded), with PLAYER_ID matched by normalized name + season, flagged SOURCE='bref'.
"""
import glob, re, unicodedata
import pandas as pd
from common import RAW, OUT, SEASONS, DATA1_COLS, SCHEDULED_GAMES


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().replace("-", " ")
    s = re.sub(r"[^a-z ]", "", s); s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def read_all(pattern):
    files = sorted(glob.glob(str(RAW / pattern)))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = read_all("nba/base_*.csv")
    d1 = base[DATA1_COLS].assign(SOURCE="nba_api") if len(base) else pd.DataFrame(columns=DATA1_COLS + ["SOURCE"])

    # ---- fallback for seasons nba_api could not deliver
    missing = [s for s in SEASONS if s not in set(d1["SEASON"])]
    bref = read_all("bref/per_game_*.csv")
    if missing and len(bref):
        ids = {(norm(n), s): i for n, s, i in zip(d1["PLAYER_NAME"], d1["SEASON"], d1["PLAYER_ID"])}
        anyid = {norm(n): i for n, i in zip(d1["PLAYER_NAME"], d1["PLAYER_ID"])}
        b = bref[bref["SEASON"].isin(missing)].copy()
        team_col = "Team" if "Team" in b.columns else "Tm"
        b["multi"] = b[team_col].isin(["TOT", "2TM", "3TM", "4TM", "5TM"])
        b = b.sort_values("multi", ascending=False).drop_duplicates(["SEASON", "Player"])   # keep season total row
        m = {"Age": "AGE", "G": "GP", "MP": "MIN", "FG": "FGM", "FGA": "FGA", "FG%": "FG_PCT", "3P": "FG3M",
             "3PA": "FG3A", "3P%": "FG3_PCT", "FT": "FTM", "FTA": "FTA", "FT%": "FT_PCT", "PTS": "PTS",
             "TRB": "REB", "AST": "AST", "STL": "STL", "BLK": "BLK", "TOV": "TOV", team_col: "TEAM_ABBREVIATION"}
        b = b.rename(columns=m)
        b["PLAYER_NAME"] = b["Player"].str.replace("*", "", regex=False)
        b["PLAYER_ID"] = [ids.get((norm(n), s), anyid.get(norm(n))) for n, s in zip(b["PLAYER_NAME"], b["SEASON"])]
        b["SOURCE"] = "bref"
        d1 = pd.concat([d1, b[DATA1_COLS + ["SOURCE"]]], ignore_index=True)
    d1 = d1.sort_values(["SEASON", "PLAYER_NAME"])
    d1.to_csv(OUT / "Data 1 extended.csv", index=False)
    print("Data 1 extended:", d1.groupby("SEASON").size().to_dict())

    # ---- extra player-season features
    extra = d1[["SEASON", "PLAYER_ID", "PLAYER_NAME"]].copy()
    adv = read_all("nba/adv_*.csv")
    if len(adv):
        keep = [c for c in ["USG_PCT", "TS_PCT", "PACE", "PIE", "NET_RATING", "OFF_RATING", "DEF_RATING",
                            "AST_PCT", "REB_PCT", "POSS"] if c in adv.columns]
        extra = extra.merge(adv[["SEASON", "PLAYER_ID"] + keep], on=["SEASON", "PLAYER_ID"], how="left")
    bio = read_all("nba/bio_*.csv")
    if len(bio):
        keep = [c for c in ["PLAYER_HEIGHT_INCHES", "PLAYER_WEIGHT", "DRAFT_YEAR", "DRAFT_ROUND",
                            "DRAFT_NUMBER"] if c in bio.columns]
        extra = extra.merge(bio[["SEASON", "PLAYER_ID"] + keep], on=["SEASON", "PLAYER_ID"], how="left")
    logs = read_all("nba/gamelogs_*.csv")
    if len(logs):
        logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
        logs = logs.sort_values("GAME_DATE")
        # Actual games each team played per season (distinct GAME_ID), from
        # the cached game logs. This is not flat: e.g. 2019-20 teams played
        # 64-75 games around the bubble, so SCHEDULED_GAMES = 72 is wrong
        # for that season. A player with 74 GP played for POR (74 games).
        team_games = logs.groupby(["SEASON", "TEAM_ABBREVIATION"])["GAME_ID"].nunique().to_dict()
        g = logs.groupby(["SEASON", "PLAYER_ID"])
        mv = pd.DataFrame({
            "TEAM_FIRST_GAME": g["TEAM_ABBREVIATION"].first(),   # team when he first played: ~ draft-day team
            "TEAM_LAST_GAME": g["TEAM_ABBREVIATION"].last(),
            "N_TEAMS": g["TEAM_ABBREVIATION"].nunique(),
            "FIRST_GAME": g["GAME_DATE"].min().dt.date, "LAST_GAME": g["GAME_DATE"].max().dt.date,
            "GP_LOGS": g.size()}).reset_index()
        mv["TEAM_GAMES_LAST"] = [team_games.get((s, t)) for s, t in zip(mv["SEASON"], mv["TEAM_LAST_GAME"])]
        player_teams = logs.groupby(["SEASON", "PLAYER_ID"])["TEAM_ABBREVIATION"].unique().to_dict()
        mv["TEAM_GAMES_MAX"] = [
            max(team_games.get((s, t), 0) for t in player_teams[(s, p)])
            for s, p in zip(mv["SEASON"], mv["PLAYER_ID"])]
        extra = extra.merge(mv, on=["SEASON", "PLAYER_ID"], how="left")
    extra["SCHEDULED_GAMES"] = extra["SEASON"].map(SCHEDULED_GAMES).fillna(82).astype(int)
    extra.to_csv(OUT / "player_season_extra.csv", index=False)
    print("player_season_extra:", extra.shape)


if __name__ == "__main__":
    main()
