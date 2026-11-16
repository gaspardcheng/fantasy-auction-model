"""Parse the cached injury-report PDFs into one table: data/injury_reports.csv.

One row = (report date, player, status, reason), e.g.
  2022-01-15, Damian Lillard, Out, "Injury/Illness - Core; muscle recovery"

Layout parsing works on word positions rather than PDF tables, because the 2025-26 reports wrap long
reasons onto a line above and a line below the player's row (and lose their table structure).
"""
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "injury_reports"
OUT = ROOT / "data" / "injury_reports.csv"
STATUSES = {"Out", "Doubtful", "Questionable", "Probable", "Available"}


def season_of(d: pd.Timestamp) -> str:
    y = d.year if d.month >= 8 else d.year - 1
    return f"{y}-{str(y + 1)[2:]}"


def columns(page):
    """x positions of the Player / Status / Reason columns and the header's bottom, from page 1's header."""
    words = page.extract_words(x_tolerance=1.5)
    head = {w["text"]: w for w in words if w["top"] < 125}
    if "Reason" not in head:
        return None
    player = head.get("Player") or head.get("PlayerName")
    status = head.get("Current") or head.get("CurrentStatus")
    if player is None or status is None:
        return None
    return player["x0"] - 3, status["x0"] - 3, head["Reason"]["x0"] - 3, head["Reason"]["top"]


def parse_page(page, cols, first):
    x_player, x_status, x_reason, hdr_top = cols
    start = hdr_top + 2 if first else 60                          # later pages: no column header, only the title
    words = page.extract_words(x_tolerance=1.5)

    lines = {}                                                    # group words into text lines
    for w in words:
        if w["top"] <= start or w["text"].startswith("Page"):
            continue
        key = next((k for k in lines if abs(k - w["top"]) < 2.5), w["top"])
        lines.setdefault(key, []).append(w)

    rows, orphans = [], []
    for top, ws in sorted(lines.items()):
        ws.sort(key=lambda w: w["x0"])
        player = " ".join(w["text"] for w in ws if x_player <= w["x0"] < x_status)
        status = " ".join(w["text"] for w in ws if x_status <= w["x0"] < x_reason)
        reason = " ".join(w["text"] for w in ws if w["x0"] >= x_reason)
        if "NOT YET SUBMITTED" in reason.upper():
            continue
        if player and status in STATUSES:
            rows.append({"top": top, "player": player, "status": status, "parts": [(top, reason)] if reason else []})
        elif reason:
            orphans.append((top, reason))
    for top, reason in orphans:                                   # wrapped reason -> nearest player row
        if rows:
            r = min(rows, key=lambda r: abs(r["top"] - top))
            if abs(r["top"] - top) < 20:
                r["parts"].append((top, reason))
    out = []
    for r in rows:
        reason = " ".join(t for _, t in sorted(r["parts"]))
        reason = re.sub(r"\s*-\s*", " - ", reason, count=1)
        out.append({"player": r["player"], "status": r["status"], "reason": re.sub(r"\s+", " ", reason).strip()})
    return out


def parse_file(path: Path):
    d = pd.Timestamp(path.stem)
    try:
        with pdfplumber.open(path) as pdf:
            cols = columns(pdf.pages[0])
            if cols is None:
                print("no header", path.name)
                return []
            rows = [r for i, p in enumerate(pdf.pages) for r in parse_page(p, cols, i == 0)]
    except Exception as e:                                        # a few PDFs are malformed
        print("skip", path.name, e)
        return []
    for r in rows:
        last, _, first = r["player"].partition(",")
        r["player"] = f"{first.strip()} {last.strip()}".strip()
        r["date"], r["season"] = d.date().isoformat(), season_of(d)
    return rows


def main():
    files = sorted(CACHE.glob("*.pdf"))
    with ProcessPoolExecutor() as ex:
        rows = [r for rs in ex.map(parse_file, files, chunksize=8) for r in rs]
    df = pd.DataFrame(rows)[["date", "season", "player", "status", "reason"]].drop_duplicates()
    df.to_csv(OUT, index=False)
    print(f"{len(df):,} rows from {len(files)} reports -> {OUT}")
    print(df.groupby("season").agg(rows=("player", "size"), players=("player", "nunique")))


if __name__ == "__main__":
    main()
