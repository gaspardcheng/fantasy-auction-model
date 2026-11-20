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
CACHE = ROOT / "raw" / "injury_pdfs"
OUT = ROOT / "output" / "injury_reports_pre2021.csv"
STATUSES = {"Out", "Doubtful", "Questionable", "Probable", "Available"}


def season_of(d: pd.Timestamp) -> str:
    # Date ranges, not just month: the 2019-20 season ran to 11 Oct 2020
    # (bubble), so Aug-Oct 2020 reports belong to 2019-20, not 2020-21.
    if d <= pd.Timestamp("2019-06-13"):
        return "2018-19"
    if d <= pd.Timestamp("2020-10-11"):
        return "2019-20"
    if d <= pd.Timestamp("2021-07-20"):
        return "2020-21"
    y = d.year if d.month >= 8 else d.year - 1
    return f"{y}-{str(y + 1)[2:]}"


def columns(page):
    """x positions of each column header on page 1, as a dict.

    Column order differs by season, so do not assume it:
      2018-19: Player | Category | Reason | Current Status | Previous Status
      2019-20 (some): Player | Current Status | Reason | Previous Status
      2020-21: Player | Current Status | Reason
    Returns {name: x0} for every header present, or None if the
    Player/Reason/Current-Status headers are missing.
    """
    words = page.extract_words(x_tolerance=1.5)
    head = {w["text"]: w for w in words if w["top"] < 125}
    if "Reason" not in head:
        return None
    player = head.get("Player") or head.get("PlayerName")
    status = head.get("Current") or head.get("CurrentStatus")
    if player is None or status is None:
        return None
    cols = {"player": player["x0"] - 3, "status": status["x0"] - 3,
            "reason": head["Reason"]["x0"] - 3, "hdr_top": head["Reason"]["top"]}
    if "Category" in head:
        cols["category"] = head["Category"]["x0"] - 3
    if "Previous" in head:
        cols["previous"] = head["Previous"]["x0"] - 3
    return cols


def parse_page(page, cols, first):
    hdr_top = cols["hdr_top"]
    # column intervals in x order: each column runs to the next column's x0
    bounds = sorted((x, name) for name, x in cols.items() if name != "hdr_top")
    bounds.append((10 ** 9, None))

    def field(ws, name):
        xs = [x for x, n in bounds if n == name]
        if not xs:
            return ""
        lo = xs[0]
        hi = min((x for x, n in bounds if x > lo), default=10 ** 9)
        return " ".join(w["text"] for w in ws if lo <= w["x0"] < hi)

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
        player = field(ws, "player")
        status = field(ws, "status")
        reason = field(ws, "reason")
        category = field(ws, "category")
        if "NOT YET SUBMITTED" in reason.upper():
            continue
        if player and status in STATUSES:
            rows.append({"top": top, "player": player, "status": status, "category": category,
                         "parts": [(top, reason)] if reason else []})
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
        reason = re.sub(r"\s+", " ", reason).strip()
        category = re.sub(r"\s+", " ", r.get("category", "")).strip()
        # 2018-19 reports carry the category in its own column, so the
        # reason on its own is bare ("Right Ankle"); later seasons already
        # embed it ("Injury/Illness - Right Ankle"). Prepend it when present
        # so every season matches the later format. A blank reason ("-")
        # with a category becomes "<Category> -" and nothing after the dash.
        if category:
            if reason in ("", "-"):
                reason = f"{category} -"
            elif not reason.startswith(category):
                reason = f"{category} - {reason}"
        out.append({"player": r["player"], "status": r["status"], "reason": reason})
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
