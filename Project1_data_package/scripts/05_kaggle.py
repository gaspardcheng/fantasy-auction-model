"""Source E (optional): free Kaggle datasets. Needs a FREE Kaggle account API token (~/.kaggle/kaggle.json).
If no token is available, SKIP this step and say so in the report; never use someone else's credentials.
Datasets (check each page's licence and record it in SOURCES.md):
  ghopkins/nba-injuries-2010-2018      injury/IL transactions 2010-2018 (built from prosportstransactions)
Search Kaggle for more free NBA injury or player-season datasets; add any with an open licence and a clear source.
"""
import subprocess
from common import RAW, log

DATASETS = ["ghopkins/nba-injuries-2010-2018"]

for ds in DATASETS:
    dest = RAW / "kaggle" / ds.replace("/", "__")
    dest.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["kaggle", "datasets", "download", "-d", ds, "-p", str(dest), "--unzip"],
                       capture_output=True, text=True)
    log("kaggle", ds, "ok" if r.returncode == 0 else "failed", note=(r.stderr or r.stdout)[-300:])
    print(ds, "ok" if r.returncode == 0 else "FAILED: " + r.stderr[-300:])
