# Sources and terms of use (fill in while downloading)

| ID | Source | URL | Cost | Terms checked (date + one-line summary) | Redistribute in public repo? |
|---|---|---|---|---|---|
| A | NBA.com stats via `nba_api` (MIT-licensed client) | https://www.nba.com/stats, https://github.com/swar/nba_api | Free | 2026-11-17: free public stats via MIT-licensed client; 120/120 calls ok, no login/block. Underlying NBA data redistribution not yet verified — do not publish raw data until final check. | Not yet confirmed — see REPORT.md |
| B | Basketball-Reference | https://www.basketball-reference.com, https://www.sports-reference.com/data_use.html, https://www.sports-reference.com/bot-traffic.html | Free | 2026-11-17: 2018-19 per-game + advanced only, 2 pages, both ok (709 rows each); cross-check vs A: 527/530 matched, stats agree exactly. 4 s between requests, under 20/min. Redistribution not yet verified. | Not yet confirmed — see REPORT.md |
| C | NBA official injury report PDFs | https://official.nba.com (CDN ak-static.cms.nba.com) | Free | 2026-11-17: free public PDFs, no login/block; 587 downloaded, 39,976 rows parsed. Underlying NBA data redistribution not yet verified — do not publish raw PDFs/data until final check. | Not yet confirmed — see REPORT.md |
| D | Pro Sports Transactions | https://www.prosportstransactions.com/basketball/ | Free | 2026-11-17: BLOCKED — site returned 403; run stopped per hard rules, 0 rows, no data delivered. Terms not checked (no data obtained). | No data — nothing to redistribute |
| E | Kaggle: ghopkins/nba-injuries-2010-2018 | https://www.kaggle.com/datasets/ghopkins/nba-injuries-2010-2018 | Free (account) | Skipped 2026-11-17: no token provided; licence not checked, no data downloaded. | Skipped — no data included |
