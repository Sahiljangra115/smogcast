"""Render the existing drift.py output as a static HTML report (stand-in for Evidently, not installed).

Usage: python scripts/drift_report.py [out.html]
Sections: current 28-day window vs seasonal reference (PSI per covariate, KS on PM2.5),
and the drift events from reports/replay.json if present.
"""
import html
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smogcast import config as C, drift as D, features as F  # noqa: E402

out = Path(sys.argv[1]) if len(sys.argv) > 1 else C.REPORTS / "drift_report.html"
df = F.build_all()
now = df["time"].max()
cur = df[df["time"] > now - pd.Timedelta(days=28)]
ref = D.seasonal_reference(df, now, 28)
rep = D.check(ref, cur)
rows = [(c, v, D.DRIFT_RULES["psi"]) for c, v in rep["psi"].items()]
if "ks_pm25" in rep:
    rows.append(("KS D on pm2_5", rep["ks_pm25"], D.DRIFT_RULES["ks_d"]))


def tr(name, val, thr):
    bad = val == val and val > thr
    style = ' style="background:#d33a;font-weight:600"' if bad else ""
    return f"<tr{style}><td>{html.escape(name)}</td><td>{val:.3f}</td><td>{thr}</td><td>{'DRIFT' if bad else 'ok'}</td></tr>"


# Known calendar periods. An event at date D looks at the trailing 28-day window (D-28, D]; a period "overlaps" if it
# intersects that window. Stubble season and monsoon onset are approximate seasonal windows, not measured dates.
# No fire data is used, so an overlap is a coincidence in time, not evidence that fires caused the flag.
CALENDAR = [("Diwali 2024", "2024-11-01", "2024-11-01"), ("Diwali 2025", "2025-10-20", "2025-10-20"),
            ("stubble season 2024", "2024-10-01", "2024-11-30"), ("stubble season 2025", "2025-10-01", "2025-11-30"),
            ("stubble season 2026", "2026-10-01", "2026-11-30"),
            ("monsoon onset 2024", "2024-06-20", "2024-07-05"), ("monsoon onset 2025", "2025-06-20", "2025-07-05"), ("monsoon onset 2026", "2026-06-20", "2026-07-05")]


def calendar_tags(date: str, window_days: int = 28) -> list[str]:
    d1 = pd.Timestamp(date)
    d0 = d1 - pd.Timedelta(days=window_days)
    return [n for n, a, b in CALENDAR if pd.Timestamp(a) <= d1 and pd.Timestamp(b) > d0]


ev_html = ""
rp = C.REPORTS / "replay.json"
if rp.exists():
    ev = json.loads(rp.read_text())["drift_events"]
    for e in ev:
        e["calendar"] = calendar_tags(e["date"])
    (C.REPORTS / "drift_calendar.json").write_text(json.dumps(
        [{k: e[k] for k in ("date", "flags", "retrained", "promoted", "calendar")} for e in ev], indent=2))
    # chance rate: share of ALL weekly decision dates whose trailing window overlaps each kind of period
    rj = json.loads(rp.read_text())
    end = pd.Timestamp(rj["data_end"])
    dates = [min(pd.Timestamp(w["week"]) + pd.Timedelta(days=7), end) for w in rj["weekly"]]
    kinds = {"stubble season": "stubble", "Diwali": "Diwali", "monsoon onset": "monsoon"}
    summ = {"n_events": len(ev), "n_decision_dates": len(dates), "periods": {}}
    for label, key in kinds.items():
        k_ev = sum(any(key in t for t in e["calendar"]) for e in ev)
        k_all = sum(any(key in t for t in calendar_tags(str(d.date()))) for d in dates)
        summ["periods"][label] = {"events": k_ev, "share_of_events": k_ev / len(ev) if ev else None,
                                  "share_of_all_decision_dates": k_all / len(dates)}
    n_none = sum(not e["calendar"] for e in ev)
    summ["periods"]["none"] = {"events": n_none, "share_of_events": n_none / len(ev) if ev else None,
                               "share_of_all_decision_dates": sum(not calendar_tags(str(d.date())) for d in dates) / len(dates)}
    summ["october_events"] = [e["date"] for e in ev if e["date"][5:7] == "10"]
    (C.REPORTS / "drift_calendar_summary.json").write_text(json.dumps(summ, indent=2))
    ev_html = ("<h2>Replay drift events</h2><p>Calendar column: known periods that overlap the trailing 28-day window. "
               "Overlap in time only; no fire data is used.</p><table><tr><th>date</th><th>flags</th><th>retrained</th><th>promoted</th><th>calendar</th></tr>") + "".join(
        f"<tr><td>{e['date']}</td><td>{html.escape(', '.join(e['flags']))}</td><td>{e['retrained']}</td><td>{e['promoted']}</td>"
        f"<td>{html.escape(', '.join(e['calendar']))}</td></tr>" for e in ev) + "</table>"
out.write_text(f"""<!doctype html><meta charset="utf-8"><title>Smogcast drift report</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>:root{{color-scheme:light dark}}body{{font:14px system-ui,sans-serif;max-width:900px;margin:2em auto;padding:0 16px}}td,th{{border:1px solid #8884;padding:4px 8px}}table{{border-collapse:collapse;display:block;overflow-x:auto}}</style>
<h1>Smogcast drift report</h1>
<p>Window: 28 days to {now}. Reference: {len(ref)} rows of seasonally matched history. Data is CAMS model output, not CPCB stations.</p>
<p>Flags: {html.escape(', '.join(rep['flags']) or 'none')}</p>
<table><tr><th>metric</th><th>value</th><th>threshold</th><th>status</th></tr>{''.join(tr(*r) for r in rows)}</table>{ev_html}""")
print("wrote", out)
