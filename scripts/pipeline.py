"""One cron-able pipeline pass: ingest -> drift check -> (if drift) retrain candidate -> gate -> promote.

Same decision rule as the replay in smogcast/retrain.py, applied once at "now".
Usage: python scripts/pipeline.py [--no-collect] [--dry-run]
Prefect is not installed; this plain script is the orchestration. Schedule with deploy/.
Exit code 0 always unless a step crashes; the outcome is written to reports/pipeline_runs.jsonl.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smogcast import collect, config as C, drift as D, features as F, metrics as M  # noqa: E402
from smogcast import registry as R, retrain, tracking, train as T  # noqa: E402

args = set(sys.argv[1:])
out = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "steps": []}

if "--no-collect" not in args:
    for s in C.SITES:
        try:
            collect.update_site(s)
        except Exception as e:  # outage on one site must not stop the rest
            out["steps"].append({"collect_failed": s, "err": str(e)[:200]})
    out["steps"].append("collect_done")

df = F.build_all()
feats = F.feature_cols(df)
now = df["time"].max()
cur_win = df[df["time"] > now - pd.Timedelta(days=retrain.DRIFT_WINDOW_DAYS)]
ref = D.seasonal_reference(df, now, retrain.DRIFT_WINDOW_DAYS)
rep = D.check(ref, cur_win) if len(ref) > 500 else {"flags": [], "psi": {}}
current = R.load()
real = retrain._realised(df, now)
rw = real[real["time"] > now - pd.Timedelta(days=7)].dropna(subset=[f"y{retrain.H}"])
if len(rw) > 100 and D.error_drift(M.mae(rw[f"y{retrain.H}"], T.predict(current, rw, retrain.H)),
                                   M.mae(rw[f"y{retrain.H}"], rw["base_persist"])):
    rep["flags"].append("error_ratio")
drift = len(rep["flags"]) >= retrain.MIN_FLAGS or "error_ratio" in rep["flags"]
out.update(data_end=str(now), flags=rep["flags"], drift=drift, current=R.current())
C.REPORTS.mkdir(exist_ok=True)
# read by serve.py /metrics (Prometheus gauges smogcast_drift_*)
(C.REPORTS / "drift_latest.json").write_text(json.dumps(
    {"at": out["at"], "data_end": str(now), "psi": rep["psi"], "ks_pm25": rep.get("ks_pm25"), "flags": rep["flags"], "drift": drift}, indent=2))

if drift and "--dry-run" not in args:
    gate_start = now - pd.Timedelta(days=retrain.GATE_DAYS)
    cand = T.fit(retrain._realised(df[df["time"] < gate_start], gate_start, 48), feats)
    g = retrain.gate(cand, current, real[real["time"] >= gate_start])
    out["gate"] = g
    if g["promote"]:
        final = T.fit(retrain._realised(df, now, 48), feats)
        v = R.register(final, {"trained_at": out["at"], "reason": rep["flags"], "gate": g})
        R.promote(v)
        out["promoted"] = v
tracking.log_run("pipeline", {"dry_run": "--dry-run" in args},
                 {"drift": int(drift), "n_flags": len(rep["flags"]), **{f"psi_{k}": v for k, v in rep["psi"].items()}},
                 {"flags": rep["flags"]}, model_dir=C.MODELS / out["promoted"] if "promoted" in out else None)
if "promoted" in out:
    tracking.sync()
with (C.REPORTS / "pipeline_runs.jsonl").open("a") as f:
    f.write(json.dumps(out, default=str) + "\n")
print(json.dumps(out, indent=2, default=str))
