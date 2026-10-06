"""Drift-triggered retraining with a promotion gate, plus a walk-forward replay.

gate(): candidate passes if its MAE is lower than or equal to the current model's on the evaluation
window, at every horizon gated (so an identical candidate passes, a worse one is blocked). The CI job
runs this same function, see .github/workflows/gate.yml.

replay(): walks through history week by week. At each step the CURRENT model forecasts
that week; drift is checked on data available at the step; a flagged drift (after a
cool-down) retrains a candidate, the candidate must beat the current model on the last
28 realised days, and only then is it promoted (refit on all data realised so far).
Compares four policies on identical weeks: static, drift-triggered, blind periodic refit (control), persistence.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from . import drift as D
from . import features as F
from . import metrics as M
from . import train as T
from . import tracking

GATE_DAYS = 28
COOLDOWN_DAYS = 28
DRIFT_WINDOW_DAYS = 28
MIN_FLAGS = 2  # one noisy covariate alone does not trigger; the error flag always does
H = 24


def gate(candidate: dict, current: dict, eval_df: pd.DataFrame, horizons=C.HORIZONS) -> dict:
    """Pass if the candidate MAE <= current MAE at every horizon. MAE is reported per horizon
    (`per_horizon`); `candidate_mae`/`current_mae` are the plain mean over the gated horizons."""
    per = {}
    for h in horizons:
        d = eval_df.dropna(subset=[f"y{h}"])
        per[f"h{h}"] = {"candidate_mae": M.mae(d[f"y{h}"], T.predict(candidate, d, h)),
                        "current_mae": M.mae(d[f"y{h}"], T.predict(current, d, h))}
    cand = float(np.mean([v["candidate_mae"] for v in per.values()]))
    cur = float(np.mean([v["current_mae"] for v in per.values()]))
    ok = all(v["candidate_mae"] <= v["current_mae"] for v in per.values())  # equal or better passes
    return {"candidate_mae": cand, "current_mae": cur, "per_horizon": per, "promote": ok}


def describe(g: dict) -> str:
    """One labelled line per horizon, e.g. 'h24 MAE: candidate 27.10 vs current 27.12 (ug/m3)'."""
    return "\n".join(f"{h} MAE: candidate {v['candidate_mae']:.3f} vs current {v['current_mae']:.3f} ug/m3"
                     for h, v in g["per_horizon"].items())


def _realised(df: pd.DataFrame, upto: pd.Timestamp, h: int = H) -> pd.DataFrame:
    """Rows whose h-hour target was already observed at time `upto`."""
    return df[df["time"] <= upto - pd.Timedelta(hours=h)]


def replay(start="2024-07-01", end=None, init_train_end="2024-06-30", step_days=7, verbose=True) -> dict:
    df = F.build_all()
    feats = F.feature_cols(df)
    end = pd.Timestamp(end) if end else df["time"].max()
    t0 = pd.Timestamp(start)
    init_end = pd.Timestamp(init_train_end) + pd.Timedelta(hours=23)
    static = T.fit(_realised(df, init_end, 48), feats, (H,))
    current = static
    periodic, last_periodic = static, init_end
    last_retrain = t0 - pd.Timedelta(days=COOLDOWN_DAYS)
    events, rows = [], []
    while t0 < end:
        t1 = min(t0 + pd.Timedelta(days=step_days), end)
        week = df[(df["time"] >= t0) & (df["time"] < t1)].dropna(subset=[f"y{H}", "base_persist"])
        if len(week):
            rows.append({"week": t0, "n": len(week),
                         "static": M.mae(week[f"y{H}"], T.predict(static, week, H)),
                         "triggered": M.mae(week[f"y{H}"], T.predict(current, week, H)),
                         "periodic": M.mae(week[f"y{H}"], T.predict(periodic, week, H)),
                         "persistence": M.mae(week[f"y{H}"], week["base_persist"])})
        if (t1 - last_periodic).days >= COOLDOWN_DAYS:  # control: blind refit every 28 days, no drift, no gate
            periodic, last_periodic = T.fit(_realised(df, t1, 48), feats, (H,)), t1
        # decision point at t1 using only data observed up to t1
        cur_win = df[(df["time"] > t1 - pd.Timedelta(days=DRIFT_WINDOW_DAYS)) & (df["time"] <= t1)]
        ref = D.seasonal_reference(df, t1, DRIFT_WINDOW_DAYS)
        rep = D.check(ref, cur_win) if len(ref) > 500 else {"flags": [], "drift": False, "psi": {}}
        real = _realised(df, t1)
        rw = real[real["time"] > t1 - pd.Timedelta(days=7)].dropna(subset=[f"y{H}"])
        err_flag = len(rw) > 100 and D.error_drift(M.mae(rw[f"y{H}"], T.predict(current, rw, H)), M.mae(rw[f"y{H}"], rw["base_persist"]))
        if err_flag:
            rep["flags"].append("error_ratio")
        rep["drift"] = len(rep["flags"]) >= MIN_FLAGS or "error_ratio" in rep["flags"]
        ev = {"date": str(t1.date()), "flags": rep["flags"], "retrained": False, "promoted": False}
        if rep["drift"]:
            events.append(ev)
        if rep["drift"] and (t1 - last_retrain).days >= COOLDOWN_DAYS:
            gate_start = t1 - pd.Timedelta(days=GATE_DAYS)
            pre = _realised(df[df["time"] < gate_start], gate_start, 48)
            gwin = real[real["time"] >= gate_start]
            cand = T.fit(pre, feats, (H,))
            g = gate(cand, current, gwin, (H,))
            ev.update(retrained=True, gate=g)
            last_retrain = t1
            if g["promote"]:
                current = T.fit(_realised(df, t1, 48), feats, (H,))
                ev["promoted"] = True
            if verbose:
                print(ev["date"], ev["flags"], "gate h24 MAE candidate", round(g["candidate_mae"], 2), "vs current", round(g["current_mae"], 2), "promote", g["promote"])
        t0 = t1
    w = pd.DataFrame(rows)
    wt = w["n"]
    summary = {k: float(np.average(w[k], weights=wt)) for k in ("static", "triggered", "periodic", "persistence")}
    return {"summary_mae_h24": summary, "weeks": len(w), "data_end": str(end), "drift_events": events,
            "retrains": sum(e["retrained"] for e in events), "promotions": sum(e["promoted"] for e in events),
            "weekly": w.assign(week=w["week"].astype(str)).round(3).to_dict("records")}


def main() -> None:
    r = replay()
    C.REPORTS.mkdir(exist_ok=True)
    tracking.log_run("replay", {"step_days": 7, "cooldown_days": COOLDOWN_DAYS, "min_flags": MIN_FLAGS},
                     {**r["summary_mae_h24"], "retrains": r["retrains"], "promotions": r["promotions"]})
    (C.REPORTS / "replay.json").write_text(json.dumps(r, indent=2, default=str))
    print(json.dumps({k: v for k, v in r.items() if k != "weekly"}, indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()
