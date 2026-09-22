"""FastAPI serving. Loads the CURRENT registry model; /reload picks up promote, rollback and canary changes.

GET  /                         static forecast page (smogcast/static/index.html)
POST /predict  {"site": "...", "history": [{"time":..., "pm2_5":..., weather...}, ...]}
   history = at least the last 169 hourly rows ending at forecast time (the lag window).
GET  /predict/latest?site=...  builds the request from the newest rows in data/raw.
GET  /predict/all              latest forecast for every site plus persistence and backtest MAE (used by /).
GET  /health, POST /reload
GET  /metrics                  Prometheus exposition: latency histogram, prediction counts, model version, drift PSI.

Canary: if models/CANARY names a version and CANARY_WEIGHT (env, 0..1, default 0) is > 0, that fraction
of forecasts is served by the canary model. Every response says which version and variant served it.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

from . import config as C
from . import features as F
from . import registry as R
from . import train as T

app = FastAPI(title="smogcast")
_state = {"bundle": None, "canary": None, "lock": threading.Lock()}
_raw_cache: dict[str, tuple[float, pd.DataFrame]] = {}
STATIC = Path(__file__).resolve().parent / "static"

REG = CollectorRegistry()
LATENCY = Histogram("smogcast_request_latency_seconds", "Request latency", ["endpoint"], registry=REG,
                    buckets=(.002, .005, .01, .02, .03, .05, .075, .1, .2, .5, 1, 2))
PREDICTIONS = Counter("smogcast_predictions_total", "Forecasts served", ["model_version", "variant"], registry=REG)
ERRORS = Counter("smogcast_errors_total", "Requests that returned 4xx/5xx", ["endpoint"], registry=REG)
MODEL = Gauge("smogcast_model_info", "Loaded model versions (value 1)", ["model_version", "variant"], registry=REG)
CANARY_W = Gauge("smogcast_canary_weight", "Fraction of traffic sent to the canary", registry=REG)
PSI = Gauge("smogcast_drift_psi", "PSI of the last pipeline drift check", ["feature"], registry=REG)
KS = Gauge("smogcast_drift_ks_pm25", "KS D on PM2.5 at the last pipeline drift check", registry=REG)
DRIFT = Gauge("smogcast_drift_flagged", "1 if the last pipeline drift check flagged drift", registry=REG)
DRIFT_LATEST = C.REPORTS / "drift_latest.json"

RAW_COLS = ["pm2_5", "pm10", "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
            "wind_direction_10m", "boundary_layer_height", "surface_pressure"]
MIN_HISTORY = 169


class Row(BaseModel):
    time: str
    pm2_5: float | None = None
    pm10: float | None = None
    temperature_2m: float | None = None
    relative_humidity_2m: float | None = None
    wind_speed_10m: float | None = None
    wind_direction_10m: float | None = None
    boundary_layer_height: float | None = None
    surface_pressure: float | None = None


class Req(BaseModel):
    site: str
    history: list[Row]


def canary_weight() -> float:
    try:
        return min(max(float(os.environ.get("CANARY_WEIGHT", "0")), 0.0), 1.0)
    except ValueError:
        return 0.0


def _load() -> None:
    """Load CURRENT (and CANARY if set) into memory. Caller holds the lock."""
    _state["bundle"] = R.load()
    cv = R.canary()
    _state["canary"] = R.load(cv) if cv and cv != _state["bundle"]["version"] else None
    MODEL.clear()
    MODEL.labels(_state["bundle"]["version"], "current").set(1)
    if _state["canary"]:
        MODEL.labels(cv, "canary").set(1)


def _bundle() -> dict:
    with _state["lock"]:
        if _state["bundle"] is None:
            _load()
        return _state["bundle"]


def pick() -> tuple[dict, str]:
    """Weighted split: CANARY_WEIGHT of requests go to the canary model, the rest to CURRENT."""
    cur = _bundle()
    can = _state["canary"]
    if can is not None and random.random() < canary_weight():
        return can, "canary"
    return cur, "current"


def forecast(bundle: dict, site: str, rows: pd.DataFrame, variant: str = "current") -> dict:
    if site not in C.SITES:
        raise HTTPException(404, f"unknown site {site}")
    if len(rows) < MIN_HISTORY:
        raise HTTPException(422, f"need >= {MIN_HISTORY} hourly rows, got {len(rows)}")
    df = rows.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index().asfreq("h")
    for c in RAW_COLS:
        if c not in df:
            df[c] = np.nan
    feats = F.build_site(df[RAW_COLS], site, with_target=False).iloc[[-1]]
    if np.isnan(feats["pm_lag1"].iloc[0]) and np.isnan(feats["pm_lag24"].iloc[0]):
        raise HTTPException(422, "no recent pm2_5 observations")
    out = {"site": site, "issued_for": str(feats.index[0]), "model_version": bundle["version"], "variant": variant,
           "pm2_5_now": None if np.isnan(feats["pm2_5"].iloc[0]) else round(float(feats["pm2_5"].iloc[0]), 2),
           "forecast": {}}
    for h in bundle["horizons"]:
        out["forecast"][f"{h}h"] = {"time": str(feats.index[0] + pd.Timedelta(hours=h)),
                                     "pm2_5": round(float(T.predict(bundle, feats, h)[0]), 2)}
    PREDICTIONS.labels(bundle["version"], variant).inc()
    return out


def _recent(site: str) -> pd.DataFrame:
    """Newest rows of data/raw/<site>.csv, cached until the file changes (the collector rewrites it)."""
    p = C.RAW / f"{site}.csv"
    mt = p.stat().st_mtime
    hit = _raw_cache.get(site)
    if hit is None or hit[0] != mt:
        rows = pd.read_csv(p, index_col=0, parse_dates=True).tail(MIN_HISTORY + 24)
        hit = (mt, rows.reset_index(names="time").astype({"time": str}))
        _raw_cache[site] = hit
    return hit[1]


@app.middleware("http")
async def _timing(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    path = request.scope.get("route").path if request.scope.get("route") else "other"
    LATENCY.labels(path).observe(time.perf_counter() - t0)
    if resp.status_code >= 400:
        ERRORS.labels(path).inc()
    return resp


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/predict")
def predict(req: Req):
    rows = pd.DataFrame([r.model_dump() for r in req.history])
    b, variant = pick()
    return forecast(b, req.site, rows, variant)


@app.get("/predict/latest")
def predict_latest(site: str):
    if site not in C.SITES:
        raise HTTPException(404, f"unknown site {site}")
    b, variant = pick()
    return forecast(b, site, _recent(site), variant)


@app.get("/predict/all")
def predict_all():
    """Every site's latest forecast (CURRENT model only), persistence, and its frozen-test-period MAE."""
    b = _bundle()
    mp = C.REPORTS / "metrics.json"
    m = json.loads(mp.read_text()) if mp.exists() else {}
    backtest = {}
    if m.get("version") == b["version"]:  # only show a backtest that belongs to the served model
        for h, r in m["eval"].items():
            for s, v in r["per_site"].items():
                backtest.setdefault(s, {})[h] = {"model_mae": round(v["model"]["mae"], 1),
                                                 "persistence_mae": round(v["persistence"]["mae"], 1)}
    sites, cells = [], {}
    for s in C.SITES:
        rows = _recent(s)
        f = forecast(b, s, rows)
        key = tuple(rows["pm2_5"].tail(168).round(3).fillna(-1))  # same CAMS cell -> identical series
        f["cell"] = cells.setdefault(key, s)
        f["backtest"] = backtest.get(s)
        f["also_covers"] = C.CELL_MEMBERS.get(s, [])  # named points in the same CAMS cell (identical PM2.5)
        sites.append(f)
    return {"model_version": b["version"], "canary": R.canary(), "canary_weight": canary_weight(),
            "backtest_period": f"{C.TEST_START[:10]} to {m.get('data_end', '?')[:10]}" if backtest else None,
            "sites": sites}


@app.get("/health")
def health():
    return {"status": "ok", "model_version": _bundle()["version"],
            "canary": _state["canary"]["version"] if _state["canary"] else None, "canary_weight": canary_weight()}


@app.post("/reload")
def reload():
    with _state["lock"]:
        _load()
    return {"model_version": _state["bundle"]["version"]}


@app.get("/metrics")
def metrics():
    CANARY_W.set(canary_weight())
    if DRIFT_LATEST.exists():
        d = json.loads(DRIFT_LATEST.read_text())
        for k, v in d.get("psi", {}).items():
            if v == v:
                PSI.labels(k).set(v)
        if d.get("ks_pm25") is not None:
            KS.set(d["ks_pm25"])
        DRIFT.set(1 if d.get("drift") else 0)
    return Response(generate_latest(REG), media_type=CONTENT_TYPE_LATEST)
