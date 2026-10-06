"""Drift checks: PSI on covariates, KS on the PM2.5 distribution, and realised-error drift.

Reference = history from the same time of year (+-21 days, earlier years or earlier in the
record), because comparing a monsoon fortnight against an all-year reference flags drift
every single week (the first replay did exactly that, see README). Current = trailing 28 days.
Thresholds are stated in DRIFT_RULES and are conventions (PSI 0.25 is the usual
"significant shift" rule of thumb), not tuned to make the numbers look good.
Fire counts (NASA FIRMS) are NOT monitored: FIRMS needs a MAP_KEY that was not available.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PSI_COLS = ["wind_speed_10m", "nw_frac24", "boundary_layer_height", "temperature_2m"]
DRIFT_RULES = {"psi": 0.25, "ks_d": 0.30, "error_ratio": 1.25}


def psi(ref, cur, bins: int = 10, eps: float = 1e-4) -> float:
    ref = np.asarray(ref, float)
    cur = np.asarray(cur, float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) < 50 or len(cur) < 50:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + eps
    c = np.histogram(cur, edges)[0] / len(cur) + eps
    return float(np.sum((c - r) * np.log(c / r)))


def check(ref: pd.DataFrame, cur: pd.DataFrame, rules: dict = DRIFT_RULES) -> dict:
    out = {"psi": {}, "flags": []}
    for c in PSI_COLS:
        v = psi(ref[c], cur[c])
        out["psi"][c] = v
        if v > rules["psi"]:
            out["flags"].append(f"psi:{c}")
    a, b = ref["pm2_5"].dropna(), cur["pm2_5"].dropna()
    if len(a) > 50 and len(b) > 50:
        d = float(ks_2samp(a, b).statistic)
        out["ks_pm25"] = d
        if d > rules["ks_d"]:
            out["flags"].append("ks:pm2_5")
    out["drift"] = bool(out["flags"])
    return out


def seasonal_reference(df: pd.DataFrame, window_end: pd.Timestamp, window_days: int = 28, band: int = 21) -> pd.DataFrame:
    """History before the current window whose day-of-year is within +-band days of the window centre."""
    start = window_end - pd.Timedelta(days=window_days)
    hist = df[df["time"] <= start]
    centre = (window_end - pd.Timedelta(days=window_days // 2)).dayofyear
    dd = (hist["time"].dt.dayofyear - centre).abs()
    dd = np.minimum(dd, 365 - dd)
    return hist[dd <= band]


def error_drift(model_mae: float, persistence_mae: float, rules: dict = DRIFT_RULES) -> bool:
    """Scale-free: the model is failing if it is clearly worse than the naive baseline it should beat."""
    return bool(model_mae > rules["error_ratio"] * persistence_mae)
