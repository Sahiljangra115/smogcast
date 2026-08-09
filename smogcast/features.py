"""Feature building. All features use only information available at forecast time t.

Target for horizon h is pm2_5 at t+h at the same site. Weather at t+h is deliberately
NOT used (it would need a weather forecast and would leak if taken from reanalysis).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

LAGS = (1, 2, 3, 6, 12, 24, 48, 72, 168)
WX = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "boundary_layer_height", "surface_pressure"]
SITE_IDS = C.SITE_IDS


def load_site(site: str) -> pd.DataFrame:
    df = pd.read_csv(C.RAW / f"{site}.csv", index_col=0, parse_dates=True)
    return df.asfreq("h")  # explicit hourly grid: missing hours become NaN rows


def build_site(df: pd.DataFrame, site: str, horizons=C.HORIZONS, with_target: bool = True) -> pd.DataFrame:
    pm = df["pm2_5"]
    out = pd.DataFrame(index=df.index)
    out["pm2_5"] = pm
    for l in LAGS:
        out[f"pm_lag{l}"] = pm.shift(l)
    for w in (6, 24, 72):
        out[f"pm_mean{w}"] = pm.rolling(w, min_periods=max(2, w // 2)).mean()
    out["pm_std24"] = pm.rolling(24, min_periods=12).std()
    out["pm_trend6"] = pm - pm.shift(6)
    out["pm10"] = df["pm10"]
    for c in WX:
        out[c] = df[c]
    out["wind_speed_mean24"] = df["wind_speed_10m"].rolling(24, min_periods=12).mean()
    out["blh_mean24"] = df["boundary_layer_height"].rolling(24, min_periods=12).mean()
    rad = np.deg2rad(df["wind_direction_10m"])
    out["wind_u"] = -df["wind_speed_10m"] * np.sin(rad)
    out["wind_v"] = -df["wind_speed_10m"] * np.cos(rad)
    # north-westerly flow (smoke corridor from Punjab/Haryana): direction between 270 and 360 deg
    wd = df["wind_direction_10m"]
    out["nw_flow"] = ((wd >= 270) & (wd <= 360)).astype(float).where(wd.notna())
    out["nw_frac24"] = out["nw_flow"].rolling(24, min_periods=12).mean()
    hr, doy, dow = out.index.hour, out.index.dayofyear, out.index.dayofweek
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    out["doy_sin"], out["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    out["dow"] = dow
    out["site_id"] = SITE_IDS[site]
    if with_target:
        for h in horizons:
            out[f"y{h}"] = pm.shift(-h)
    out["base_persist"] = pm
    for h in horizons:
        out[f"base_snaive{h}"] = pm.shift(168 - h)  # same hour, one week before the target hour
    out["site"] = site
    return out


def build_all(sites=None, with_target: bool = True) -> pd.DataFrame:
    frames = [build_site(load_site(s), s, with_target=with_target) for s in (sites or C.SITES)]
    return pd.concat(frames).rename_axis("time").reset_index()


def feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {"time", "site", "pm2_5"} | {c for c in df.columns if (c.startswith("y") and c[1:].isdigit()) or c.startswith("base_")}
    return [c for c in df.columns if c not in drop]
