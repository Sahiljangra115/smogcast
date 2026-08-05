"""Key-gated sources from the plan: OpenAQ v3 (CPCB station PM2.5) and NASA FIRMS (VIIRS fire counts).

Neither is used by the current model: no key was available, so these paths have never fetched real data.
They are ready for when the keys exist and fail with a clear message when they do not.

    OPENAQ_API_KEY   register at https://explore.openaq.org/register, key is under account settings
    FIRMS_MAP_KEY    request at https://firms.modaps.eosdis.nasa.gov/api/map_key/

    python -m smogcast.external openaq [site]       # list PM2.5 stations within 25 km of a site
    python -m smogcast.external firms 2025-11-01    # daily VIIRS fire counts, Punjab+Haryana box, 5 days to that date

Plan note: use FIRMS science-quality data (VIIRS_SNPP_SP) for training and NRT (VIIRS_SNPP_NRT) for inference;
NRT counts run high (CSE), so the two should not be mixed in one feature without a check.
"""
from __future__ import annotations

import io
import os
import sys

import pandas as pd
import requests

from . import config as C

OPENAQ_URL = "https://api.openaq.org/v3"
FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
PUNJAB_HARYANA_BBOX = (73.8, 27.6, 77.6, 32.6)  # west, south, east, north
PM25_PARAMETER_ID = 2  # OpenAQ parameter id for pm25


class MissingKey(RuntimeError):
    pass


def _key(var: str, url: str) -> str:
    k = os.environ.get(var, "").strip()
    if not k:
        raise MissingKey(f"{var} is not set. Get one at {url} and run: export {var}=<your key>")
    return k


def openaq_locations(lat: float, lon: float, radius_m: int = 25000) -> pd.DataFrame:
    """PM2.5 monitoring locations near a point, with their pm25 sensor ids."""
    key = _key("OPENAQ_API_KEY", "https://explore.openaq.org/register")
    r = requests.get(f"{OPENAQ_URL}/locations", headers={"X-API-Key": key}, timeout=60,
                     params={"coordinates": f"{lat},{lon}", "radius": radius_m, "parameters_id": PM25_PARAMETER_ID, "limit": 100})
    r.raise_for_status()
    rows = []
    for loc in r.json()["results"]:
        for s in loc.get("sensors", []):
            if s.get("parameter", {}).get("id") == PM25_PARAMETER_ID:
                rows.append({"location_id": loc["id"], "name": loc.get("name"), "provider": (loc.get("provider") or {}).get("name"),
                             "sensor_id": s["id"], "lat": loc["coordinates"]["latitude"], "lon": loc["coordinates"]["longitude"]})
    return pd.DataFrame(rows)


def openaq_hours(sensor_id: int, start: str, end: str) -> pd.Series:
    """Hourly PM2.5 for one sensor, UTC index. Missing hours stay missing (no fill)."""
    key = _key("OPENAQ_API_KEY", "https://explore.openaq.org/register")
    out, page = [], 1
    while True:
        r = requests.get(f"{OPENAQ_URL}/sensors/{sensor_id}/hours", headers={"X-API-Key": key}, timeout=60,
                         params={"datetime_from": start, "datetime_to": end, "limit": 1000, "page": page})
        r.raise_for_status()
        res = r.json()["results"]
        out += [(x["period"]["datetimeFrom"]["utc"], x["value"]) for x in res]
        if len(res) < 1000:
            break
        page += 1
    s = pd.Series({pd.Timestamp(t).tz_convert(None): v for t, v in out}, name="pm2_5", dtype=float)
    return s.sort_index()


def firms_daily_counts(end_date: str, days: int = 5, source: str = "VIIRS_SNPP_SP",
                       bbox: tuple = PUNJAB_HARYANA_BBOX) -> pd.Series:
    """Fire detections per day in the bbox for `days` days ending on end_date."""
    key = _key("FIRMS_MAP_KEY", "https://firms.modaps.eosdis.nasa.gov/api/map_key/")
    area = ",".join(str(x) for x in bbox)
    start = (pd.Timestamp(end_date) - pd.Timedelta(days=days - 1)).date().isoformat()
    r = requests.get(f"{FIRMS_URL}/{key}/{source}/{area}/{days}/{start}", timeout=120)
    r.raise_for_status()
    if not r.text.lstrip().startswith("latitude"):
        raise RuntimeError(f"FIRMS returned no CSV: {r.text[:200]}")  # e.g. an invalid key message
    df = pd.read_csv(io.StringIO(r.text))
    return df.groupby("acq_date").size().rename("fire_count")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        if cmd == "openaq":
            site = sys.argv[2] if len(sys.argv) > 2 else "delhi_centre"
            print(openaq_locations(*C.SITES[site]).to_string())
        elif cmd == "firms":
            print(firms_daily_counts(sys.argv[2] if len(sys.argv) > 2 else pd.Timestamp.utcnow().date().isoformat()).to_string())
        else:
            sys.exit("usage: python -m smogcast.external openaq [site] | firms [YYYY-MM-DD]")
    except MissingKey as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
