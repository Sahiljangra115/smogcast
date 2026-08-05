"""Hourly ingestion with retry/backoff, chunked history, explicit missing-data handling.

Every fetch attempt is appended to data/ingest_log.csv, which is what the
ingestion-uptime number in the README is computed from.
"""
from __future__ import annotations

import csv
import sys
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

from . import config as C

CHUNK_DAYS = 120


def _get(url: str, params: dict, tries: int = 4, timeout: int = 60) -> dict:
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
        except (requests.RequestException, ValueError) as e:
            last = type(e).__name__
        time.sleep(2 ** i)
    raise RuntimeError(last)


def _log(source: str, site: str, ok: bool, rows: int, err: str = "") -> None:
    C.INGEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not C.INGEST_LOG.exists()
    with open(C.INGEST_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts_utc", "source", "site", "ok", "rows", "err"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), source, site, int(ok), rows, err])


def _frame(payload: dict) -> pd.DataFrame:
    h = payload["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def _chunks(start: date, end: date):
    s = start
    while s <= end:
        e = min(s + timedelta(days=CHUNK_DAYS - 1), end)
        yield s, e
        s = e + timedelta(days=1)


def _fetch_range(source: str, site: str, url: str, params: dict, start: date, end: date) -> pd.DataFrame:
    parts = []
    for s, e in _chunks(start, end):
        p = dict(params, start_date=s.isoformat(), end_date=e.isoformat())
        try:
            df = _frame(_get(url, p))
            parts.append(df)
            _log(source, site, True, len(df))
        except Exception as ex:  # outage: log, keep going, leave a gap (NaN) instead of crashing
            _log(source, site, False, 0, str(ex))
    return pd.concat(parts) if parts else pd.DataFrame()


def fetch_site(site: str, start: date, end: date) -> pd.DataFrame:
    lat, lon = C.SITES[site]
    base = {"latitude": lat, "longitude": lon, "timezone": "UTC"}
    aq = _fetch_range("aq", site, C.AQ_URL, dict(base, hourly="pm2_5,pm10"), start, end)
    wx = _fetch_range("wx_archive", site, C.WX_ARCHIVE_URL, dict(base, hourly=",".join(C.WX_VARS)), start, end)
    # Archive lacks boundary layer height for part of 2024; fill gaps from historical-forecast.
    wf = _fetch_range("wx_histfc", site, C.WX_HISTFC_URL, dict(base, hourly=",".join(C.WX_VARS)), start, end)
    if not wx.empty and not wf.empty:
        wx = wx.combine_first(wf)
    elif wx.empty:
        wx = wf
    df = aq.join(wx, how="outer") if not aq.empty else wx
    return df[~df.index.duplicated(keep="last")].sort_index()


def update_site(site: str, end: date | None = None) -> pd.DataFrame:
    """Incremental: re-fetch the last 7 days (data is revised) plus anything new."""
    end = end or date.today()
    path = C.RAW / f"{site}.csv"
    if path.exists():
        old = pd.read_csv(path, index_col=0, parse_dates=True)
        start = (old.index.max() - pd.Timedelta(days=7)).date()
        new = fetch_site(site, start, end)
        df = pd.concat([old, new])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    else:
        df = fetch_site(site, date.fromisoformat(C.START_DATE), end)
    # Open-Meteo returns the whole of today, so later hours are CAMS forecast. Keep only past hours.
    df = df[df.index <= pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")]
    C.RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def main() -> None:
    sites = sys.argv[1:] or list(C.SITES)
    for s in sites:
        df = update_site(s)
        print(f"{s}: {len(df)} rows, {df.index.min()} -> {df.index.max()}, pm2_5 null={df['pm2_5'].isna().mean():.3%}")


if __name__ == "__main__":
    main()
