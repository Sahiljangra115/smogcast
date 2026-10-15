"""Latency test against a running server. Usage: python scripts/loadtest.py http://127.0.0.1:8077 [n] [workers]
Writes reports/loadtest_w<workers>.json."""
import json
import os
import platform
from datetime import datetime, timezone
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

base = sys.argv[1].rstrip("/")
n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 1
sites = ["delhi_centre", "dwarka", "panipat", "meerut", "rewari"]  # the 5 distinct CAMS cells


def one(i):
    t = time.perf_counter()
    try:
        ok = requests.get(f"{base}/predict/latest", params={"site": sites[i % len(sites)]}, timeout=30).ok
    except requests.RequestException:
        ok = False
    return (time.perf_counter() - t) * 1000, ok


t0 = time.perf_counter()
with ThreadPoolExecutor(workers) as ex:
    res = list(ex.map(one, range(n)))
wall = time.perf_counter() - t0
lat = sorted(r[0] for r in res if r[1])
errors = sum(not r[1] for r in res)
q = lambda p: lat[int(p * (len(lat) - 1))] if lat else float("nan")  # noqa: E731
out = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "endpoint": "GET /predict/latest",
       "host": f"{platform.processor() or platform.machine()}, {os.cpu_count()} logical cpus", "n": n, "workers": workers, "errors": errors, "p50_ms": statistics.median(lat) if lat else None,
       "p95_ms": q(.95), "p99_ms": q(.99), "max_ms": lat[-1] if lat else None, "rps": n / wall,
       "loadavg_1m_after": round(os.getloadavg()[0], 2)}  # the dev machine is shared; this shows how busy it was
print(json.dumps(out, indent=2))
rep = Path(__file__).resolve().parent.parent / "reports"
rep.mkdir(exist_ok=True)
(rep / f"loadtest_w{workers}.json").write_text(json.dumps(out, indent=2))
