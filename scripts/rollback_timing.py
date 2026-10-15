"""Measure rollback end to end against a running server, then restore. Needs two registered versions.
Usage: python scripts/rollback_timing.py http://127.0.0.1:8077 [repeats]
Each repeat: swap CURRENT/PREVIOUS, POST /reload, confirm /health reports the new version, then swap back.
Writes reports/rollback_timing.json."""
import json
from datetime import datetime, timezone
import statistics
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smogcast import registry as R  # noqa: E402

base = sys.argv[1].rstrip("/")
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
times = []
for _ in range(reps):
    before = R.current()
    t0 = time.perf_counter()
    R.rollback()
    requests.post(base + "/reload", timeout=60).raise_for_status()
    served = requests.get(base + "/health", timeout=60).json()["model_version"]
    dt = time.perf_counter() - t0
    assert served == R.current() and served != before, f"server serves {served}, expected {R.current()}"
    times.append(dt)
    R.rollback()  # restore original CURRENT
    requests.post(base + "/reload", timeout=60).raise_for_status()
out = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "repeats": reps, "median_s": statistics.median(times), "max_s": max(times), "all_s": times}
print(json.dumps(out, indent=2))
(Path(__file__).resolve().parent.parent / "reports" / "rollback_timing.json").write_text(json.dumps(out, indent=2))
