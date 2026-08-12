"""Roll CURRENT back to the previous version and reload a running server. Prints elapsed seconds."""
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smogcast import registry as R  # noqa: E402

url = sys.argv[1] if len(sys.argv) > 1 else None
t0 = time.perf_counter()
R.rollback()
if url:
    requests.post(url.rstrip("/") + "/reload", timeout=30).raise_for_status()
print(f"rolled back to {R.current()} in {time.perf_counter() - t0:.3f}s" + (" (including server reload)" if url else " (registry swap only)"))
try:  # keep the MLflow champion alias in step; never let tracking break a rollback
    from smogcast import tracking  # noqa: E402
    tracking.sync()
except (Exception, SystemExit) as e:
    print("mlflow alias not updated:", e)
