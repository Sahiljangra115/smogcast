"""Promotion gate CLI. Usage: python scripts/gate.py <candidate_version> [--promote | --canary]
Exit 1 if the candidate loses to CURRENT at either horizon on the frozen eval window (TEST_START onward).
Equal or better at both 24 h and 48 h passes, so a candidate identical to CURRENT passes.
--canary: on a pass, point models/CANARY at the candidate (serve.py sends CANARY_WEIGHT of traffic to it).
--promote: on a pass, make it CURRENT and update the MLflow champion alias."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smogcast import config as C, features as F, registry as R, retrain, train as T  # noqa: E402

if len(sys.argv) < 2 or not sys.argv[1].startswith("v"):
    sys.exit("usage: gate.py <candidate_version> [--promote | --canary]")
cand_v = sys.argv[1]
cur_v = R.current()
df = F.build_all()
_, test = T.split(df)
g = retrain.gate(R.load(cand_v), R.load(cur_v), test)
print(f"candidate {cand_v} vs current {cur_v}, frozen window {C.TEST_START[:10]} onward")
print(retrain.describe(g))
if not g["promote"]:
    print("BLOCKED (candidate worse at one or more horizons)")
    sys.exit(1)
print("PASS (equal or better at every horizon)")
if "--canary" in sys.argv:
    R.set_canary(cand_v)
    print("canary ->", cand_v)
elif "--promote" in sys.argv:
    R.promote(cand_v)
    print("promoted", cand_v)
    from smogcast import tracking  # noqa: E402
    tracking.sync()
