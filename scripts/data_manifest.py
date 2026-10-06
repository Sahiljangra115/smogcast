"""Data versioning fallback (DVC not installed): sha256 + size + row count of every raw CSV.
Writes data/MANIFEST.json. Commit the manifest, not the data. Compare manifests to see what changed.
For real DVC use see dvc.yaml and README."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smogcast import config as C  # noqa: E402

m = {}
for p in sorted(C.RAW.glob("*.csv")):
    b = p.read_bytes()
    m[p.name] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b), "rows": b.count(b"\n") - 1}
(C.ROOT / "data" / "MANIFEST.json").write_text(json.dumps(m, indent=2))
print(f"{len(m)} files hashed")
