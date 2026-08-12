"""File-based model registry: versioned dirs, CURRENT/PREVIOUS pointers, a CANARY pointer, rollback.

This is the source of truth the server reads. smogcast/tracking.py mirrors versions into MLflow.
CANARY: a version that serve.py sends a CANARY_WEIGHT fraction of traffic to (see serve.py).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib

from . import config as C


def _dir(root: Path | None) -> Path:
    d = root or C.MODELS
    d.mkdir(parents=True, exist_ok=True)
    return d


def versions(root: Path | None = None) -> list[str]:
    return sorted(p.name for p in _dir(root).iterdir() if p.is_dir() and p.name.startswith("v"))


def register(bundle: dict, meta: dict, root: Path | None = None) -> str:
    d = _dir(root)
    vs = versions(root)
    v = f"v{int(vs[-1][1:]) + 1:03d}" if vs else "v001"
    (d / v).mkdir()
    joblib.dump(bundle, d / v / "model.joblib")
    (d / v / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return v


def _ptr(name: str, root: Path | None) -> Path:
    return _dir(root) / name


def current(root: Path | None = None) -> str | None:
    p = _ptr("CURRENT", root)
    return p.read_text().strip() if p.exists() else None


def promote(v: str, root: Path | None = None) -> None:
    cur = current(root)
    if cur and cur != v:
        _ptr("PREVIOUS", root).write_text(cur)
    _ptr("CURRENT", root).write_text(v)
    if canary(root) == v:
        set_canary(None, root)


def canary(root: Path | None = None) -> str | None:
    p = _ptr("CANARY", root)
    return (p.read_text().strip() or None) if p.exists() else None


def set_canary(v: str | None, root: Path | None = None) -> None:
    """Point CANARY at a registered version, or clear it with None."""
    p = _ptr("CANARY", root)
    if v is None:
        p.unlink(missing_ok=True)
        return
    if v not in versions(root):
        raise RuntimeError(f"unknown version {v}")
    p.write_text(v)


def rollback(root: Path | None = None) -> float:
    """Swap CURRENT back to PREVIOUS. Returns seconds taken (file level; excludes server reload)."""
    t0 = time.perf_counter()
    prev = _ptr("PREVIOUS", root)
    if not prev.exists():
        raise RuntimeError("no previous version to roll back to")
    cur = current(root)
    _ptr("CURRENT", root).write_text(prev.read_text().strip())
    prev.write_text(cur or "")
    return time.perf_counter() - t0


def load(v: str | None = None, root: Path | None = None) -> dict:
    v = v or current(root)
    if not v:
        raise RuntimeError("no model promoted")
    b = joblib.load(_dir(root) / v / "model.joblib")
    b["version"] = v
    return b


def meta(v: str, root: Path | None = None) -> dict:
    return json.loads((_dir(root) / v / "meta.json").read_text())
