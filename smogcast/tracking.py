"""Experiment tracking: reports/runs.jsonl always, plus MLflow when it is importable.

JSONL: one line per run (name, UTC time, params, metrics, artifacts). Read with `python -m smogcast.tracking`.
MLflow: local SQLite store at mlflow/mlflow.db, artifacts under mlflow/artifacts (both gitignored).
mlflow-skinny lives in .pydeps/ (installing it into the shared venv would downgrade protobuf and anyio),
so .pydeps is appended to the END of sys.path: venv packages always win on conflicts.
Model versions from the file registry are mirrored as MLflow model versions of "smogcast-pm25"
(tag registry_version=vNNN); the "champion" alias follows models/CURRENT.
Disable MLflow with SMOGCAST_MLFLOW=0. Telemetry is switched off before import.

    python -m smogcast.tracking          # list jsonl runs
    python -m smogcast.tracking --sync   # mirror every registry version into MLflow, set champion alias
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config as C

RUNS = C.REPORTS / "runs.jsonl"
MLFLOW_DIR = Path(os.environ.get("SMOGCAST_MLFLOW_DIR", C.ROOT / "mlflow"))
MODEL_NAME = "smogcast-pm25"
EXPERIMENT = "smogcast"


def _mlflow():
    """Return the mlflow module pointed at the local store, or None if disabled or not installed."""
    if os.environ.get("SMOGCAST_MLFLOW", "1") == "0":
        return None
    for k, v in {"MLFLOW_DISABLE_TELEMETRY": "true", "DO_NOT_TRACK": "true", "MLFLOW_DISABLE_AGENT_HINT": "1"}.items():
        os.environ.setdefault(k, v)
    deps = str(C.ROOT / ".pydeps")
    if deps not in sys.path:
        sys.path.append(deps)
    try:
        import mlflow
    except ImportError:
        return None
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(MLFLOW_DIR / "artifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def _flat(d: dict) -> dict:
    """MLflow params/metrics must be scalars."""
    return {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool)) and v == v}


def _register(mlflow, run_id: str, artifact_uri: str, version: str) -> str:
    c = mlflow.MlflowClient()
    if not c.search_registered_models(f"name='{MODEL_NAME}'"):
        c.create_registered_model(MODEL_NAME, description="HistGradientBoosting PM2.5 forecaster, 24 h and 48 h")
    mv = c.create_model_version(MODEL_NAME, f"{artifact_uri}/model", run_id=run_id, tags={"registry_version": version})
    return mv.version


def log_run(name: str, params: dict, metrics: dict, artifacts: dict | None = None, model_dir: Path | None = None) -> None:
    """Append to runs.jsonl; mirror to MLflow. If model_dir is given, also register it as a model version."""
    C.REPORTS.mkdir(exist_ok=True)
    rec = {"name": name, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "params": params, "metrics": metrics, "artifacts": dict(artifacts or {})}
    mlflow = _mlflow()
    if mlflow is not None:
        with mlflow.start_run(run_name=name) as r:
            mlflow.log_params(_flat(params))
            mlflow.log_metrics({k: float(v) for k, v in _flat(metrics).items() if not isinstance(v, str)})
            mlflow.set_tags(_flat({k: json.dumps(v) if isinstance(v, list) else v for k, v in (artifacts or {}).items()}))
            if model_dir is not None:
                mlflow.log_artifacts(str(model_dir), "model")
                rec["artifacts"]["mlflow_model_version"] = _register(mlflow, r.info.run_id, r.info.artifact_uri, model_dir.name)
        rec["artifacts"]["mlflow_run_id"] = r.info.run_id
    with RUNS.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def sync() -> None:
    """Backfill registry versions missing from MLflow and point the champion alias at models/CURRENT."""
    from . import registry as R
    mlflow = _mlflow()
    if mlflow is None:
        raise SystemExit("mlflow not importable: install with uv pip install --target .pydeps mlflow-skinny (see README)")
    c = mlflow.MlflowClient()
    have = {mv.tags.get("registry_version"): mv.version for mv in c.search_model_versions(f"name='{MODEL_NAME}'")}
    for v in R.versions():
        if v not in have:
            m = R.meta(v)
            ev = m.get("eval", {})
            log_run("register", _flat({k: m.get(k) for k in ("train_end", "n_train", "trained_at")}),
                    {f"{h}_{k}_mae": ev[h][k]["mae"] for h in ev for k in ("model", "persistence")},
                    {"version": v}, model_dir=C.MODELS / v)
    have = {mv.tags.get("registry_version"): mv.version for mv in c.search_model_versions(f"name='{MODEL_NAME}'")}
    cur = R.current()
    if cur in have:
        c.set_registered_model_alias(MODEL_NAME, "champion", have[cur])
    print(f"{len(have)} model versions in MLflow; champion -> {cur} (mlflow version {have.get(cur)})")


def main() -> None:
    if "--sync" in sys.argv:
        sync()
        return
    if not RUNS.exists():
        print("no runs logged yet")
        return
    for line in RUNS.read_text().splitlines():
        r = json.loads(line)
        print(r["at"], r["name"], json.dumps(r["metrics"], default=str)[:200])


if __name__ == "__main__":
    main()
