import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from smogcast import collect, drift, features as F, metrics as M, registry as R, train as T
from smogcast import config as C


def synth(n=24 * 40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    t = np.arange(n)
    pm = 100 + 40 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 10, n)
    return pd.DataFrame({
        "pm2_5": pm, "pm10": pm * 1.8, "temperature_2m": 15 + rng.normal(0, 2, n),
        "relative_humidity_2m": 60 + rng.normal(0, 5, n), "wind_speed_10m": 8 + rng.normal(0, 2, n),
        "wind_direction_10m": rng.uniform(0, 360, n), "boundary_layer_height": 500 + rng.normal(0, 50, n),
        "surface_pressure": 1000 + rng.normal(0, 3, n)}, index=idx)


def test_features_no_future_leak():
    df = synth()
    a = F.build_site(df, "dwarka")
    df2 = df.copy()
    df2.iloc[-30:, df2.columns.get_loc("pm2_5")] += 500  # change the future only
    b = F.build_site(df2, "dwarka")
    cut = len(df) - 31
    cols = F.feature_cols(a)
    pd.testing.assert_frame_equal(a[cols].iloc[:cut], b[cols].iloc[:cut])


def test_target_alignment():
    df = synth()
    a = F.build_site(df, "dwarka")
    assert a["y24"].iloc[10] == df["pm2_5"].iloc[34]
    assert a["base_snaive24"].iloc[200] == df["pm2_5"].iloc[200 + 24 - 168]


def test_missing_hours_become_nan_not_crash():
    df = synth().drop(synth().index[100:110])
    out = F.build_site(df.asfreq("h"), "dwarka")
    assert out["pm_lag1"].iloc[100:111].isna().sum() >= 1


def test_metrics():
    assert M.mae([1, 2, 3], [2, 2, 5]) == pytest.approx(1.0)
    assert np.isnan(M.mape([1, 2], [1, 2]))  # below the floor, undefined
    assert M.mape([100], [110]) == pytest.approx(10.0)


def test_psi_and_ks_detect_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 5000)
    assert drift.psi(a, rng.normal(0, 1, 2000)) < 0.1
    assert drift.psi(a, rng.normal(2, 1, 2000)) > 0.25
    cols = drift.PSI_COLS + ["pm2_5"]
    ref = pd.DataFrame({c: rng.normal(0, 1, 3000) for c in cols})
    same = pd.DataFrame({c: rng.normal(0, 1, 500) for c in cols})
    shifted = same.assign(pm2_5=same["pm2_5"] + 5, wind_speed_10m=same["wind_speed_10m"] + 3)
    assert not drift.check(ref, same)["drift"]
    r = drift.check(ref, shifted)
    assert r["drift"] and "ks:pm2_5" in r["flags"]


def test_registry_promote_rollback(tmp_path):
    v1 = R.register({"x": 1}, {}, tmp_path)
    v2 = R.register({"x": 2}, {}, tmp_path)
    R.promote(v1, tmp_path)
    R.promote(v2, tmp_path)
    assert R.current(tmp_path) == v2
    R.rollback(tmp_path)
    assert R.current(tmp_path) == v1
    assert R.load(root=tmp_path)["x"] == 1


def test_gate_passes_equal_blocks_worse_candidate():
    from smogcast import retrain
    df = F.build_site(synth(), "dwarka").reset_index(names="time")
    df["site"] = "dwarka"
    feats = F.feature_cols(df)
    good = T.fit(df.iloc[:600], feats, (24, 48))
    bad = T.fit(df.iloc[:600].assign(y24=lambda d: d["y24"] * 0 + 500, y48=lambda d: d["y48"] * 0 + 500), feats, (24, 48))
    tie = retrain.gate(good, good, df.iloc[600:])
    assert tie["promote"]  # identical candidate passes (equal or better)
    assert set(tie["per_horizon"]) == {"h24", "h48"}  # MAE reported per horizon, not only as a mean
    assert tie["per_horizon"]["h24"]["candidate_mae"] == tie["per_horizon"]["h24"]["current_mae"]
    assert "h24 MAE: candidate" in retrain.describe(tie) and "h48 MAE: candidate" in retrain.describe(tie)
    assert retrain.gate(good, bad, df.iloc[600:])["promote"]
    assert not retrain.gate(bad, good, df.iloc[600:])["promote"]


def test_sites_are_distinct_cams_cells():
    assert len(C.SITES) == 5 and set(C.SITES) == set(C.CELL_MEMBERS)
    members = [m for ms in C.CELL_MEMBERS.values() for m in ms]
    assert set(C.SITES) | set(members) == set(C.ALL_POINTS) and len(set(C.SITE_IDS.values())) == 10
    if not all((C.RAW / f"{s}.csv").exists() for s in C.ALL_POINTS):
        pytest.skip("data/raw not populated")
    pm = {s: pd.read_csv(C.RAW / f"{s}.csv", index_col=0)["pm2_5"] for s in C.ALL_POINTS}
    sites = list(C.SITES)
    for i, a in enumerate(sites):  # the modelled points are pairwise different series
        for b in sites[i + 1:]:
            assert not pm[a].equals(pm[b]), (a, b)
    for s, ms in C.CELL_MEMBERS.items():  # each dropped point really duplicates its cell representative
        for m in ms:
            assert pm[m].equals(pm[s]), (s, m)


def test_collector_survives_outage(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "INGEST_LOG", tmp_path / "log.csv")
    monkeypatch.setattr(collect.time, "sleep", lambda s: None)

    class R500:
        status_code = 500

    monkeypatch.setattr(collect.requests, "get", lambda *a, **k: R500())
    df = collect._fetch_range("aq", "dwarka", C.AQ_URL, {}, pd.Timestamp("2025-01-01").date(), pd.Timestamp("2025-01-10").date())
    assert df.empty  # no crash, gap left
    assert (tmp_path / "log.csv").read_text().count(",0,0,HTTP 500") == 1


def test_serve_predict_and_errors(tmp_path, monkeypatch):
    from smogcast import serve
    df = F.build_site(synth(), "dwarka").reset_index(names="time")
    bundle = T.fit(df, F.feature_cols(df), (24, 48))
    bundle["version"] = "vtest"
    serve._state["bundle"] = bundle
    c = TestClient(serve.app)
    hist = synth(200).reset_index(names="time").astype({"time": str}).to_dict("records")
    r = c.post("/predict", json={"site": "dwarka", "history": hist})
    assert r.status_code == 200
    body = r.json()
    assert set(body["forecast"]) == {"24h", "48h"} and body["model_version"] == "vtest"
    assert c.post("/predict", json={"site": "mars", "history": hist}).status_code == 404
    assert c.post("/predict", json={"site": "dwarka", "history": hist[:10]}).status_code == 422
    m = c.get("/metrics").text
    assert 'smogcast_predictions_total{model_version="vtest",variant="current"} 1.0' in m
    assert 'smogcast_request_latency_seconds_count{endpoint="/predict"} 3.0' in m
    assert 'smogcast_errors_total{endpoint="/predict"} 2.0' in m


def test_canary_split(tmp_path, monkeypatch):
    from smogcast import serve
    R.register({"x": 1}, {}, tmp_path)
    R.register({"x": 2}, {}, tmp_path)
    R.promote("v001", tmp_path)
    with pytest.raises(RuntimeError):
        R.set_canary("v009", tmp_path)
    R.set_canary("v002", tmp_path)
    assert R.canary(tmp_path) == "v002"
    serve._state["bundle"] = {"version": "v001"}
    serve._state["canary"] = {"version": "v002"}
    for w, lo, hi in (("0", 0, 0), ("0.2", 0.15, 0.25), ("1", 1, 1)):
        monkeypatch.setenv("CANARY_WEIGHT", w)
        share = np.mean([serve.pick()[1] == "canary" for _ in range(4000)])
        assert lo <= share <= hi
    R.promote("v002", tmp_path)  # promoting the canary clears the pointer
    assert R.canary(tmp_path) is None and R.current(tmp_path) == "v002"


def test_external_sources_fail_clearly_without_keys(monkeypatch):
    from smogcast import external
    monkeypatch.delenv("OPENAQ_API_KEY", raising=False)
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    with pytest.raises(external.MissingKey, match="explore.openaq.org/register"):
        external.openaq_locations(28.6, 77.2)
    with pytest.raises(external.MissingKey, match="firms.modaps.eosdis.nasa.gov/api/map_key"):
        external.firms_daily_counts("2025-11-01")


def test_firms_parse(monkeypatch):
    from smogcast import external
    monkeypatch.setenv("FIRMS_MAP_KEY", "dummy")

    class Resp:
        text = "latitude,longitude,acq_date\n30,75,2025-11-01\n30,75,2025-11-01\n31,76,2025-11-02\n"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(external.requests, "get", lambda *a, **k: Resp())
    assert external.firms_daily_counts("2025-11-02", 2).to_dict() == {"2025-11-01": 2, "2025-11-02": 1}


def test_tracking_mlflow_roundtrip(tmp_path, monkeypatch):
    from smogcast import tracking
    monkeypatch.setattr(tracking, "RUNS", tmp_path / "runs.jsonl")
    monkeypatch.setattr(tracking, "MLFLOW_DIR", tmp_path / "mlflow")
    mlflow = tracking._mlflow()
    if mlflow is None:
        pytest.skip("mlflow not importable (.pydeps missing)")
    mdir = tmp_path / "v001"
    mdir.mkdir()
    (mdir / "meta.json").write_text("{}")
    tracking.log_run("train", {"lr": 0.05}, {"h24_model_mae": 27.1}, {"version": "v001"}, model_dir=mdir)
    rec = __import__("json").loads((tmp_path / "runs.jsonl").read_text())
    assert str(rec["artifacts"]["mlflow_model_version"]) == "1"
    run = mlflow.get_run(rec["artifacts"]["mlflow_run_id"])
    assert run.data.metrics["h24_model_mae"] == pytest.approx(27.1)
