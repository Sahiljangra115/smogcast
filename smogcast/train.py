from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config as C
from . import features as F
from . import metrics as M
from . import registry as R
from . import tracking

PARAMS = dict(loss="absolute_error", learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
              min_samples_leaf=50, l2_regularization=1.0, random_state=0)


def fit(train: pd.DataFrame, feats: list[str], horizons=C.HORIZONS) -> dict:
    models = {}
    for h in horizons:
        d = train.dropna(subset=[f"y{h}"])
        models[h] = HistGradientBoostingRegressor(**PARAMS).fit(d[feats], d[f"y{h}"])
    return {"models": models, "feats": feats, "horizons": tuple(horizons)}


def predict(bundle: dict, X: pd.DataFrame, h: int) -> np.ndarray:
    return np.clip(bundle["models"][h].predict(X[bundle["feats"]]), 0, None)


def evaluate(bundle: dict, df: pd.DataFrame) -> dict:
    """Model vs persistence vs seasonal naive, overall and per site, on rows with a realised target."""
    res: dict = {}
    for h in bundle["horizons"]:
        d = df.dropna(subset=[f"y{h}", "base_persist", f"base_snaive{h}"]).copy()
        d["pred"] = predict(bundle, d, h)
        preds = {"model": "pred", "persistence": "base_persist", "seasonal_naive": f"base_snaive{h}"}
        r = {k: M.score(d[f"y{h}"], d[c]) for k, c in preds.items()}
        r["per_site"] = {s: {k: M.score(g[f"y{h}"], g[c]) for k, c in preds.items()} for s, g in d.groupby("site")}
        res[f"h{h}"] = r
    return res


def monthly(bundle: dict, df: pd.DataFrame, h: int) -> pd.DataFrame:
    d = df.dropna(subset=[f"y{h}", "base_persist"]).copy()
    d["pred"] = predict(bundle, d, h)
    d["month"] = d["time"].dt.to_period("M")
    rows = []
    for m, g in d.groupby("month"):
        rows.append({"month": str(m), "n": len(g), "mean_pm25": g[f"y{h}"].mean(),
                     "mae_model": M.mae(g[f"y{h}"], g["pred"]), "mae_persist": M.mae(g[f"y{h}"], g["base_persist"])})
    out = pd.DataFrame(rows)
    out["skill"] = 1 - out["mae_model"] / out["mae_persist"]
    return out


def split(df: pd.DataFrame):
    t = pd.Timestamp(C.TRAIN_END)
    # 48h purge: train rows whose y48 target would land in the test window are dropped
    return df[df["time"] <= t - pd.Timedelta(hours=max(C.HORIZONS))], df[df["time"] >= pd.Timestamp(C.TEST_START)]


def main() -> None:
    """Fit on the frozen train split, register a new version, evaluate on the test split.
    --eval-only: re-evaluate the CURRENT model on the data now on disk, register nothing."""
    df = F.build_all()
    feats = F.feature_cols(df)
    train, test = split(df)
    eval_only = "--eval-only" in sys.argv
    if eval_only:
        bundle = R.load()
        v = bundle["version"]
    else:
        bundle = fit(train, feats)
    res = evaluate(bundle, test)
    mon = {f"h{h}": monthly(bundle, test, h).round(3).to_dict("records") for h in C.HORIZONS}
    mvals = {f"{h}_{k}_mae": res[h][k]["mae"] for h in res for k in ("model", "persistence", "seasonal_naive")}
    if not eval_only:
        meta = {"trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "train_end": C.TRAIN_END, "test_start": C.TEST_START, "n_train": int(len(train)),
                "features": feats, "params": PARAMS, "eval": res}
        v = R.register(bundle, meta)
        if R.current() is None:
            R.promote(v)
    tracking.log_run("eval" if eval_only else "train", {**PARAMS, "train_end": C.TRAIN_END, "n_train": int(len(train))},
                     mvals, {"version": v}, model_dir=None if eval_only else C.MODELS / v)
    C.REPORTS.mkdir(exist_ok=True)
    data_end = str(df["time"].max())
    (C.REPORTS / "metrics.json").write_text(json.dumps({"version": v, "data_end": data_end, "eval": res, "monthly": mon}, indent=2))
    for h in C.HORIZONS:
        r = res[f"h{h}"]
        print(f"h{h}: " + "  ".join(f"{k}: MAE {r[k]['mae']:.1f} MAPE {r[k]['mape']:.1f}%" for k in ("model", "persistence", "seasonal_naive")))


if __name__ == "__main__":
    main()
