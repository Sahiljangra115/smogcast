import numpy as np
import pandas as pd

MAPE_FLOOR = 10.0  # ug/m3; MAPE is undefined near zero, so score it only where truth >= this


def mae(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p)))


def mape(y, p, floor: float = MAPE_FLOOR):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = y >= floor
    return float(np.mean(np.abs(y[m] - p[m]) / y[m]) * 100) if m.any() else float("nan")


def score(y, p) -> dict:
    return {"mae": mae(y, p), "mape": mape(y, p), "n": int(len(y))}


def clean_pairs(y, p):
    d = pd.DataFrame({"y": y, "p": p}).dropna()
    return d["y"].to_numpy(), d["p"].to_numpy()
