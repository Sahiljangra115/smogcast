from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
INGEST_LOG = ROOT / "data" / "ingest_log.csv"

# Delhi NCR query points. These are model grid cells (CAMS via Open-Meteo), NOT CPCB stations.
# The CAMS grid is ~0.4 deg, so the 10 named points return only 5 distinct PM2.5 series
# (max abs difference 0.0 over 2023-01-01 to 2026-10-14; weather still differs per point).
ALL_POINTS = {
    "delhi_centre": (28.61, 77.21),
    "anand_vihar": (28.65, 77.32),
    "dwarka": (28.59, 77.05),
    "rohini": (28.74, 77.12),
    "noida": (28.57, 77.32),
    "gurugram": (28.46, 77.03),
    "faridabad": (28.41, 77.31),
    "panipat": (29.39, 76.97),
    "meerut": (28.98, 77.71),
    "rewari": (28.20, 76.62),
}
# One point per distinct CAMS cell. Training, evaluation, replay, drift and serving use only these,
# so each distinct PM2.5 series counts once (before v003, all 10 points were pooled and the two big
# cells were counted 3x and 4x).
CELL_MEMBERS = {
    "delhi_centre": ["anand_vihar", "rohini"],
    "dwarka": ["noida", "gurugram", "faridabad"],
    "panipat": [],
    "meerut": [],
    "rewari": [],
}
SITES = {s: ALL_POINTS[s] for s in CELL_MEMBERS}
# Fixed site_id per point (the original 10-point order), so v001/v002, which were fit on all 10 points, stay valid.
SITE_IDS = {s: i for i, s in enumerate(ALL_POINTS)}

HORIZONS = (24, 48)
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WX_HISTFC_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
WX_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
           "wind_direction_10m", "boundary_layer_height", "surface_pressure"]
START_DATE = "2023-01-01"

# Frozen split: never changes, so metrics stay comparable across model versions.
TRAIN_END = "2025-06-30 23:00"
TEST_START = "2025-07-01 00:00"
