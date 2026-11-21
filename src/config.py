"""Central configuration (single source of truth for paths, splits, tiers)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "bangladesh_dengue_weekly.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

# Chronological split (weekly data, Mondays). A purge gap equal to the
# longest forecast horizon keeps target windows from crossing boundaries.
TRAIN_END = "2022-12-31"   # train: forecast dates <= TRAIN_END - gap
VAL_END = "2024-06-30"     # val: (TRAIN_END, VAL_END - gap]; test: > VAL_END
MAX_HORIZON = 4

# Forecast windows = which weeks-ahead are summed into the target.
WINDOWS = {
    "Weeks 2-4 ahead (total)": (2, 3, 4),
    "Week 2 ahead": (2,),
    "Week 3 ahead": (3,),
    "Week 4 ahead": (4,),
}
MAIN_WINDOW = "Weeks 2-4 ahead (total)"

RISK_TIERS = ["Low", "Medium", "High", "Critical"]
# Cut-points are expressed per forecast week (cases/week) and scaled by window length.
# Fit from TRAINING data only (see train.py) -> stored in model bundle.
TIER_QUANTILES = (0.60, 0.85, 0.95)

WEATHER_COLS = ["avg_temp_c", "rainfall_mm", "relative_humidity_pct"]
STATIC_COLS = ["population_density_sqkm", "water_logging_risk_index", "sanitation_score"]
LAGS = list(range(1, 9))       # weekly lags (covers 2-4 week breeding clock + margin)
ROLL_WINDOWS = [2, 3, 4, 6, 8]

# ---- Operational early-warning flag (sensitive operating point, chosen on VALIDATION only) ----
ALERT_MAX_FALSE_ALARM = 0.10   # cap on validation false-alarm rate when choosing the alert threshold
ALERT_BETA = 2.0               # F-beta with beta=2 weights recall twice as much as precision
ALERT_FACTORS = [round(x, 2) for x in [0.40 + 0.05 * i for i in range(13)]]   # alert cut = factor x High cut

# ---- Geography ----
# The dataset has NO ward coordinates and the ward ids (e.g. DNCC_Ward_01) are not mapped to real,
# verifiable ward polygons, so we never fabricate ward positions.  Two honest layers instead:
#  1) APPROXIMATE city-centre coordinates (+-5 km, public geographic knowledge) used ONLY for the
#     city-level map and as the weather-lookup point of the optional live-weather layer.
#  2) OPTIONAL data/ward_coordinates.csv (columns ward_id,lat,lon) -> real ward-level heatmap as soon
#     as verified coordinates are supplied.  The file is intentionally NOT shipped.
CITY_CENTRES = {
    "Dhaka North": (23.82, 90.39),
    "Dhaka South": (23.71, 90.41),
    "Chattogram": (22.36, 91.78),
    "Gazipur": (23.99, 90.42),
    "Narayanganj": (23.62, 90.50),
}
WARD_COORD_PATH = ROOT / "data" / "ward_coordinates.csv"
LIVE_CACHE_DIR = ROOT / ".cache" / "live_weather"
