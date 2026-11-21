"""Ward-level weather lag / rolling feature engineering (leak-free).

Every feature at forecast date t uses only information observed at or before t.
Lags/rolls are computed per ward_id (groupby), never across wards.
"""
import numpy as np
import pandas as pd

from .config import (DATA_PATH, LAGS, ROLL_WINDOWS, STATIC_COLS, WEATHER_COLS)

SHORT = {"avg_temp_c": "temp", "rainfall_mm": "rain", "relative_humidity_pct": "hum"}
REQUIRED = ["date", "city", "ward_id", *STATIC_COLS, *WEATHER_COLS, "dengue_cases"]


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df.sort_values(["ward_id", "date"]).reset_index(drop=True)


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Weather lags (1-8w), trailing rolling sums/means, seasonality."""
    df = df.sort_values(["ward_id", "date"]).reset_index(drop=True)
    g = df.groupby("ward_id", sort=False)
    new = {}
    for col in WEATHER_COLS:
        s = SHORT[col]
        for k in LAGS:
            new[f"{s}_lag_{k}w"] = g[col].shift(k)
        for w in ROLL_WINDOWS:
            # trailing window ending at t (includes current week -> known at forecast time)
            fn = "sum" if col == "rainfall_mm" else "mean"
            new[f"{s}_roll{w}w_{fn}"] = g[col].transform(
                lambda x, w=w, fn=fn: getattr(x.rolling(w, min_periods=w), fn)())
    woy = df["date"].dt.isocalendar().week.astype(int)
    new["week_sin"] = np.sin(2 * np.pi * woy / 52.0)
    new["week_cos"] = np.cos(2 * np.pi * woy / 52.0)
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def add_case_history(df: pd.DataFrame) -> pd.DataFrame:
    """Optional autoregressive features (observed cases up to t). Kept separate so
    weather-only vs weather+history can be compared honestly."""
    g = df.groupby("ward_id", sort=False)["dengue_cases"]
    new = {f"cases_lag_{k}w": g.shift(k) for k in (1, 2, 3, 4)}
    new["cases_roll4w_mean"] = g.transform(lambda x: x.rolling(4, min_periods=4).mean())
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def add_target(df: pd.DataFrame, horizons=(2, 3, 4)) -> pd.Series:
    """Future cases summed over the given weeks-ahead (NaN where unobserved)."""
    g = df.groupby("ward_id", sort=False)["dengue_cases"]
    parts = [g.shift(-h) for h in horizons]
    out = parts[0].copy()
    for p in parts[1:]:
        out = out + p
    return out


def feature_sets(df: pd.DataFrame) -> dict:
    """Named feature configurations for the ablation study."""
    wx = [c for c in df.columns if any(c.startswith(f"{s}_") for s in SHORT.values())]
    cur = list(WEATHER_COLS)
    season = ["week_sin", "week_cos"]
    hist = [c for c in df.columns if c.startswith("cases_")]
    return {
        "weather_only": cur + wx + STATIC_COLS + season,
        "weather_no_static": cur + wx + season,
        "weather_plus_history": cur + wx + STATIC_COLS + season + hist,
    }


def feature_groups(cols) -> dict:
    """Group columns into human-readable drivers for explanations."""
    groups = {"Rainfall (lagged)": [], "Rainfall (current)": [], "Temperature": [],
              "Humidity": [], "Ward vulnerability": [], "Season": [], "Recent cases": []}
    for c in cols:
        if c == "rainfall_mm":
            groups["Rainfall (current)"].append(c)
        elif c.startswith("rain_"):
            groups["Rainfall (lagged)"].append(c)
        elif c.startswith("temp_") or c == "avg_temp_c":
            groups["Temperature"].append(c)
        elif c.startswith("hum_") or c == "relative_humidity_pct":
            groups["Humidity"].append(c)
        elif c in STATIC_COLS:
            groups["Ward vulnerability"].append(c)
        elif c.startswith("week_"):
            groups["Season"].append(c)
        elif c.startswith("cases_"):
            groups["Recent cases"].append(c)
    return {k: v for k, v in groups.items() if v}


def build_dataset(path=DATA_PATH, with_history=True) -> pd.DataFrame:
    df = add_weather_features(load_raw(path))
    return add_case_history(df) if with_history else df
