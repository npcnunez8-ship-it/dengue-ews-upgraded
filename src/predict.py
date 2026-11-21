"""Fast local inference from the serialized bundle (no network / API calls)."""
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C
from .features import build_dataset
from .model_io import BUNDLE_PATH, ensure_bundle
from .risk import to_tier


@lru_cache(maxsize=1)
def load_bundle(path=str(BUNDLE_PATH)):
    """Load the serialized bundle (auto-rebuilds from the CSV if it is unusable on this machine)."""
    from pathlib import Path
    return ensure_bundle(Path(path))


def load_features(path=C.DATA_PATH) -> pd.DataFrame:
    """Feature table for every ward-week (computed once; wrap in st.cache_data in the UI)."""
    return build_dataset(path, with_history=False)


def predict_frame(df: pd.DataFrame, window=C.MAIN_WINDOW, bundle=None) -> pd.DataFrame:
    """Vectorised batch prediction -> predicted cases, 80% interval, risk tier."""
    b = bundle or load_bundle()
    w = b["windows"][window]
    X = df[b["features"]]
    ok = X.notna().all(axis=1)
    out = pd.DataFrame(index=df.index, columns=["pred", "lo", "hi", "tier", "alert"], dtype=object)
    if ok.any():
        Xo = X[ok]
        p = np.maximum(w["model"].predict(Xo), 0)
        lo = np.maximum(w["lo"].predict(Xo) - w["qhat"], 0)
        hi = np.maximum(w["hi"].predict(Xo) + w["qhat"], lo)
        if w.get("interval_method") == "scaled":
            h = w["qhat"] * np.sqrt(p + 1.0)
            lo, hi = np.maximum(p - h, 0), p + h
        out.loc[ok, "pred"], out.loc[ok, "lo"], out.loc[ok, "hi"] = p, lo, hi
        out.loc[ok, "tier"] = to_tier(p, w["cutpoints"])
        out.loc[ok, "alert"] = p >= w.get("alert_cut", w["cutpoints"][1])
    for c in ["pred", "lo", "hi"]:
        out[c] = out[c].astype(float)
    out["alert"] = out["alert"].map(lambda v: bool(v) if pd.notna(v) else False)
    return out


def explain_row(row: pd.DataFrame, window=C.MAIN_WINDOW, bundle=None) -> pd.DataFrame:
    """Driver breakdown for ONE ward-week via grouped occlusion.

    For each driver group, replace its features by the seasonal-normal value
    (same ISO week, historical mean; ward attributes -> city-wide mean) and report
    the change in predicted cases. Positive = pushes risk UP vs normal conditions.
    This is a what-if effect, not an additive Shapley decomposition.
    """
    b = bundle or load_bundle()
    w = b["windows"][window]
    cols = b["features"]
    x = row[cols].iloc[[0]].copy()
    base = float(max(w["model"].predict(x)[0], 0))
    wk = int(min(row["date"].iloc[0].isocalendar().week, 52))
    clim = b["climatology"].loc[wk] if wk in b["climatology"].index else b["static_ref"]
    rows = []
    for g, gcols in b["groups"].items():
        xr = x.copy()
        ref = b["static_ref"] if g == "Ward vulnerability" else clim
        for c in gcols:
            xr[c] = ref[c]
        delta = base - float(max(w["model"].predict(xr)[0], 0))
        rows.append({"driver": g, "effect_cases": delta})
    return pd.DataFrame(rows).sort_values("effect_cases", ascending=False).reset_index(drop=True)


def predict_one(df: pd.DataFrame, ward_id: str, date, window=C.MAIN_WINDOW) -> dict:
    """Programmatic API used by tests / external callers."""
    row = df[(df.ward_id == ward_id) & (df.date == pd.Timestamp(date))]
    if row.empty:
        raise ValueError("ward/date not found")
    pr = predict_frame(row, window).iloc[0]
    return {"ward_id": ward_id, "date": str(pd.Timestamp(date).date()), "window": window,
            "predicted_cases": round(pr.pred, 2), "interval_80": [round(pr.lo, 1), round(pr.hi, 1)],
            "risk_tier": pr.tier}
