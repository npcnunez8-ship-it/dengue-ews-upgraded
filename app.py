"""Dengue Early-Warning System - Streamlit decision-support dashboard.   Run:  streamlit run app.py"""
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.features import add_target
from src.geo import city_summary, ward_coordinates
from src.live_weather import fetch_city_weather, history_for_scoring
from src.predict import explain_row, load_bundle, load_features, predict_frame
from src.quality import repair_history, unusual_observations
from src.scenario import N_HISTORY, all_horizons, apply_scenario, predict_history, ward_history

st.set_page_config(page_title="Dengue Early Warning - Bangladesh", page_icon="🦟", layout="wide")
TIER_COLOR = {"Low": "#2e9e5b", "Medium": "#e6b800", "High": "#e8710a", "Critical": "#c62828"}
TIER_IDX = {t: i for i, t in enumerate(C.RISK_TIERS)}
ADVICE = {
    "Low": "Routine surveillance. No extra action needed.",
    "Medium": "Increase larval surveys and drain-clearing in this ward over the next 2 weeks.",
    "High": "Deploy larvicide/spraying teams and alert nearby hospitals to prepare beds and saline.",
    "Critical": "Priority ward: immediate vector control, community alerts and surge capacity at hospitals.",
}
st.markdown("""
<style>
.block-container{padding-top:1.2rem;max-width:1400px}
.hero{background:linear-gradient(120deg,#0b3d3a 0%,#0f766e 60%,#2a9d8f 100%);color:#fff;padding:18px 24px;border-radius:14px;margin-bottom:14px}
.hero h1{margin:0;font-size:1.65rem;color:#fff}.hero p{margin:4px 0 0 0;opacity:.88;font-size:.95rem}
[data-testid="stMetric"]{background:#f6faf9;border:1px solid #dbe8e5;border-radius:12px;padding:10px 14px}
.badge{display:block;color:#fff;padding:10px 12px;border-radius:10px;font-size:1.35rem;font-weight:700;text-align:center}
.hcard{border:1px solid #dbe8e5;border-radius:12px;padding:10px 14px;background:#fbfdfc}
.hcard .t{font-size:.8rem;color:#52706a;text-transform:uppercase;letter-spacing:.04em}
.hcard .v{font-size:1.7rem;font-weight:700;line-height:1.2}.hcard .i{font-size:.85rem;color:#52706a}
.pill{display:inline-block;color:#fff;border-radius:999px;padding:1px 10px;font-size:.8rem;font-weight:600}
.small{font-size:.82rem;color:#52706a}
</style>""", unsafe_allow_html=True)


# ---------------- cached resources ----------------
@st.cache_resource(show_spinner="Loading model (rebuilds automatically if this machine needs it)...")
def bundle():
    return load_bundle()


@st.cache_data(show_spinner="Building ward-level weather-lag features (once)...")
def features():
    return load_features()


@st.cache_data(show_spinner="Running batch inference...")
def all_predictions(window):
    df = features()
    p = predict_frame(df, window, bundle())
    out = pd.concat([df[["date", "city", "ward_id", "dengue_cases"]], p], axis=1)
    out["actual"] = add_target(df, C.WINDOWS[window])
    return out


@st.cache_data
def lag_corr():
    d = features().sort_values(["ward_id", "date"])
    g = d.groupby("ward_id")["rainfall_mm"]
    return pd.DataFrame([{"lag_weeks": k, "correlation": d.dengue_cases.corr(g.shift(k))} for k in range(0, 9)])


@st.cache_data
def report_csv(name):
    return pd.read_csv(C.REPORT_DIR / name)


@st.cache_data
def metrics():
    return json.loads((C.REPORT_DIR / "metrics.json").read_text())


@st.cache_data
def audit():
    return json.loads((C.REPORT_DIR / "audit.json").read_text())


@st.cache_data
def test_preds():
    return pd.read_csv(C.REPORT_DIR / "test_predictions.csv", parse_dates=["date"])


def badge(tier):
    return f"<span class='badge' style='background:{TIER_COLOR[tier]}'>{tier}</span>"


def pill(tier):
    return f"<span class='pill' style='background:{TIER_COLOR[tier]}'>{tier}</span>"


df, B, M = features(), bundle(), metrics()
VAL_END = pd.Timestamp(C.VAL_END)

# ---------------- sidebar ----------------
st.sidebar.title("🦟 Dengue Early Warning")
window = st.sidebar.selectbox("Forecast window", list(C.WINDOWS), index=0,
                              help="Cases summed over the selected weeks ahead of the prediction date.")
city = st.sidebar.selectbox("City", sorted(df.city.unique()))
ward = st.sidebar.selectbox("Ward", sorted(df[df.city == city].ward_id.unique()))
valid_dates = df.loc[df[B["features"]].notna().all(axis=1), "date"].drop_duplicates().sort_values()
default = valid_dates[(valid_dates.dt.year == 2025) & (valid_dates.dt.month == 8)]
default = default.iloc[0] if len(default) else valid_dates.iloc[-1]
date = st.sidebar.select_slider("Prediction date (weekly)", options=list(valid_dates), value=default,
                                format_func=lambda d: d.strftime("%Y-%m-%d"))
if date <= VAL_END:
    st.sidebar.warning(f"In-sample date (training/validation period). For honest out-of-sample behaviour pick a date after {VAL_END.date()}.")
else:
    st.sidebar.success("Out-of-sample date: never used for training or model selection.")
st.sidebar.info(f"Data: {df.date.min().date()} → {df.date.max().date()} (supplied **simulated-looking** dataset). Weather source here is "
                "**observed historical** data from the CSV - no forecast weather and no live feed unless you use the Live tab.")
st.sidebar.caption("Forecast timing: made at the end of week *t* using only weather observed up to week *t*; "
                   "'weeks 2-4 ahead' = cases in weeks t+2..t+4 (week t+1 is the unavoidable blind week).")

P = all_predictions(window)
snap_all = P[P.date == date].dropna(subset=["pred"]).copy()
row = P[(P.ward_id == ward) & (P.date == date)].iloc[0]
cuts = B["windows"][window]["cutpoints"]
alert_cut = B["windows"][window].get("alert_cut", cuts[1])

st.markdown(f"<div class='hero'><h1>🦟 Urban Dengue Outbreak Early-Warning</h1><p>Ward-level risk 2-4 weeks ahead from multi-week weather lags · "
            f"{city} / {ward} · prediction date <b>{date.date()}</b> · window <b>{window}</b></p></div>", unsafe_allow_html=True)
tabs = st.tabs(["🏠 Overview", "📍 Ward forecast", "🗺️ Risk maps & ward ranking", "🧪 What-if", "🌐 Live weather (optional)",
                "📊 Validation", "ℹ️ Methodology & limits"])

# ================= 1. OVERVIEW =================
with tabs[0]:
    n_crit = int((snap_all.tier == "Critical").sum()); n_high = int((snap_all.tier == "High").sum())
    k = st.columns(5)
    k[0].metric("Wards scored", len(snap_all))
    k[1].metric("Critical wards", n_crit)
    k[2].metric("High wards", n_high)
    k[3].metric("Watch-flag wards (sensitive)", int(snap_all.alert.sum()), help=f"Sensitive early-warning flag: predicted ≥ {alert_cut:.1f} cases (validation-chosen, "
                f"~{alert_cut / cuts[1]:.0%} of the High cut-point). Trades many more false alarms (test: 19% false-alarm rate, 35% precision) for fewer missed outbreaks - in peak season most wards are flagged.")
    k[4].metric("Predicted cases (all wards)", f"{snap_all.pred.sum():,.0f}")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Who needs action first")
        top = snap_all.sort_values("pred", ascending=False).head(10)
        st.dataframe(top.assign(rank=range(1, len(top) + 1))[["rank", "city", "ward_id", "pred", "lo", "hi", "tier", "alert"]]
                     .rename(columns={"pred": "pred cases", "lo": "80% lo", "hi": "80% hi", "alert": "watch flag"}).round(1),
                     hide_index=True, width="stretch")
        st.caption(f"Top-10 wards by predicted cases ({window}). Intervals are 80% prediction intervals. Tier cut-points (cases): "
                   f"Medium ≥ {cuts[0]:.0f}, High ≥ {cuts[1]:.0f}, Critical ≥ {cuts[2]:.0f} - dataset-relative, not official alert levels.")
    with right:
        st.subheader("Risk tiers by city")
        cnt = snap_all.groupby(["city", "tier"]).size().reset_index(name="wards")
        fig = px.bar(cnt, x="wards", y="city", color="tier", orientation="h", color_discrete_map=TIER_COLOR,
                     category_orders={"tier": C.RISK_TIERS})
        fig.update_layout(height=330, margin=dict(l=0, r=0, t=10, b=0), legend_title="", yaxis_title="", xaxis_title="wards")
        st.plotly_chart(fig, width="stretch")
    st.subheader("Selected ward at a glance")
    st.markdown(f"**{ward}** ({city}): {pill(row.tier)} &nbsp; predicted **{row.pred:.1f}** cases (80% interval {row.lo:.0f}-{row.hi:.0f}). "
                f"{ADVICE[row.tier]}", unsafe_allow_html=True)
    t = M["test_by_window"][window]
    st.info(f"**How much to trust this?** On the held-out test period the model's MAE is **{t['MAE']:.2f}** cases vs **"
            f"{t['baselines']['Seasonal-naive (52w earlier)']['MAE']:.2f}** for the seasonal-naive baseline; High/Critical recall {t['HC_recall']:.0%}, "
            f"precision {t['HC_precision']:.0%}. The dataset looks **synthetic**, so these numbers show the method works on this data - "
            "they are not proof of real-world accuracy, and the model is not operationally validated.")

# ================= 2. WARD FORECAST =================
with tabs[1]:
    mh = all_horizons(df[(df.ward_id == ward) & (df.date == date)], B)
    st.subheader("Separate 2-, 3- and 4-week forecasts")
    cols = st.columns(4)
    for c, (_, r) in zip(cols, mh.sort_values("window", key=lambda s: s.map({w: i for i, w in enumerate(["Week 2 ahead", "Week 3 ahead", "Week 4 ahead", C.MAIN_WINDOW])})).iterrows()):
        c.markdown(f"<div class='hcard'><div class='t'>{r.window}</div><div class='v'>{r.pred:.1f} cases</div>"
                   f"<div class='i'>80% interval {r.lo:.0f} - {r.hi:.0f}{' · ⚠ watch flag' if r.alert else ''}</div>"
                   f"<div style='margin-top:6px'>{pill(r.tier)}</div></div>", unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric(f"Predicted ({window})", f"{row.pred:.1f}")
    c2.metric("80% prediction interval", f"{row.lo:.0f} - {row.hi:.0f}")
    if pd.notna(row.actual):
        c3.metric("Observed cases in that window (hindsight)", f"{row.actual:.0f}", f"{row.actual - row.pred:+.1f} vs forecast", delta_color="off")
    else:
        c3.metric("Observed cases in that window", "not yet observed")
    banner = st.error if row.tier in ("High", "Critical") else st.warning if row.tier == "Medium" else st.success
    banner(f"**{row.tier} risk** - {ADVICE[row.tier]}  \nTier cut-points for this window (cases): Medium ≥ {cuts[0]:.0f}, High ≥ {cuts[1]:.0f}, Critical ≥ {cuts[2]:.0f}.")

    left, right = st.columns(2)
    with left:
        st.subheader("Main prediction drivers")
        ex = explain_row(df[(df.ward_id == ward) & (df.date == date)], window, B)
        fig = px.bar(ex.iloc[::-1], x="effect_cases", y="driver", orientation="h", color=ex.iloc[::-1].effect_cases > 0,
                     color_discrete_map={True: "#c62828", False: "#2e9e5b"},
                     labels={"effect_cases": "Effect on predicted cases vs. normal conditions", "driver": ""})
        fig.update_layout(showlegend=False, height=340, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        top = ex.iloc[0]
        st.caption(f"Red = pushes risk up relative to typical weather for this week of year (ward attributes vs. the average ward). "
                   f"Biggest upward driver: **{top.driver}** ({top.effect_cases:+.1f} cases). What-if effects, not causal and not an exact additive split.")
    with right:
        st.subheader("Observed weather (last 16 weeks)")
        hist = df[(df.ward_id == ward) & (df.date <= date)].tail(16)
        fig = go.Figure()
        fig.add_bar(x=hist.date, y=hist.rainfall_mm, name="Rainfall (mm)", marker_color="#4a90d9")
        fig.add_scatter(x=hist.date, y=hist.avg_temp_c, name="Avg temp (°C)", yaxis="y2", line=dict(color="#e8710a"))
        fig.add_scatter(x=hist.date, y=hist.relative_humidity_pct, name="Humidity (%)", yaxis="y2", line=dict(color="#7b1fa2", dash="dot"))
        if len(hist) > 3:
            fig.add_vline(x=hist.date.iloc[-4], line_dash="dash", line_color="gray", annotation_text="rain 3w ago (peak lag)", annotation_position="top left")
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Rainfall (mm)",
                          yaxis2=dict(overlaying="y", side="right", title="Temp °C / Humidity %"), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Historical vs predicted dengue trend")
    w = P[(P.ward_id == ward) & (P.date.between(date - pd.Timedelta(weeks=26), date + pd.Timedelta(weeks=26)))]
    fig = go.Figure()
    fig.add_scatter(x=w.date, y=w.hi, line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=w.date, y=w.lo, fill="tonexty", fillcolor="rgba(232,113,10,.2)", line=dict(width=0), name="80% interval")
    fig.add_scatter(x=w.date, y=w.pred, name="Predicted (window total)", line=dict(color="#e8710a"))
    fig.add_scatter(x=w.date, y=w.actual, name="Observed (window total)", line=dict(color="black", dash="dot"))
    fig.add_vline(x=date, line_color="red")
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Cases in window", legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")
    st.caption("Each point is a forecast issued that week for the following window; the dotted line is what was later reported. Red line = selected prediction date.")

    with st.expander("✅ Weather-lag pipeline check for this ward-week (per-ward lags, verified live against raw data)"):
        hw = df[df.ward_id == ward].reset_index(drop=True)
        i = int(hw.index[hw.date == date][0])
        rows = []
        for k_ in range(1, 9):
            for col, short in (("rainfall_mm", "rain"), ("avg_temp_c", "temp"), ("relative_humidity_pct", "hum")):
                src = hw.loc[i - k_, col] if i - k_ >= 0 else np.nan
                val = hw.loc[i, f"{short}_lag_{k_}w"]
                rows.append({"feature": f"{short}_lag_{k_}w", "source week": str(hw.loc[i - k_, "date"].date()) if i - k_ >= 0 else "-",
                             "feature value": val, "raw value in source week": src, "match": bool(np.isclose(val, src, equal_nan=True))})
        lt = pd.DataFrame(rows)
        a_, b_ = st.columns([2, 1])
        a_.dataframe(lt.round(2), hide_index=True, width="stretch", height=300)
        b_.metric("Lag features verified", f"{int(lt['match'].sum())}/{len(lt)}")
        b_.caption("Rolling sums/means (2/3/4/6/8 weeks) are trailing windows ending at the prediction week and are computed per ward_id only.")
        lc = lag_corr()
        fig = px.bar(lc, x="lag_weeks", y="correlation", labels={"lag_weeks": "Rainfall lag (weeks before cases)", "correlation": "Correlation with weekly cases"})
        fig.update_layout(height=260, margin=dict(t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Correlation peaks at lag {int(lc.loc[lc.correlation.idxmax(), 'lag_weeks'])} weeks, matching the 7-14 day breeding cycle plus incubation (descriptive; the model is validated chronologically).")

# ================= 3. MAPS & RANKING =================
with tabs[2]:
    scope = st.radio("Scope", ["All cities", f"{city} only"], horizontal=True)
    snap = snap_all if scope == "All cities" else snap_all[snap_all.city == city]
    st.subheader(f"Ward ranking - {date.date()} - {window}")
    rk = snap.sort_values("pred", ascending=False).reset_index(drop=True)
    rk.insert(0, "rank", rk.index + 1)
    rk["vs city median"] = rk.pred - rk.groupby("city").pred.transform("median")
    st.dataframe(rk[["rank", "city", "ward_id", "pred", "lo", "hi", "tier", "alert", "vs city median"]].round(1), hide_index=True, width="stretch", height=300,
                 column_config={"pred": st.column_config.ProgressColumn("pred cases", min_value=0, max_value=float(max(rk.pred.max(), 1)), format="%.1f")})
    top15 = rk.head(15).copy()
    fig = px.bar(top15.iloc[::-1], x="pred", y="ward_id", orientation="h", color="tier", color_discrete_map=TIER_COLOR,
                 error_x=(top15.hi - top15.pred).iloc[::-1], labels={"pred": "Predicted cases (bar) with 80% upper bound", "ward_id": ""})
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), legend_title="")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Ward × week risk heatmap (forecast issued each week)")
    hm = P[(P.city == city) & P.date.between(date - pd.Timedelta(weeks=25), date)].dropna(subset=["pred"]).copy()
    hm["ti"] = hm.tier.map(TIER_IDX)
    piv, pv_pred = hm.pivot(index="ward_id", columns="date", values="ti"), hm.pivot(index="ward_id", columns="date", values="pred")
    colorscale = [[0, TIER_COLOR["Low"]], [.25, TIER_COLOR["Low"]], [.25, TIER_COLOR["Medium"]], [.5, TIER_COLOR["Medium"]],
                  [.5, TIER_COLOR["High"]], [.75, TIER_COLOR["High"]], [.75, TIER_COLOR["Critical"]], [1, TIER_COLOR["Critical"]]]
    fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, zmin=0, zmax=3, colorscale=colorscale, showscale=False, xgap=1, ygap=1,
                               customdata=pv_pred.values, hovertemplate="%{y}<br>%{x|%Y-%m-%d}<br>predicted %{customdata:.1f} cases<extra></extra>"))
    fig.update_layout(height=max(320, 22 * len(piv) + 80), margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Rows = wards of {city}, columns = the last 26 weekly forecasts up to the selected date. Green/yellow/orange/red = Low/Medium/High/Critical.")

    st.subheader("Geographic risk map")
    wc = ward_coordinates()
    if wc is not None:
        g = snap.merge(wc, on="ward_id", how="inner")
        fig = px.scatter_geo(g, lat="lat", lon="lon", color="tier", size="pred", hover_name="ward_id", color_discrete_map=TIER_COLOR,
                             category_orders={"tier": C.RISK_TIERS}, size_max=22)
        st.caption(f"Ward-level map from verified coordinates in `data/ward_coordinates.csv` ({len(g)} wards matched).")
    else:
        cs = city_summary(snap_all)
        fig = px.scatter_geo(cs, lat="lat", lon="lon", size="mean_pred_cases", color="share_high_critical", hover_name="city",
                             hover_data={"wards": True, "high_critical_wards": True, "watch_flag_wards": True, "lat": False, "lon": False,
                                         "mean_pred_cases": ":.1f", "share_high_critical": ":.0%"},
                             color_continuous_scale=["#2e9e5b", "#e6b800", "#e8710a", "#c62828"], range_color=(0, 1), size_max=38)
        st.caption("**City-level** map (approximate city-centre coordinates). The dataset contains no ward coordinates and its ward IDs "
                   "cannot be matched to verified ward boundaries, so a ward-level geographic heatmap is **not** drawn - "
                   "see the heatmap above and the ranking. To enable it, add `data/ward_coordinates.csv` (ward_id, lat, lon) from a verified source. "
                   "The basemap outline needs internet; markers do not.")
    fig.update_geos(fitbounds="locations", showcountries=True, showland=True, landcolor="#f3f7f5", countrycolor="#9db3ad", resolution=50)
    fig.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

# ================= 4. WHAT-IF =================
with tabs[3]:
    st.subheader("What-if weather scenario")
    st.caption(f"Edit the last {N_HISTORY} weeks of weather for **{ward}** and re-score with the same serialized model and the same lag code "
               "(tested to be identical to batch inference). **Scenario inputs are hypothetical - not observed and not a weather forecast.**")
    base_h = ward_history(df, ward, date)
    s1, s2, s3, s4 = st.columns(4)
    rm = s1.slider("Rainfall multiplier", 0.0, 3.0, 1.0, 0.1)
    ts_ = s2.slider("Temperature shift (°C)", -5.0, 5.0, 0.0, 0.5)
    hs = s3.slider("Humidity shift (pts)", -20.0, 20.0, 0.0, 1.0)
    nw = s4.slider("Apply to last N weeks", 1, N_HISTORY, 4)
    scen = apply_scenario(base_h, rm, ts_, hs, nw)
    edited = st.data_editor(scen[["date", "rainfall_mm", "avg_temp_c", "relative_humidity_pct"]], hide_index=True, disabled=["date"],
                            width="stretch", key=f"ed_{ward}_{date}_{rm}_{ts_}_{hs}_{nw}")
    try:
        sc_in = scen.assign(**{c: edited[c].values for c in C.WEATHER_COLS})
        unusual = unusual_observations(sc_in, df)
        for u in unusual:
            st.warning("Unusual input: " + u + " - the model would extrapolate.")
        res_b, res_s = predict_history(base_h, window, B), predict_history(sc_in, window, B)
        m1, m2, m3 = st.columns(3)
        m1.metric("Baseline (observed weather)", f"{res_b['pred']:.1f}", res_b["tier"], delta_color="off")
        m2.metric("Scenario prediction", f"{res_s['pred']:.1f}", f"{res_s['pred'] - res_b['pred']:+.1f} cases", delta_color="inverse")
        m3.markdown(f"**Scenario tier**{badge(res_s['tier'])}", unsafe_allow_html=True)
        st.caption(f"Scenario 80% interval: {res_s['lo']:.0f} - {res_s['hi']:.0f} cases ({window}).")
        figs = go.Figure()
        for nm, hh, colr in (("Observed", base_h, "#4a90d9"), ("Scenario", sc_in, "#c62828")):
            figs.add_bar(x=hh.date, y=hh.rainfall_mm, name=f"{nm} rainfall (mm)", marker_color=colr)
        figs.update_layout(barmode="group", height=260, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(figs, width="stretch")
    except ValueError as e:
        st.error(f"Invalid scenario input: {e}")

# ================= 5. LIVE WEATHER =================
with tabs[4]:
    st.subheader("Optional live weather ingestion (Open-Meteo)")
    st.markdown("The app is fully functional offline. This panel makes **one** HTTP request per city, cached for 6 h on disk and in memory, and "
                "scores the wards of the selected city from the last complete weeks of **recent observed/analysis** weather. "
                "It never mixes *forecast* days into the lag inputs. **Experimental:** the model was trained on the supplied (apparently synthetic) "
                "dataset, city-level weather is applied to every ward, and nothing here has been validated against real cases.")
    if st.button(f"Fetch latest weather for {city}", type="primary"):
        with st.spinner("Contacting Open-Meteo (1 request; skipped if cached < 6 h)..."):
            st.session_state.setdefault("live_res", {})[city] = fetch_city_weather(city)   # failures are not memoised
    if city in st.session_state.get("live_res", {}):
        res = st.session_state["live_res"][city]
        if not res.ok:
            st.warning(res.message)
        else:
            a, b_, c_ = st.columns(3)
            a.metric("Source", res.source); b_.metric("Latest observed day", res.freshness.get("latest_observation", "?"),
                                                      f"{res.freshness.get('age_days', '?')} days old", delta_color="off")
            c_.metric("Complete weeks available", len(res.weekly))
            for n_ in res.notes:
                (st.warning if "STALE" in n_ or "unavailable" in n_ else st.caption)(n_)
            if res.freshness.get("stale"):
                st.error("Data is stale - predictions below are not up to date.")
            lw = res.weekly.tail(N_HISTORY)
            st.markdown("**Weekly inputs fed to the lag features** (provenance: observed/analysis, Monday-start weeks; rain = weekly sum, temp/humidity = weekly mean)")
            st.dataframe(lw[["date", "rainfall_mm", "avg_temp_c", "relative_humidity_pct"]].round(1), hide_index=True, width="stretch")
            if len(lw) < N_HISTORY:
                st.error(f"Only {len(lw)} complete weeks available; {N_HISTORY} are required for the lag features.")
            else:
                out = []
                for wid in sorted(df[df.city == city].ward_id.unique()):
                    wr = df[df.ward_id == wid].iloc[-1]
                    hist, notes = history_for_scoring(res, wr)
                    for u in unusual_observations(hist, df)[:1]:
                        st.warning(u)
                    r = predict_history(hist, window, B)
                    out.append({"ward_id": wid, "pred cases": r["pred"], "80% lo": r["lo"], "80% hi": r["hi"], "tier": r["tier"], "watch flag": r["alert"]})
                st.markdown(f"**Experimental ward scores for {city}** using live city-level weather (as of week starting {lw.date.iloc[-1].date()}; {window})")
                st.dataframe(pd.DataFrame(out).sort_values("pred cases", ascending=False).round(1), hide_index=True, width="stretch")
                st.caption("These are model outputs on a different weather source than the training data (distribution shift) - treat as a technical demonstration of the live pipeline, not as an operational alert.")
    else:
        st.info("Press the button to fetch. If you are offline, nothing is lost: set `LIVE_WEATHER_OFFLINE=1` to disable the network entirely.")

# ================= 6. VALIDATION =================
with tabs[5]:
    t = M["test_by_window"][window]
    st.subheader(f"Out-of-sample test performance - {window} ({M['split']['test'][0]} → {M['split']['test'][1]}, n={t['n']})")
    k = st.columns(6)
    k[0].metric("MAE (cases)", f"{t['MAE']:.2f}", f"seasonal-naive {t['baselines']['Seasonal-naive (52w earlier)']['MAE']:.2f}", delta_color="off")
    k[1].metric("RMSE", f"{t['RMSE']:.2f}"); k[2].metric("R²", f"{t['R2']:.3f}")
    k[3].metric("High/Crit recall", f"{t['HC_recall']:.0%}"); k[4].metric("High/Crit precision", f"{t['HC_precision']:.0%}")
    k[5].metric("High/Crit PR-AUC", f"{t['HC_PR_AUC']:.2f}" if t.get("HC_PR_AUC") is not None else "n/a")
    st.caption(f"High/Critical F1 {t['HC_F1']:.2f}, false-alarm rate {t['HC_false_alarm_rate']:.1%}. Tier exact accuracy {t['tier_exact_accuracy']:.0%}, within one tier {t['tier_within_one']:.0%}. "
               f"80% interval empirical coverage **{t['interval_coverage']:.0%}** at mean width {t['interval_width']:.1f} (over-covers: conservative).")
    bl = pd.DataFrame({"Model": ["**ML model (HistGB-Poisson)**"] + list(t["baselines"]),
                       "MAE": [t["MAE"]] + [v["MAE"] for v in t["baselines"].values()], "RMSE": [t["RMSE"]] + [v["RMSE"] for v in t["baselines"].values()],
                       "R²": [t["R2"]] + [v["R2"] for v in t["baselines"].values()],
                       "High/Crit recall": [t["HC_recall"]] + [v["HC_recall"] for v in t["baselines"].values()]})
    st.dataframe(bl.round(3), hide_index=True, width="stretch")

    st.markdown("#### Early-warning operating points (High/Critical outbreak detection)")
    if "ALERT_recall" in t:
        op = pd.DataFrame([
            {"operating point": f"Tier ≥ High (predicted ≥ {cuts[1]:.0f} cases)", "recall": t["HC_recall"], "precision": t["HC_precision"], "false-alarm rate": t["HC_false_alarm_rate"]},
            {"operating point": f"Watch flag (predicted ≥ {t['ALERT_alert_cut']:.1f} cases)", "recall": t["ALERT_recall"], "precision": t["ALERT_precision"], "false-alarm rate": t["ALERT_false_alarm_rate"]}])
        st.dataframe(op.round(3), hide_index=True, width="stretch")
        va_far = M["alert_threshold_selection"][window]
        st.caption(f"The watch-flag threshold was chosen on **validation only** (max F2 with false-alarm cap {C.ALERT_MAX_FALSE_ALARM:.0%}). On the test period the false-alarm rate "
                   f"came out at {t['ALERT_false_alarm_rate']:.0%} - above that cap - because outbreak levels differ between periods; thresholds do not transfer perfectly.")
    c1, c2 = st.columns(2)
    if window == C.MAIN_WINDOW:
        with c1:
            pr = report_csv("pr_curve_test.csv")
            fig = px.line(pr, x="recall", y="precision", title="Precision-recall: High/Critical (test)")
            fig.add_scatter(x=[t["HC_recall"]], y=[t["HC_precision"]], mode="markers", marker=dict(size=12, color="#c62828"), name="Tier ≥ High")
            fig.add_scatter(x=[t["ALERT_recall"]], y=[t["ALERT_precision"]], mode="markers", marker=dict(size=12, color="#0f766e", symbol="diamond"), name="Watch flag")
            fig.update_layout(height=330, yaxis_range=[0, 1.02])
            st.plotly_chart(fig, width="stretch")
        with c2:
            cal = report_csv("interval_calibration_test.csv")
            fig = go.Figure()
            fig.add_bar(x=[f"{m:.1f}" for m in cal.mean_pred], y=cal.coverage, name="Empirical coverage", marker_color="#0f766e")
            fig.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="nominal 80%")
            fig.update_layout(title="Interval calibration by predicted-risk level (test)", height=330, yaxis_range=[0, 1.05],
                              xaxis_title="mean predicted cases in bin (quintiles)", yaxis_title="coverage")
            st.plotly_chart(fig, width="stretch")
        tp = test_preds()
        monthly = tp.assign(m=tp.date.dt.to_period("M").astype(str), ae=(tp.actual - tp.pred).abs(), se=(tp.actual - tp.pred) ** 2) \
            .groupby("m").agg(MAE=("ae", "mean"), se=("se", "mean")).reset_index()
        monthly["RMSE"] = np.sqrt(monthly.se)
        c1, c2 = st.columns(2)
        fig = px.line(monthly.melt("m", ["MAE", "RMSE"]), x="m", y="value", color="variable", markers=True,
                      title="Chronological error by forecast month (test period)", labels={"m": "", "value": "cases"})
        c1.plotly_chart(fig, width="stretch")
        s = tp.sample(min(1500, len(tp)), random_state=0)
        fig = px.scatter(s, x="actual", y="pred", opacity=.4, title="Actual vs predicted (test)")
        mx = float(max(s.actual.max(), s.pred.max()))
        fig.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx, line=dict(dash="dash", color="red"))
        c2.plotly_chart(fig, width="stretch")
        cA, cB = st.columns(2)
        cdf = report_csv("test_by_city.csv")
        cA.markdown("**By city (test)**")
        cA.dataframe(cdf[["city", "n", "mean_actual", "MAE", "RMSE", "R2", "relative_MAE", "HC_recall", "HC_precision", "interval_coverage", "unreliable_flag"]].round(3), hide_index=True, width="stretch")
        sdf = report_csv("test_by_season.csv")
        cB.markdown("**By season (origin month)**")
        cB.dataframe(sdf[["season", "n", "mean_actual", "MAE", "RMSE", "R2", "HC_recall", "HC_precision", "HC_false_alarm_rate", "interval_coverage"]].round(3), hide_index=True, width="stretch")
        st.caption("Dry/post-monsoon seasons have almost no cases, so recall/precision are undefined there; monsoon error is what matters operationally.")
        wdf = report_csv("test_by_ward.csv")
        st.markdown("**Ward-level performance (test; 75 forecasts per ward)**")
        st.dataframe(wdf[["ward", "n", "mean_actual", "MAE", "RMSE", "R2", "relative_MAE", "interval_coverage"]].sort_values("MAE", ascending=False).round(3), hide_index=True, width="stretch", height=260)
    else:
        st.info("Detailed city/season/ward slices, PR curve and calibration are produced for the main window. Select 'Weeks 2-4 ahead (total)' in the sidebar.")
    c1, c2 = st.columns(2)
    cv = report_csv("rolling_origin_cv.csv")
    c1.markdown("**Rolling-origin chronological CV** (train before year, test on that year)")
    c1.dataframe(cv.round(3), hide_index=True, width="stretch")
    imp = pd.Series(M["grouped_importance_val_MAE_increase"]).sort_values()
    fig = px.bar(imp, orientation="h", title="Grouped permutation importance (validation, ΔMAE)", labels={"value": "MAE increase when group shuffled", "index": ""})
    fig.update_layout(showlegend=False, height=300)
    c2.plotly_chart(fig, width="stretch")
    cm = pd.DataFrame(t["tier_confusion"], index=[f"actual {x}" for x in C.RISK_TIERS], columns=[f"pred {x}" for x in C.RISK_TIERS])
    st.markdown("**Risk-tier confusion matrix (test)**")
    st.dataframe(cm, width="stretch")
    st.markdown("**Model comparison on the validation set (main window)**")
    st.dataframe(report_csv("model_comparison_validation.csv").round(3), hide_index=True, width="stretch")
    st.markdown("**Forecast-strategy comparison (validation, single-week targets)**")
    hs_ = report_csv("horizon_strategy_validation.csv")
    st.dataframe(hs_.round(3), hide_index=True, width="stretch")
    st.caption("Direct per-horizon models had the lowest mean validation MAE and are what the app uses (pooled model was a close second).")

# ================= 7. METHODOLOGY =================
with tabs[6]:
    a_ = audit()
    st.markdown(f"""
**Model:** {B['model_name']} on `{B['feature_set']}` features ({len(B['features'])} inputs): current weather, per-ward 1-8 week lags and trailing 2-8 week rolling
sums/means of rainfall, temperature and humidity, ward vulnerability attributes (density, water-logging, sanitation) and seasonality. No reported-case history is needed,
so wards without surveillance data can be scored. One direct model per horizon (week 2, 3, 4 and the 2-4 week total).

**Timing / leakage control:** the forecast for week *t* only uses weather observed up to week *t*. Split: train ≤ 2022, validation 2023-mid 2024, test mid-2024+, with a 4-week purge
gap; model, features, tier percentiles, interval method and alert threshold are selected on validation only; the test period is scored once. A shuffled-target placebo gives
R² = {a_['placebo_shuffled_target_val']['R2']:.2f} (no leakage), and past rain correlates most with cases at a {a_['peak_past_lag_weeks']}-week lag.

**Limitations (please read)**
- The dataset **looks synthetic** (validation R² {a_['full_weather_model_val']['R2']:.2f}, clean lag peak, strongly autocorrelated residuals). Results show the method works on this data; they are **not** evidence of real-world accuracy or operational validity.
- Risk tiers are percentiles of the training case distribution - **not** official public-health alert levels. The watch flag is a sensitive screening threshold with many false alarms.
- Intervals (quantile GBMs + split-conformal calibration) are conservative (≈94% coverage vs 80% nominal) and carry no guarantee under distribution shift.
- Driver explanations are what-if occlusions versus seasonal-normal conditions, not causal effects.
- No ward coordinates exist in the data, so there is no ward-level geographic map (a verified `data/ward_coordinates.csv` enables it). The city map uses approximate city centres.
- The live-weather panel is experimental (different weather source, city-level weather for all wards, unvalidated).

**AI & external services:** Claude (Anthropic) assisted with code review, refactoring, tests and documentation. Runtime: **no LLM calls**; the only optional external service is Open-Meteo (weather, no key), used only when you press the Live button.
Full disclosure in `README.md`.
""")
