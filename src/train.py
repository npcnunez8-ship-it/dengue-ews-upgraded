"""Train / validate / select / serialize. Run:  python -m src.train"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import cohen_kappa_score
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import add_target, build_dataset, feature_groups, feature_sets
from .risk import fit_cutpoints, to_tier_idx
from .evaluate import (slice_report, core_metrics, SEASONS, season_of, pick_alert_cut, pr_curve_frame,
                       calibration_by_level)
from .model_io import save_bundle, versions

try:  # optional, comparison only (not needed at inference)
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

SEED = 42
ALPHA = 0.10   # 80% prediction interval (quantiles 0.10 / 0.90)


def make_model(name):
    if name == "Mean baseline":
        return DummyRegressor(strategy="mean")
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.5,
                                   n_jobs=-1, random_state=SEED)
    if name == "HistGB (squared)":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=24,
                                             min_samples_leaf=30, l2_regularization=1.0, random_state=SEED)
    if name == "HistGB (Poisson)":
        return HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.05,
                                             max_leaf_nodes=24, min_samples_leaf=30,
                                             l2_regularization=1.0, random_state=SEED)
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=SEED)
    if name == "XGBoost (Poisson)":
        return XGBRegressor(objective="count:poisson", n_estimators=400, learning_rate=0.04, max_depth=5,
                            subsample=0.8, colsample_bytree=0.7, min_child_weight=5, n_jobs=2, random_state=SEED)
    raise KeyError(name)


def model_list():
    base = ["Ridge", "RandomForest", "ExtraTrees", "HistGB (squared)", "HistGB (Poisson)"]
    return base + (["XGBoost (Poisson)"] if XGBRegressor is not None else [])


def metrics(y, p):
    p = np.maximum(p, 0)
    return {"MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": float(r2_score(y, p))}


def split_masks(df, target):
    gap = pd.Timedelta(weeks=C.MAX_HORIZON)
    te, ve = pd.Timestamp(C.TRAIN_END), pd.Timestamp(C.VAL_END)
    ok = target.notna()
    tr = ok & (df["date"] <= te - gap)
    va = ok & (df["date"] > te) & (df["date"] <= ve - gap)
    ts = ok & (df["date"] > ve)
    return tr, va, ts


def grouped_importance(model, X, y, groups, repeats=3):
    rng = np.random.default_rng(SEED)
    base = mean_absolute_error(y, np.maximum(model.predict(X), 0))
    out = {}
    for g, cols in groups.items():
        deltas = []
        for _ in range(repeats):
            Xp = X.copy()
            idx = rng.permutation(len(X))
            Xp[cols] = X[cols].values[idx]
            deltas.append(mean_absolute_error(y, np.maximum(model.predict(Xp), 0)) - base)
        out[g] = float(np.mean(deltas))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def fit_quantiles(X, y):
    lo = HistGradientBoostingRegressor(loss="quantile", quantile=ALPHA / 2, max_iter=200, learning_rate=0.05,
                                       max_leaf_nodes=16, min_samples_leaf=30, random_state=SEED).fit(X, y)
    hi = HistGradientBoostingRegressor(loss="quantile", quantile=1 - ALPHA / 2, max_iter=200, learning_rate=0.05,
                                       max_leaf_nodes=16, min_samples_leaf=30, random_state=SEED).fit(X, y)
    return lo, hi


def cqr_qhat(lo, hi, X, y):
    """Conformalized quantile regression (Romano et al. 2019) calibration constant."""
    s = np.maximum(lo.predict(X) - y, y - hi.predict(X))
    n = len(y)
    level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
    return float(np.quantile(s, level))


def interval(lo, hi, X, qhat):
    l, h = lo.predict(X) - qhat, hi.predict(X) + qhat
    l = np.maximum(l, 0)
    return l, np.maximum(h, l)


TIER_CANDIDATES = [(0.50, 0.80, 0.95), (0.60, 0.85, 0.95), (0.70, 0.90, 0.97), (0.60, 0.80, 0.90)]
INTERVAL_TARGET = 1 - ALPHA


def scaled_q(p, y):
    """Locally-scaled split-conformal constant: |y-p| / sqrt(p+1) (count-like heteroscedasticity)."""
    s = np.abs(y - p) / np.sqrt(p + 1.0)
    n = len(y)
    return float(np.quantile(s, min(1.0, np.ceil((n + 1) * INTERVAL_TARGET) / n)))


def scaled_interval(p, q):
    h = q * np.sqrt(p + 1.0)
    return np.maximum(p - h, 0), p + h


def choose_interval_method(m_tr, lo, hi, Xv, yv, dates_v):
    """Chronological check inside validation: calibrate on the first half, test on the later half."""
    order = np.argsort(dates_v.values, kind="stable")
    half = len(order) // 2
    a, b = order[:half], order[half:]
    p = np.maximum(m_tr.predict(Xv), 0)
    res = {}
    qc = cqr_qhat(lo, hi, Xv.iloc[a], yv.iloc[a])
    l, h = interval(lo, hi, Xv.iloc[b], qc)
    res["cqr"] = (float(((yv.iloc[b] >= l) & (yv.iloc[b] <= h)).mean()), float((h - l).mean()))
    qs = scaled_q(p[a], yv.iloc[a].values)
    l, h = scaled_interval(p[b], qs)
    res["scaled"] = (float(((yv.iloc[b].values >= l) & (yv.iloc[b].values <= h)).mean()), float((h - l).mean()))
    ok = {k: v for k, v in res.items() if v[0] >= INTERVAL_TARGET - 0.03}
    pick = min(ok, key=lambda k: ok[k][1]) if ok else "cqr"
    return pick, res


def window_interval(bw, X):
    p = np.maximum(bw["model"].predict(X), 0)
    if bw["interval_method"] == "scaled":
        return (p, *scaled_interval(p, bw["qhat"]))
    l, h = interval(bw["lo"], bw["hi"], X, bw["qhat"])
    return p, l, h


def pick_tiers(y_tr, y_va, p_va):
    best, rows = None, []
    for qs in TIER_CANDIDATES:
        cuts = fit_cutpoints(y_tr, qs)
        kap = cohen_kappa_score(to_tier_idx(y_va, cuts), to_tier_idx(p_va, cuts), weights="quadratic")
        rows.append({"quantiles": list(qs), "cutpoints": cuts, "val_quadratic_kappa": float(kap)})
        if best is None or kap > best[0]:
            best = (kap, qs)
    return best[1], rows


def _finish_bundle(bundle, df, cols, trva, tr, va, split_info):
    """Seasonal-normal reference (train+val only), metadata, portable serialization."""
    wk = df["date"].dt.isocalendar().week.astype(int).clip(upper=52)
    bundle["climatology"] = df.loc[trva, cols].groupby(wk[trva]).mean()
    bundle["static_ref"] = df.loc[trva, cols].mean()
    bundle["meta"] = {"trained_on": f"{split_info['train'][0]}..{split_info['val'][1]}", "seed": SEED}
    save_bundle(bundle)


def main(bundle_only=False):
    """Full experiment (default) or `bundle_only`: refit + recalibrate the already-selected model only
    (no model comparison, no test scoring, no report files) - used to rebuild the bundle on a new machine."""
    t0 = time.time()
    df = build_dataset(with_history=True)
    fsets = feature_sets(df)
    gdf = df.groupby("ward_id", sort=False)
    y_main = add_target(df, C.WINDOWS[C.MAIN_WINDOW])
    tr, va, ts = split_masks(df, y_main)
    fs_all = sorted(set(sum(fsets.values(), [])))
    okf = df[fs_all].notna().all(axis=1)
    # identical evaluation samples for ML models AND baselines: seasonal-naive needs the target 52 weeks earlier
    snaive_main = y_main.groupby(df.ward_id, sort=False).shift(52)
    ev = okf & snaive_main.notna()
    tr, va, ts = tr & okf, va & ev, ts & ev
    print(f"rows train/val/test = {tr.sum()}/{va.sum()}/{ts.sum()}")
    d_ = lambda m: [str(df.loc[m, 'date'].min().date()), str(df.loc[m, 'date'].max().date()), int(m.sum())]
    split_info = {"train": d_(tr), "val": d_(va), "test": d_(ts), "purge_gap_weeks": C.MAX_HORIZON,
                  "note": "val/test evaluation rows additionally require a seasonal-naive reference (target 52 weeks earlier)"}

    if bundle_only:
        try:
            sel = json.loads((C.REPORT_DIR / "metrics.json").read_text())["selected"]
            model_name, feat_name, use_hist = sel["model"], sel["features"], False
        except Exception:  # noqa: BLE001
            model_name, feat_name, use_hist = "HistGB (Poisson)", "weather_only", False
        cols = fsets[feat_name]
        if model_name.startswith("XGBoost") and XGBRegressor is None:
            model_name = "HistGB (Poisson)"
    else:
        # ---- 1. Baselines + model x feature-set comparison (validation only, identical samples) ----
        nh = len(C.WINDOWS[C.MAIN_WINDOW])
        rows = []
        def add(feat, model, y, p):
            r = metrics(y, p); rows.append({"features": feat, "model": model, "n_val": int(len(y)), **r})
            print(f"{feat:22s} {model:20s} MAE={r['MAE']:.3f} RMSE={r['RMSE']:.3f} R2={r['R2']:.3f}")
        yv = y_main[va]
        add("-", "Mean baseline", yv, np.full(len(yv), y_main[tr].mean()))
        add("-", "Persistence (cases_t x horizons)", yv, df.loc[va, "dengue_cases"].values * nh)
        add("-", "Seasonal-naive (52w earlier)", yv, snaive_main[va].values)
        for fname, cols in fsets.items():
            for mname in model_list():
                if mname == "RandomForest" and fname != "weather_only":
                    continue
                m = make_model(mname).fit(df.loc[tr, cols], y_main[tr])
                add(fname, mname, yv, m.predict(df.loc[va, cols]))
        comp = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

        ml = comp[comp.features != "-"]
        def pick(fname):
            c = ml[ml.features == fname]
            top, core = c.iloc[0], c[c.model != "XGBoost (Poisson)"].iloc[0]
            # parsimony: XGBoost adds a runtime dependency, so it must beat the best sklearn model by >1%
            return top if (top.model != "XGBoost (Poisson)" or top.MAE < 0.99 * core.MAE) else core
        wo, wh = pick("weather_only"), pick("weather_plus_history")
        use_hist = wh.MAE < 0.97 * wo.MAE      # history only if >3% better on validation
        best = wh if use_hist else wo
        feat_name, model_name = best.features, best.model
        cols = fsets[feat_name]
        print(f"\nSELECTED: {model_name} on {feat_name} (val MAE {best.MAE:.3f}); history adopted={use_hist}")


    # ---- 2. Per-window bundles ----
    bundle = {"features": cols, "feature_set": feat_name, "model_name": model_name,
              "windows": {}, "alpha": ALPHA, "groups": feature_groups(cols)}
    test_report, window_val, tier_sel, interval_sel, slices, alert_sel = {}, {}, {}, {}, {}, {}
    for wname, hz in C.WINDOWS.items():
        y = add_target(df, hz)
        sn = y.groupby(df.ward_id, sort=False).shift(52)
        trw, vaw, tsw = (m & y.notna() for m in (tr, va, ts))
        m_tr = make_model(model_name).fit(df.loc[trw, cols], y[trw])
        lo, hi = fit_quantiles(df.loc[trw, cols], y[trw])
        Xv, yvw = df.loc[vaw, cols], y[vaw]
        pv = np.maximum(m_tr.predict(Xv), 0)
        window_val[wname] = metrics(yvw, pv)
        meth, meth_res = choose_interval_method(m_tr, lo, hi, Xv, yvw, df.loc[vaw, "date"])
        qhat = cqr_qhat(lo, hi, Xv, yvw) if meth == "cqr" else scaled_q(pv, yvw.values)
        interval_sel[wname] = {"chosen": meth, "val_second_half_(coverage,width)": meth_res}
        qs, tier_rows = pick_tiers(y[trw].values, yvw.values, pv)
        cuts = fit_cutpoints(y[trw].values, qs)
        tier_sel[wname] = {"chosen_quantiles": list(qs), "candidates": tier_rows}
        alert_cut, alert_rows = pick_alert_cut(yvw.values, pv, cuts[1])
        alert_sel[wname] = {"chosen_alert_cut": alert_cut, "high_cut": cuts[1], "validation_grid": alert_rows}
        tv = trw | vaw
        m_f = make_model(model_name).fit(df.loc[tv, cols], y[tv])
        lo_f, hi_f = fit_quantiles(df.loc[tv, cols], y[tv])
        bw = {"horizons": hz, "model": m_f, "lo": lo_f, "hi": hi_f, "qhat": qhat,
              "interval_method": meth, "cutpoints": cuts, "tier_quantiles": list(qs), "alert_cut": alert_cut}
        bundle["windows"][wname] = bw
        if bundle_only:
            continue
        # ---- test (touched once per window, after all selection) ----
        Xt, yt = df.loc[tsw, cols], y[tsw]
        pt, l, h = window_interval(bw, Xt)
        meta = df.loc[tsw, ["date", "city", "ward_id"]]
        rep = slice_report(meta, yt, pt, l, h, cuts, alert_cut)
        a_t, p_t = to_tier_idx(yt, cuts), to_tier_idx(pt, cuts)
        base = {"Mean baseline": np.full(len(yt), y[trw].mean()),
                "Persistence (cases_t x horizons)": df.loc[tsw, "dengue_cases"].values * len(hz),
                "Seasonal-naive (52w earlier)": sn[tsw].values}
        rep["overall"]["baselines"] = {k: core_metrics(yt, v, cuts=cuts) for k, v in base.items()}
        rep["overall"].update({"tier_exact_accuracy": float((a_t == p_t).mean()),
                               "tier_within_one": float((np.abs(a_t - p_t) <= 1).mean()),
                               "tier_confusion": pd.crosstab(pd.Series(a_t, name="a"), pd.Series(p_t, name="p"))
                               .reindex(index=range(4), columns=range(4), fill_value=0).values.tolist(),
                               "cutpoints": cuts, "tier_quantiles": list(qs), "interval_method": meth})
        test_report[wname] = rep["overall"]
        slices[wname] = rep
        o = rep["overall"]
        print(f"[{wname}] TEST n={o['n']} MAE={o['MAE']:.3f} RMSE={o['RMSE']:.3f} R2={o['R2']:.3f} "
              f"cov={o['interval_coverage']:.3f} w={o['interval_width']:.1f} HC rec/prec/F1={o['HC_recall']:.2f}/"
              f"{o['HC_precision']:.2f}/{o['HC_F1']:.2f} FAR={o['HC_false_alarm_rate']:.3f} [{meth}; tiers {qs}]")
        if wname == C.MAIN_WINDOW:
            tdf = meta.copy()
            tdf["actual"], tdf["pred"], tdf["lo"], tdf["hi"] = yt.values, pt, l, h
            tdf["persistence"], tdf["seasonal_naive"] = base["Persistence (cases_t x horizons)"], base["Seasonal-naive (52w earlier)"]
            tdf.to_csv(C.REPORT_DIR / "test_predictions.csv", index=False)
            pr_curve_frame(yt, pt, cuts[1]).to_csv(C.REPORT_DIR / "pr_curve_test.csv", index=False)
            calibration_by_level(yt, pt, l, h).to_csv(C.REPORT_DIR / "interval_calibration_test.csv", index=False)
            pd.DataFrame(alert_rows).to_csv(C.REPORT_DIR / "alert_threshold_validation.csv", index=False)
            for k in ("city", "season", "ward"):
                rep[k].to_csv(C.REPORT_DIR / f"test_by_{k}.csv", index=False)
            groups = feature_groups(cols)
            imp = grouped_importance(m_tr, Xv.reset_index(drop=True), yvw.reset_index(drop=True), groups)

    if bundle_only:
        _finish_bundle(bundle, df, cols, tr | va, tr, va, split_info)
        print(f"bundle rebuilt in {time.time()-t0:.0f}s ({model_name} on {feat_name})")
        return

    # ---- 3. Rolling-origin chronological CV ----
    folds = []
    for yr in [2021, 2022, 2023, 2024]:
        sdt, edt = pd.Timestamp(f"{yr}-01-01"), pd.Timestamp(f"{yr}-12-31")
        trn = y_main.notna() & okf & (df.date <= sdt - pd.Timedelta(weeks=4))
        tst = y_main.notna() & ev & (df.date >= sdt) & (df.date <= edt)
        m = make_model(model_name).fit(df.loc[trn, cols], y_main[trn])
        pp = m.predict(df.loc[tst, cols])
        folds.append({"fold": f"train<{yr} -> test {yr}", "n": int(tst.sum()), **metrics(y_main[tst], pp),
                      "persistence_MAE": metrics(y_main[tst], df.loc[tst, "dengue_cases"].values * nh)["MAE"],
                      "seasonal_naive_MAE": metrics(y_main[tst], snaive_main[tst].values)["MAE"]})
    cv = pd.DataFrame(folds)
    print(cv.round(3))

    _finish_bundle(bundle, df, cols, tr | va, tr, va, split_info)

    # ---- 5. Reports ----
    comp.to_csv(C.REPORT_DIR / "model_comparison_validation.csv", index=False)
    cv.to_csv(C.REPORT_DIR / "rolling_origin_cv.csv", index=False)
    summary = {"split": split_info, "selected": {"model": model_name, "features": feat_name,
               "history_adopted": bool(use_hist), "n_features": len(cols)},
               "validation_by_window": window_val, "test_by_window": test_report,
               "interval_method_selection": interval_sel, "alert_threshold_selection": alert_sel,
               "environment": versions(), "tier_threshold_selection": tier_sel,
               "city_flags": slices[C.MAIN_WINDOW]["city"][["city", "relative_MAE", "unreliable_flag"]].to_dict("records"),
               "grouped_importance_val_MAE_increase": imp, "train_seconds": round(time.time() - t0, 1)}
    (C.REPORT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train / validate / serialize the dengue early-warning model")
    ap.add_argument("--bundle-only", action="store_true",
                    help="only refit + recalibrate the already selected model (fast; no reports)")
    main(bundle_only=ap.parse_args().bundle_only)
