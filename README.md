# 🦟 Dengue Early-Warning System (Dhaka North/South · Chattogram · Gazipur · Narayanganj)

Ward-level dengue outbreak risk **2-4 weeks ahead** from multi-week weather lags, with Low / Medium / High / Critical tiers, 80% prediction intervals,
driver explanations, a what-if simulator and a chronological-validation dashboard (EDU AI Engineering Hackathon 2026, "AI for Urban Dengue Outbreak").
Runs **fully offline**: no API key, no LLM calls. An optional live-weather panel (Open-Meteo) is available but never required.

## Quick start (≈ 2 minutes)
```bash
git clone <YOUR_REPO_URL> && cd <repo-folder>
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # ~25 s
streamlit run app.py                 # opens http://localhost:8501
```
The trained model (`models/dengue_bundle.joblib`) and all reports are committed, so the app starts immediately.
**If your scikit-learn version differs** from the one that built the model (or the file is unreadable), the app detects it and rebuilds the model from `data/` automatically (~25 s, first launch only) - no manual step. You can also do it explicitly: `python -m src.train --bundle-only`.
Exact versions used for the reported results: `requirements-lock.txt` (Python 3.12). Tested on a fresh virtualenv: install 25 s, tests 3 s.

## Commands
| Task | Command |
|---|---|
| Launch dashboard | `streamlit run app.py` |
| Run tests (32) | `python -m pytest -q tests` |
| Full retrain + all reports (~2 min) | `python -m src.train` |
| Rebuild only the model bundle (~25 s) | `python -m src.train --bundle-only` |
| Leakage audit + horizon-strategy + hyper-parameter check (~30 s) | `python -m src.audit` |
| Notebook (re-executes everything) | `pip install -r requirements-notebook.txt && jupyter nbconvert --to notebook --execute --inplace notebooks/dengue_ews_pipeline.ipynb` |
| Predict from Python | `python -c "from src.predict import *; print(predict_one(load_features(),'DNCC_Ward_01','2025-08-04'))"` |
| Offline guarantee for live panel | `LIVE_WEATHER_OFFLINE=1 streamlit run app.py` |

Optional environment variables (live panel only): `OPEN_METEO_BASE_URL`, `OPEN_METEO_API_KEY`, `LIVE_WEATHER_CACHE_HOURS` (default 6), `LIVE_WEATHER_TIMEOUT_S` (default 8), `LIVE_WEATHER_OFFLINE`.

## Dashboard (7 tabs)
1. **Overview** - executive KPIs (Critical/High/watch-flag wards), top-10 priority wards, tier mix by city, trust statement.
2. **Ward forecast** - separate **week 2 / 3 / 4** and 2-4 week-total forecasts with 80% intervals and tiers, driver breakdown, observed weather, historical vs predicted trend, and a live *lag-pipeline check* (every lag feature compared with its raw source week).
3. **Risk maps & ranking** - ward ranking with bars, ward×week risk heatmap, city-level geographic map (ward-level map activates only if verified coordinates are supplied).
4. **What-if** - edit the last 9 weeks of weather (rain multiplier, temperature/humidity shifts, per-week editing) and re-score through the same lag code and model.
5. **Live weather (optional)** - fetch recent observed weather for a city (1 cached request), freshness/staleness checks, experimental scoring.
6. **Validation** - test metrics, baselines, High/Critical PR curve and PR-AUC, alert operating points, interval calibration, error by month/city/season/ward, tier confusion matrix, rolling-origin CV, model and strategy comparison.
7. **Methodology & limits.**

## Project layout
```
app.py                  Streamlit dashboard
src/features.py         per-ward weather lags 1-8w, rolling 2-8w, seasonality, target builder
src/train.py            chronological split, baselines, model comparison, calibration, tiers, alert threshold, serialization (--bundle-only)
src/model_io.py         portable bundle (no pandas objects), version check, automatic rebuild
src/predict.py          fast batch inference + grouped driver explanations
src/scenario.py         what-if scoring from a raw 9-week weather history (same features + bundle)
src/quality.py          missing / unusual / stale input handling, dataset quality report
src/live_weather.py     optional Open-Meteo ingestion: cache, freshness, error handling, provenance labels
src/geo.py              city summary + optional verified ward coordinates (nothing invented)
src/evaluate.py         city / ward / season slices, High-Critical metrics incl. PR-AUC, alert threshold, calibration
src/audit.py            leakage audit, horizon-strategy comparison, chronological hyper-parameter check
tests/                  32 tests: lags, leakage, split, tiers, inference parity, bundle portability, live weather (mocked), alert threshold
notebooks/              executed, reproducible notebook      legacy/  the team's original notebook
models/ reports/        serialized model, metrics.json and CSVs   ARCHITECTURE.md  diagram, split, rationale, cost controls
```

## Methodology in brief
- **Weather lag pipeline:** `avg_temp_c`, `rainfall_mm`, `relative_humidity_pct` → current value, per-`ward_id` lags 1-8 weeks, trailing 2/3/4/6/8-week rolling sums (rain) or means (temp, humidity), week-of-year sine/cosine, plus ward attributes (density, water-logging, sanitation): 47 features.
- **Targets:** reported cases in weeks t+2…t+4 (total) and each of weeks 2, 3, 4 separately (one direct model each; direct beat pooled and derived strategies on validation).
- **Model:** HistGradientBoosting with Poisson loss, selected on chronological validation against Ridge, RandomForest, ExtraTrees, HistGB-squared (and XGBoost if installed).
- **Validation:** train ≤ 2022-11, validation 2023-01…2024-05, test 2024-07…2025-12, 4-week purge gap, identical rows for every model and baseline, test scored once. Timing: a forecast for week *t* uses only weather observed through week *t*.
- **Risk tiers** from training-distribution percentiles; **intervals** = quantile GBMs + conformal calibration; **explanations** = grouped what-if occlusion.

## Results (out-of-sample test, forecasts 2024-07 → 2025-12, 4,500 ward-weeks, 60 wards, 5 cities)
Window = cases in weeks 2-4 ahead. All methods are scored on identical rows.

| Model | MAE | RMSE | R² | High/Critical recall |
|---|---|---|---|---|
| **HistGB-Poisson, weather-only (ours)** | **2.62** | **4.91** | **0.81** | 69% |
| Seasonal-naive (same window 52 weeks earlier) | 4.03 | 8.12 | 0.49 | 66% |
| Persistence (current cases × 3) | 6.22 | 10.75 | 0.11 | 33% |
| Mean baseline | 9.57 | 11.38 | 0.00 | 0% |

- **High/Critical (≥ 27 cases in 3 weeks):** precision 55%, recall 69%, F1 0.61, false-alarm rate 6.0%, **PR-AUC 0.64**. Tier exact accuracy 86%, within one tier 99.6%.
- **Sensitive watch flag** (predicted ≥ 17.6 cases, chosen on validation): recall 99%, precision 35%, but false-alarm rate **19%** on test - above the 10% validation cap, because outbreak levels differ between periods. It is a screening flag, not a tier.
- **Single-week models (weeks 2 / 3 / 4):** R² 0.71 / 0.72 / 0.62; High/Critical recall 82% / 81% / 84%; PR-AUC 0.81 / 0.81 / 0.76.
- **80% prediction interval:** empirical coverage 94% at mean width 13.0 cases (conservative); coverage is ≥ 89% in every predicted-risk quintile.
- **By season:** monsoon (Jun-Sep) MAE 5.5, R² 0.52 - this is where error matters; dry and post-monsoon seasons have almost no cases.
- **Rolling-origin CV (2021-2024):** R² 0.80-0.91, MAE 1.9-3.1, beats seasonal-naive and persistence every year.
- **Ablations:** ward attributes + season alone → validation MAE 2.56; adding lagged weather → 1.83 (−29%). Case-history features did not help. Hyper-parameter search gained 0.3% (not adopted).

## Limitations and honesty notes (please read before trusting any number)
- The supplied dataset **looks synthetic** (validation R² 0.92, smooth lag-3 correlation peak, strongly autocorrelated residuals). Everything here shows the *method* works on this data; it is **not** medically or operationally validated and real-world accuracy must be re-established.
- Tiers are dataset-relative percentiles, **not official public-health alert levels**. Intervals are conservative and carry no guarantee under distribution shift. Driver explanations are what-if effects, not causal.
- Monsoon High/Critical recall of 69% means about 3 in 10 high-risk ward-weeks are missed at the tier threshold; the watch flag trades that for many more false alarms.
- **No ward-level geographic heatmap:** the dataset has no ward coordinates and its ward IDs cannot be matched to verified boundaries; we do not invent them. The city-level map uses approximate city centres. Add verified coordinates to `data/ward_coordinates.csv` (`ward_id,lat,lon`) and the ward map appears automatically.
- **Live weather is experimental and untested against the real API from the build sandbox** (network egress to Open-Meteo was unavailable; the module is unit-tested with mocked responses, and API variable names were checked against public Open-Meteo documentation). It supplies city-level reanalysis-style weather to every ward, a different source/aggregation from the training data (the CSV does not document how weekly values were aggregated; we assume weekly sum for rain and weekly mean for temperature/humidity, Monday-start). Forecast days are never used as lag inputs.
- The dashboard is otherwise a historical backtest: no live case feed.

## AI assistants & external services disclosure
- **AI assistants used:** Claude (Anthropic) - reviewed the original notebook; wrote/refactored the code in `src/`, the Streamlit app, tests, notebook cells and documentation in two working sessions (initial build; final upgrade: portable model bundle, alert threshold + PR-AUC + calibration, strategy audit, input-quality layer, live-weather module, dashboard redesign). All reported numbers were produced by running the code. *(Team: add any other assistant you actually used - ChatGPT, Gemini, Copilot, etc.)*
- **External APIs at runtime:** none by default. Optional: **Open-Meteo** (free, no key, CC BY 4.0, attribution required) is called only when a user presses *Fetch latest weather* in the Live tab; one request per city, cached 6 h. Disable with `LIVE_WEATHER_OFFLINE=1`.
- **LLM calls at runtime:** none. **Data:** `data/bangladesh_dengue_weekly.csv` (hackathon-provided; identical to the supplied spreadsheet export).
- **Code/ideas from elsewhere:** no code copied. Ideas consulted: DengAI competition solutions (rolling weather windows), Mosqlimate / Johansson et al. PNAS 2019 (probabilistic arbovirus forecasting, coverage and baselines), conformalized quantile regression (Romano et al. 2019). *(Team: list any repo/tutorial code you copy, with URL and licence.)*
- **Libraries:** numpy, pandas, scikit-learn, joblib, pyarrow, streamlit, plotly, pytest (+ jupyter, matplotlib, seaborn for the notebook only).

## Team
*(Add team member names, roles and GitHub handles here.)*
