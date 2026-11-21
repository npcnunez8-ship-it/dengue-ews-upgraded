# Judging-rubric self-assessment (strict; verified 2026-11-22 unless marked TEAM)

| # | Criterion (pts) | Status | Evidence / how verified | Honest risk |
|---|---|---|---|---|
| 1 | Reproducible setup (10) | Ready, **TEAM to re-test from the public repo** | Fresh virtualenv from `requirements.txt` (incl. `pyarrow`): install 25 s; app launched (HTTP 200, `/_stcore/health` ok); 32 tests pass in 3 s. A bundle "built with another scikit-learn" is detected and rebuilt automatically in ~25 s (simulated and verified). `.streamlit/config.toml` disables the usage-stats prompt. | Tested on Python 3.12 / Linux only. |
| 2 | Multi-week ward lag pipeline (15) | Done | `src/features.py`: per-`ward_id` lags 1-8w + rolling 2-8w for temp/rain/humidity; tests assert lag correctness, no cross-ward leakage, no future leakage; the dashboard re-verifies every lag against raw source weeks live. | - |
| 3 | Risk prediction engine (15) | Done | Serialized HistGB-Poisson, 4 tiers, week 2/3/4 + 2-4 total, 80% intervals; test MAE 2.62 vs 4.03 seasonal-naive; High/Critical recall 69%, precision 55%, PR-AUC 0.64; chronological validation, placebo audit, direct-vs-pooled strategy check. | Data look synthetic; metrics are an upper bound. |
| 4 | Interactive UI (10) | Done | 7-tab Streamlit app; `AppTest` smoke-run with 0 exceptions across all windows, cities, in/out-of-sample dates, ward-map branch (with temporary test coordinates) and the offline-failure path of the live tab. Real server launch checked via HTTP. | **Not visually inspected in a real browser in the build sandbox** (no browser available). Map basemap outline needs internet. |
| 5 | ARCHITECTURE.md (10) | Done | Root file: mermaid data-flow diagram, split table, timing convention, model/strategy rationale, cost controls. | **TEAM:** confirm the mermaid diagram renders on GitHub; file is ~2 pages. |
| 6 | Compute & API efficiency (10) | Done | joblib bundle (2.4 MB), `st.cache_*`, vectorised inference (31k rows in ~0.5 s, one ward ≈ 8 ms), zero runtime API/LLM calls by default; live layer: 1 request/city, 6 h disk cache, offline switch, env-var config (all in ARCHITECTURE §5; unit-tested with mocks). | - |
| 7 | 3-min uncut demo (20) | **TEAM - not done** | Script in `SUBMISSION_GUIDE.md` §E. | No video exists yet; 0 points until recorded. |
| 8 | Git history + AI disclosure (10) | **TEAM - partly done** | AI/API disclosure complete in README (add other assistants actually used + names). | No Git history is shipped; commits must be real, spread across members over the 24 h. Cannot be earned by this ZIP alone. |
| Bonus (+5) | Domain extensions | Partly | Implemented: ward×week risk heatmap, multi-week 80% intervals + calibration-by-risk-level, chronological validation error dashboard (MAE/RMSE by month, rolling-origin CV), PR curve/PR-AUC. **Not** implemented: ward-level geographic heatmap (no verified coordinates exist in the data; activates automatically if `data/ward_coordinates.csv` is added). | A judge may not count the city-level map as a ward heatmap. |

## Before → after
| Area | Before | After |
|---|---|---|
| Install | `pyarrow` missing; bundle pickled pandas internals → load errors on other machines; sklearn mismatch fatal | portable bundle (no pandas objects), `pyarrow` + bounded pins + lock file, automatic rebuild on mismatch, tested in a clean venv |
| Detection quality | tiers + R²/MAE/recall | + PR-AUC, PR curve, sensitive watch-flag operating point (validation-chosen), per-risk-level interval calibration, direct-vs-pooled-vs-derived strategy audit |
| Robustness | strict rejection of any gap | gap repair (≤2 weeks), unusual-value warnings, staleness checks, data-quality report |
| Operations | none | optional cached Open-Meteo layer with provenance labels and offline kill-switch |
| Dashboard | 6 tabs, treemap called a "heatmap" in docs | executive overview, separate week-2/3/4 cards, ward ranking, real ward×week heatmap, city map, live tab, richer validation, in-app lag verification; docs corrected |
| Tests | 19 | 32 |

## Verified this session
- `python -m pytest -q tests`: **32 passed** (also in a fresh venv). Notebook re-executed end-to-end: 0 error cells; headline metrics unchanged (test MAE 2.621, RMSE 4.905, R² 0.814, coverage 0.94).
- **Not verified:** Open-Meteo requests against the real API (network blocked in the sandbox; mocked only), browser rendering, Windows/macOS, Python ≠ 3.12.
- **Finding to be aware of:** the watch flag's 10% false-alarm cap (validation) became 19% on test.
