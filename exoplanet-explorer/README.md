# Exoplanet Explorer

A Streamlit dashboard for the DVST (Data Visualization and Storytelling)
mini project that explores **confirmed exoplanets** using **live data from the
NASA Exoplanet Archive** and includes a **forecasting module** for future
discovery counts.

## Project structure

```
exoplanet-explorer/
  app.py                 # Streamlit entry point + layout
  .streamlit/
    config.toml          # Theme: bone-white background, black text
  src/
    data_loader.py       # NASA TAP API fetch, caching, fallback snapshot
    preprocess.py        # Cleaning, derived columns, log helpers
    charts.py            # Plotly visualizations, dark space theme
    forecast.py          # Holt / ARIMA / Prophet forecasts + backtest
  data/
    snapshot.csv         # Offline fallback CSV (updated from live data)
  requirements.txt
  README.md
```

## UI theme

The frontend uses a **bone-white background (`#f8f6f0`) with black text
(`#101418`)** — no emojis, no decorative chrome. The theme is set two ways:

1. `.streamlit/config.toml` — Streamlit's native light theme (applies before
   any page code runs, so there is no dark-mode flash).
2. A `<style>` block injected from `app.py` — final overrides for Streamlit
   internals (sidebar, alerts, buttons, dataframes) so everything stays in the
   black-on-white palette.

## Run instructions

```bash
cd exoplanet-explorer
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501` by default. If the NASA Exoplanet
Archive is reachable, data is fetched live and the banner reads **"Data status:
Live"**. If the API is unreachable, the app falls back to
`data/snapshot.csv` and the banner reads **"Data status: Offline snapshot"**.

### Troubleshooting

- **Port already in use** — another Streamlit process holds port 8501. On
  Windows: `powershell -Command "Get-NetTCPConnection -LocalPort 8501 -State Listen | ForEach { Stop-Process -Id $\_.OwningProcess -Force }"`, or run
  `kill_streamlit.ps1` in this folder, then restart.
- **`streamlit-autorefresh not installed`** — run
  `pip install -r requirements.txt` and restart the app.
- **Theme looks dark** — make sure `.streamlit/config.toml` exists (it ships
  with this project) and restart Streamlit from inside the project folder so
  it is picked up.

To force a fresh fetch, use the **"Refresh data now"** button in the sidebar.
The dashboard reuses the same NASA Exoplanet Archive table on every fetch, so repeated refreshes return only newly confirmed or updated rows.
The dashboard **auto-refreshes every 10 minutes** via `streamlit-autorefresh`
(installed with `requirements.txt`; the API cache TTL is also 60 minutes).

## Data source

- **NASA Exoplanet Archive** — [https://exoplanetarchive.ipac.caltech.edu/](https://exoplanetarchive.ipac.caltech.edu/)
- Table: `pscomppars` (composite best-parameters table, one row per confirmed planet).
- TAP sync endpoint: `https://exoplanetarchive.ipac.caltech.edu/TAP/sync`
- Download / citation page: [https://exoplanetarchive.ipac.caltech.edu/docs/DownloadDocumentation.html](https://exoplanetarchive.ipac.caltech.edu/docs/DownloadDocumentation.html)

### Dataset citation

If you use this dashboard or the underlying data, please cite:

> NASA Exoplanet Archive, operated by Caltech, under contract with the National
> Aeronautics and Space Administration. Accessed via the TAP API
> (`pscomppars` table).

The NASA Exoplanet Archive is funded by NASA's Exoplanet Exploration Program.

## Insights found

Below are illustrative insights surfaced by the dashboard. Update these after
running against live data.

- **Transit dominates modern discoveries.** After Kepler's 2014 data release,
  the transit method accounts for the large majority of confirmed planets; the
  stacked-area chart makes the Kepler (2014/2016) and TESS eras visually
  obvious.
- **Most confirmed planets are "Hot".** The temperature histogram is right-
  skewed toward hot Jupiters and short-period transiting planets; temperate
  worlds are the minority, which is expected given detection bias toward
  short-period, large planets.
- **Radius is the best-classified attribute.** Because `pl_rade` is widely
  reported, the `planet_type` breakdown (Earth-like / Super-Earth / Neptune-like
  / Jupiter-like) is informative, but mass is missing for many planets, so
  mass-based conclusions need care.
- **Distance is skewed.** The distance histogram (log scale) shows most
  confirmed planets within a few thousand light-years, with a long tail from
  ground-based RV and transit surveys.
- **Forecasts are highly uncertain.** Yearly discovery counts are **lumpy**:
  they spike around mission data releases (Kepler 2014/2016, TESS) rather than
  growing smoothly. Point forecasts should be read as scenario baselines, not
  predictions. The Forecast tab includes confidence bands and an explicit caveat.

## Screenshots

(Instructions preserved for the project README — capture your own screenshots.)

1. **Overview tab** — KPI row + discovery-per-year line chart with cumulative
   toggle.
2. **Discovery Trends tab** — method distribution (bar + donut) and top-10
   mission/telescope bar chart.
3. **Planet Properties tab** — mass-vs-radius scatter (log-log, colored by
   planet type), temperature histogram + box plot, distance histogram
   (log x-axis, light-years), orbital-period histogram (log x-axis).
4. **Forecast tab** — model selection, horizon slider, history + forecast with
   80%/95% bands, cumulative forecast, backtest MAE/RMSE, caveat box.
5. **Planet Explorer tab** — planet search with autocomplete, Planet Information
   Card with Earth comparisons, and compare mode for 2–3 planets.
6. **Data Quality tab** — missing-value percentages per column, row counts
   before/after each cleaning step, in an expander.

To add screenshots: run the app, resize the browser to a typical dashboard
width, and capture each tab with a tool such as your OS screenshot utility or
Selenium/Playwright. Drop the images into an `assets/` folder and insert them
with Markdown image syntax.

## Forecast module notes

- The forecast trains on **completed years only** — the current partial year is
  excluded from training and flagged on the chart.
- Available models:
  - **Holt's linear trend** (Exponential Smoothing) via `statsmodels`
  - **ARIMA** via `statsmodels`
  - **Prophet** (optional — install `prophet`; if unavailable, the option is
    disabled)
- Backtest: train on data up to **N−3 years**, test on the last 3 years, report
  **MAE** and **RMSE**.
- Cumulative forecast: projects when the total confirmed-planet count is
  expected to cross thresholds (e.g. "expected to pass X confirmed planets by YEAR").
- Caveat (shown in the app): discovery counts are **lumpy** and driven by mission
  data releases, so forecasts carry wide uncertainty and should not be read as
  precise predictions.

## Error handling

- API fetch uses a timeout and retry logic; on failure the app falls back to
  `data/snapshot.csv` and shows a "Data status: Offline snapshot" banner with
  the failure reason.
- Missing values are preserved by default (only rows missing `pl_name` are
  dropped). Charts drop missing values **per chart** rather than globally, so
  most planets remain visible.
- If a forecast model fails to fit, the Forecast tab shows a friendly error
  instead of crashing the app.

## License

This dashboard is a student/mini-project deliverable. The underlying data is
provided by the NASA Exoplanet Archive under NASA's data-use policies.
