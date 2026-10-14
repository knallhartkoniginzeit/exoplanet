"""
Forecast module for the Exoplanet Explorer dashboard.

Builds yearly discovery counts from ``disc_year``, trains on completed years
only (the current partial year is excluded from training and flagged on the
chart), and produces point forecasts plus 80% / 95% confidence bands using:

- Holt's linear trend (ExponentialSmoothing) via ``statsmodels``
- ARIMA via ``statsmodels``
- Prophet (optional — disabled if ``prophet`` is not installed)

The module also computes a simple backtest (train up to N-3 years, test on
the last 3 years) reporting MAE and RMSE, and projects the cumulative
confirmed-planet total forward.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.charts import apply_dark_theme

# ---------------------------------------------------------------------------
# Public forecast entry point
# ---------------------------------------------------------------------------

# Simulated paths used to derive HoltWinters confidence bands, plus the RNG
# seed so forecasts are reproducible between refreshes.
_SIMULATION_PATHS = 500
_SIMULATION_SEED = 42

def forecast(
    df: pd.DataFrame,
    method: str = "holt",
    horizon: int = 5,
    current_year: Optional[int] = None,
) -> dict[str, Any]:
    """
    Run the selected forecast model and return a result dict.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed exoplanet table (with a numeric ``disc_year`` column).
    method : str
        One of ``"holt"``, ``"arima"``, ``"prophet"``.
    horizon : int
        Number of future years to forecast (1–10).
    current_year : int, optional
        Calendar year to treat as partial. Defaults to current year.

    Returns
    -------
    dict
        Keys:
        - ``figure``: Plotly figure (history + forecast + bands + partial-year flag)
        - ``insight``: caveat / takeaway string
        - ``cumulative``: dict with projected cumulative milestones (list of dicts)
        - ``backtest``: dict with MAE / RMSE for train vs test, per model attempt
        - ``history_counts``: yearly counts used for fitting (pandas Series)
        - ``last_year``: last training year
        - ``note``: status note (e.g. model unavailable)
    """
    current_year = current_year or pd.Timestamp.now().year

    yearly = _build_yearly_counts(df, current_year=current_year)
    result: dict[str, Any] = {
        "figure": None,
        "insight": "",
        "cumulative": [],
        "backtest": {},
        "history_counts": yearly,
        "last_year": int(yearly.index.max()) if not yearly.empty else None,
        "note": "",
    }

    if yearly.empty:
        result["note"] = "Not enough yearly discovery data to forecast."
        return result

    # Backtest before fitting the full model, so we can report metrics.
    bt = _backtest(yearly)
    result["backtest"] = bt

    # Fit the chosen model on completed years only.
    series = yearly["count"]

    fitted = None
    forecast_values: Optional[np.ndarray] = None
    conf_80_low: Optional[np.ndarray] = None
    conf_80_high: Optional[np.ndarray] = None
    conf_95_low: Optional[np.ndarray] = None
    conf_95_high: Optional[np.ndarray] = None

    try:
        if method == "holt":
            fitted, forecast_values, conf_80_low, conf_80_high, conf_95_low, conf_95_high = _fit_holt(series, horizon)
        elif method == "arima":
            fitted, forecast_values, conf_80_low, conf_80_high, conf_95_low, conf_95_high = _fit_arima(series, horizon)
        elif method == "prophet":
            fitted, forecast_values, conf_80_low, conf_80_high, conf_95_low, conf_95_high = _fit_prophet(series, horizon)
        else:
            result["note"] = f"Unknown forecast method: {method}"
            return result
    except Exception as exc:  # noqa: BLE001
        result["note"] = f"Model fit failed ({method}): {exc}"
        return result

    if forecast_values is None:
        result["note"] = f"No forecast produced by {method}."
        return result

    last_year = int(series.index.max())
    future_years = np.arange(last_year + 1, last_year + 1 + horizon)

    # Build the Plotly figure.
    fig = _build_forecast_figure(
        history_years=series.index.astype(int),
        history_counts=series.values,
        future_years=future_years,
        forecast_values=forecast_values,
        conf_80_low=conf_80_low,
        conf_80_high=conf_80_high,
        conf_95_low=conf_95_low,
        conf_95_high=conf_95_high,
        partial_year=current_year,
        method=method,
    )
    result["figure"] = fig

    # Cumulative projection.
    cum_now = int(df.dropna(subset=["disc_year"]).shape[0])
    result["cumulative"] = _build_cumulative_projection(
        series=series,
        forecast_values=forecast_values,
        future_years=future_years,
        cum_now=cum_now,
        horizon=horizon,
        last_year=last_year,
    )

    result["insight"] = (
        "Discoveries are lumpy and driven by mission data releases "
        "(Kepler 2014/2016, TESS), so forecasts carry wide uncertainty. "
        "Treat point forecasts as scenario baselines, not certain predictions."
    )
    return result


# ---------------------------------------------------------------------------
# Yearly counts
# ---------------------------------------------------------------------------

def _build_yearly_counts(df: pd.DataFrame, current_year: int) -> pd.DataFrame:
    """
    Build a DataFrame of discovery counts indexed by discovery year.

    Returns a DataFrame with a ``count`` column (callers index it as
    ``yearly[\"count\"]``). The current (partial) calendar year is kept in
    the returned frame but excluded from model training by forecast callers.
    """
    sub = df.dropna(subset=["disc_year"]).copy()
    sub["disc_year"] = sub["disc_year"].astype(int)
    counts = (
        sub.groupby("disc_year").size().reset_index(name="count")
    ).set_index("disc_year")
    # Ensure the current year is present (as 0 if nothing discovered yet this year).
    if current_year not in counts.index:
        counts.loc[current_year] = {"count": 0}
    return counts.sort_index()


# ---------------------------------------------------------------------------
# Model fits
# ---------------------------------------------------------------------------

def _to_fit_series(series: pd.Series) -> pd.Series:
    """Return a copy of ``series`` on a RangeIndex.

    statsmodels' ARIMA rejects non-standard integer indexes (e.g. calendar
    years), and HoltWinters also behaves best on a positional index, so both
    fitters re-index to ``RangeIndex`` and map results back to year labels.
    """
    return pd.Series(series.values, index=pd.RangeIndex(len(series)))


def _fit_holt(
    series: pd.Series,
    horizon: int,
) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit Holt's linear trend (ExponentialSmoothing) and forecast with bands.

    statsmodels 0.15's HoltWinters results expose ``forecast()`` but no
    analytic prediction intervals, so 80% / 95% bands are derived from
    simulated future paths (``HoltWintersResults.simulate``).

    Returns (fitted_model, forecast, conf_80_low, conf_80_high,
             conf_95_low, conf_95_high).
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    fit_series = _to_fit_series(series)
    model = ExponentialSmoothing(
        fit_series,
        trend="add",
        seasonal=None,
        initialization_method="estimated",
    )
    fitted = model.fit()
    forecast_values = fitted.forecast(horizon).values

    # Empirical bands from simulated paths (fixed seed for reproducibility).
    sims = np.asarray(
        fitted.simulate(
            horizon, anchor="end", repetitions=_SIMULATION_PATHS,
            rng=np.random.default_rng(_SIMULATION_SEED),
        )
    )
    # sims shape: (horizon, repetitions)
    # Lower bounds are clipped at 0: yearly discovery counts cannot be
    # negative, and simulated trend paths occasionally undershoot zero.
    conf_80_low = np.maximum(np.percentile(sims, 10, axis=1), 0.0)
    conf_80_high = np.maximum(np.percentile(sims, 90, axis=1), 0.0)
    conf_95_low = np.maximum(np.percentile(sims, 2.5, axis=1), 0.0)
    conf_95_high = np.maximum(np.percentile(sims, 97.5, axis=1), 0.0)
    return fitted, forecast_values, conf_80_low, conf_80_high, conf_95_low, conf_95_high


def _fit_arima(
    series: pd.Series,
    horizon: int,
) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit an ARIMA(p,d,q) model and forecast with bands.

    Uses a modest, data-driven order selection: AIC-guided search over a small
    grid, falling back to a simple (1,1,1) if search fails.
    """
    from statsmodels.tsa.arima.model import ARIMA

    fit_series = _to_fit_series(series)

    # Simple order selection via AIC over a small grid.
    best_aic = np.inf
    best_order: Tuple[int, int, int] = (1, 1, 1)
    for p in (0, 1, 2):
        for d in (0, 1):
            for q in (0, 1, 2):
                try:
                    m = ARIMA(fit_series, order=(p, d, q))
                    res = m.fit()
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p, d, q)
                except Exception:  # noqa: BLE001
                    continue

    model = ARIMA(fit_series, order=best_order)
    fitted = model.fit()
    forecast_obj = fitted.get_forecast(steps=horizon)
    forecast_values = forecast_obj.predicted_mean.values
    conf = forecast_obj.conf_int(alpha=0.20)
    # Clip lower bounds at 0: discovery counts cannot be negative.
    conf_80_low = np.maximum(conf.iloc[:, 0].values, 0.0)
    conf_80_high = np.maximum(conf.iloc[:, 1].values, 0.0)
    conf95 = forecast_obj.conf_int(alpha=0.05)
    conf_95_low = np.maximum(conf95.iloc[:, 0].values, 0.0)
    conf_95_high = np.maximum(conf95.iloc[:, 1].values, 0.0)
    return fitted, forecast_values, conf_80_low, conf_80_high, conf_95_low, conf_95_high


def _fit_prophet(
    series: pd.Series,
    horizon: int,
) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit Prophet and return forecasts with 80% / 95% bands.

    Prophet is optional; if unavailable, the caller should disable the option.
    """
    try:
        from prophet import Prophet
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Prophet is not installed") from exc

    hist = series.reset_index()
    hist.columns = ["ds", "y"]
    hist["ds"] = pd.to_datetime(hist["ds"], format="%Y")

    model = Prophet(interval_width=0.80, uncertainty_samples=1000)
    model.fit(hist)

    future = model.make_future_dataframe(periods=horizon, freq="YE")
    forecast_df = model.predict(future)

    hist_forecast = forecast_df.iloc[:len(series)]
    future_forecast = forecast_df.iloc[len(series):].reset_index(drop=True)

    forecast_values = future_forecast["yhat"].values
    conf_80_low = future_forecast["yhat_lower"].values
    conf_80_high = future_forecast["yhat_upper"].values

    # Prophet's default interval_width gives 80%; request 95% separately.
    model95 = Prophet(interval_width=0.95, uncertainty_samples=1000)
    model95.fit(hist)
    future95 = model95.make_future_dataframe(periods=horizon, freq="YE")
    f95 = model95.predict(future95).iloc[len(series):].reset_index(drop=True)
    conf_95_low = f95["yhat_lower"].values
    conf_95_high = f95["yhat_upper"].values

    return model, forecast_values, conf_80_low, conf_80_high, conf_95_low, conf_95_high


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def _backtest(yearly: pd.DataFrame) -> dict[str, Any]:
    """
    Backtest: train on all but the last 3 years, test on the last 3 years.

    Reports MAE and RMSE for Holt and ARIMA (if they can fit). Returns a dict
    with per-model metrics and the train/test split summary.
    """
    counts = yearly["count"]
    if len(counts) < 4:
        return {
            "note": "Not enough history for a 3-year backtest.",
            "train_years": None,
            "test_years": None,
            "holt": None,
            "arima": None,
        }

    train = counts.iloc[:-3]
    test = counts.iloc[-3:]
    train_years = counts.index[:-3]
    test_years = counts.index[-3:]

    metrics: dict[str, Any] = {
        "note": "",
        "train_years": [int(y) for y in train_years],
        "test_years": [int(y) for y in test_years],
        "train_counts": [int(v) for v in train.values],
        "actual": [int(v) for v in test.values],
        "holt": None,
        "arima": None,
    }

    # Holt backtest on the train set, forecast 3 steps.
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        m = ExponentialSmoothing(_to_fit_series(train), trend="add",
                                 seasonal=None,
                                 initialization_method="estimated")
        res = m.fit()
        # statsmodels 0.15 HoltWinters exposes forecast(), not get_forecast().
        f = res.forecast(3).values
        mae = float(np.mean(np.abs(f - test.values)))
        rmse = float(np.sqrt(np.mean((f - test.values) ** 2)))
        metrics["holt"] = {"mae": round(mae, 2), "rmse": round(rmse, 2)}
    except Exception as exc:  # noqa: BLE001
        metrics["holt"] = {"error": str(exc)}

    # ARIMA backtest.
    try:
        from statsmodels.tsa.arima.model import ARIMA
        train_fit = _to_fit_series(train)
        best_aic = np.inf
        best_order: Tuple[int, int, int] = (1, 1, 1)
        for p in (0, 1, 2):
            for d in (0, 1):
                for q in (0, 1, 2):
                    try:
                        mm = ARIMA(train_fit, order=(p, d, q))
                        rr = mm.fit()
                        if rr.aic < best_aic:
                            best_aic = rr.aic
                            best_order = (p, d, q)
                    except Exception:  # noqa: BLE001
                        continue
        mm = ARIMA(train_fit, order=best_order)
        rr = mm.fit()
        f = rr.get_forecast(steps=3).predicted_mean.values
        mae = float(np.mean(np.abs(f - test.values)))
        rmse = float(np.sqrt(np.mean((f - test.values) ** 2)))
        metrics["arima"] = {"mae": round(mae, 2), "rmse": round(rmse, 2),
                            "order": list(best_order)}
    except Exception as exc:  # noqa: BLE001
        metrics["arima"] = {"error": str(exc)}

    return metrics


# ---------------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------------

def _build_forecast_figure(
    history_years: np.ndarray,
    history_counts: np.ndarray,
    future_years: np.ndarray,
    forecast_values: np.ndarray,
    conf_80_low: np.ndarray,
    conf_80_high: np.ndarray,
    conf_95_low: np.ndarray,
    conf_95_high: np.ndarray,
    partial_year: int,
    method: str,
) -> go.Figure:
    """Plot history + forecast + 80%/95% confidence bands, with partial-year flag."""
    fig = go.Figure()

    # Historical line.
    fig.add_trace(
        go.Scatter(
            x=history_years,
            y=history_counts,
            mode="lines+markers",
            name="History",
            line=dict(color="#7fc1ff", width=2.5),
            marker=dict(size=6),
        )
    )

    # Partial-year marker (current year, not used for training).
    fig.add_vline(
        x=partial_year,
        line_dash="dot",
        line_color="rgba(255, 179, 71, 0.7)",
        annotation_text="Partial year (excluded from training)",
        annotation_position="top right",
        annotation_font=dict(color="#ffb347", size=10),
        opacity=0.8,
    )

    # Future point forecast.
    fig.add_trace(
        go.Scatter(
            x=future_years,
            y=forecast_values,
            mode="lines+markers",
            name=f"{method} forecast",
            line=dict(color="#ffd34d", width=2.5, dash="solid"),
            marker=dict(size=7, symbol="diamond"),
        )
    )

    # 95% band.
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([future_years, future_years[::-1]]),
            y=np.concatenate([conf_95_high, conf_95_low[::-1]]),
            fill="toself",
            fillcolor="rgba(255, 179, 71, 0.12)",
            line=dict(color="rgba(255, 179, 71, 0)"),
            name="95% confidence band",
            showlegend=True,
        )
    )

    # 80% band (drawn after so it sits above the 95% band visually).
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([future_years, future_years[::-1]]),
            y=np.concatenate([conf_80_high, conf_80_low[::-1]]),
            fill="toself",
            fillcolor="rgba(127, 193, 255, 0.18)",
            line=dict(color="rgba(127, 193, 255, 0)"),
            name="80% confidence band",
            showlegend=True,
        )
    )

    fig.update_layout(
        title="Yearly discoveries: history + forecast",
        xaxis_title="Discovery year",
        yaxis_title="Confirmed planets per year",
        xaxis=dict(dtick=1, type="linear"),
    )
    fig = apply_dark_theme(fig)
    fig.update_yaxes(tickformat=",d")
    fig.update_xaxes(tickformat="d")
    return fig


# ---------------------------------------------------------------------------
# Cumulative projection
# ---------------------------------------------------------------------------

def _build_cumulative_projection(
    series: pd.Series,
    forecast_values: np.ndarray,
    future_years: np.ndarray,
    cum_now: int,
    horizon: int,
    last_year: int,
) -> list[dict[str, Any]]:
    """
    Project cumulative confirmed-planet count forward.

    Returns a list of milestones: years where the cumulative total is expected
    to cross integer thresholds (e.g. 6000, 6500, ...), plus the projected
    cumulative at each forecast year.
    """
    # Cumulative count at the end of the last training year.
    cum_at_last = int(series.values.sum())
    # Projected additions per future year (clipped to >= 0 for sanity).
    additions = np.maximum(forecast_values, 0.0)
    cum = cum_at_last + np.cumsum(additions)
    cum_final = int(round(cum[-1])) if len(cum) else cum_at_last

    milestones: list[dict[str, Any]] = []
    # Integer thresholds from the next round number above current total.
    start = int(np.ceil(cum_now / 100.0) * 100)
    for t in range(start, cum_final + 200, 100):
        # Find the first forecast year whose projected cumulative >= t.
        idx = np.argmax(cum >= t) if len(cum) else -1
        if idx >= 0:
            year_reach = int(future_years[idx])
            val = int(round(cum[idx]))
            milestones.append({
                "threshold": t,
                "projected_year": year_reach,
                "projected_total": val,
            })
        else:
            milestones.append({
                "threshold": t,
                "projected_year": None,
                "projected_total": None,
            })

    # Per-year projected cumulative.
    for y, v in zip(future_years.tolist(), cum.tolist()):
        milestones.append({
            "threshold": None,
            "projected_year": int(y),
            "projected_total": int(round(v)),
            "_is_year_row": True,
        })

    return milestones
