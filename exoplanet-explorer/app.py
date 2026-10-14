"""
Exoplanet Explorer — Streamlit dashboard for the DVST mini project.

Live data from the NASA Exoplanet Archive (TAP API, table `pscomppars`),
with a fallback to `data/snapshot.csv`. Includes forecast, search, compare,
and a Data Quality report.
"""

from __future__ import annotations

import io
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import fetch_exoplanet_data, refresh_snapshot
from src.preprocess import (
    CURRENT_YEAR,
    build_data_quality_report,
    preprocess,
)
from src.charts import (
    chart_correlation_heatmap,
    chart_distance_hist,
    chart_discoveries_per_year,
    chart_habitable_zone,
    chart_mass_radius,
    chart_method_donut,
    chart_method_distribution,
    chart_method_stacked_area,
    chart_mission_top10,
    chart_period_hist,
    chart_temperature_box,
    chart_temperature_hist,
    earth_comparison,
    planet_info_card,
)
import os
import base64
from datetime import datetime

from src.forecast import forecast

# ---------------------------------------------------------------------------
# Caching layer (st.cache_data with TTL) — wraps the data loader
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Fetching exoplanet data from NASA…")
def cached_fetch() -> Tuple[pd.DataFrame, dict[str, Any]]:
    """Cached fetch of live exoplanet data (or snapshot fallback)."""
    return fetch_exoplanet_data()


# ---------------------------------------------------------------------------
# App config / layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Exoplanet Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Force the page-level CSS accent color / font to override Streamlit defaults.
st.markdown(
    '<style>'
    ':root {'
    '--primary-color: #101418 !important;'
    '--background-color: #f8f6f0 !important;'
    '--secondary-background-color: #f8f6f0 !important;'
    '--text-color: #101418 !important;'
    '} '
    '</style>',
    unsafe_allow_html=True,
)

# ---- Bone-white / black UI: no emojis, no decorative chrome ----------
_BONE_WHITE = "#f8f6f0"
_BLACK = "#101418"

_CSS = """

/* ---------- Reset / base ---------- */
html, body, [class*=css-18ewl0o], .stApp, .stAppViewContainer, .stMain, .block-container, .stVerticalBlock, .stAppHeader, .stMainBlockContainer, .st-emotion-cache-13k62yr {
  background: #f8f6f0 !important;
  color: #101418 !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               "Helvetica Neue", Arial, sans-serif !important;
  font-size: 15px !important;
  line-height: 1.5 !important;
}

/* Hide favicon link in tab. */
link[rel="icon"], link[rel="shortcut icon"] { display: none !important; }

/* ---------- Title / headings ---------- */
.css-18ewl0o > div:first-child h1,
.css-18ewl0o > div:first-child h2,
.css-18ewl0o > div:first-child h3,
.css-18ewl0o > div:first-child h4 {
  color: #101418 !important;
  font-weight: 700 !important;
}

/* ---------- Buttons ---------- */
button {
  background: #101418 !important;
  color: #f8f6f0 !important;
  border: 1px solid #101418 !important;
  border-radius: 2px !important;
  padding: 6px 12px !important;
  font-weight: 600 !important;
}
button:hover {
  background: #3a424e !important;
  border-color: #3a424e !important;
}
button[data-baseweb="primary"],
button[data-baseweb="secondary"],
button[data-baseweb] {
  background: #101418 !important;
  color: #f8f6f0 !important;
  border-color: #101418 !important;
}

/* ---------- Metric cards (KPI row) ---------- */
.css-18f1gpy, .css-189tp6g {
  background: #efece4 !important;
  border: 1px solid #d9d3c6 !important;
  border-radius: 4px !important;
}
.css-18f1gpy div, .css-189tp6g div {
  color: #101418 !important;
}

/* ---------- Alerts / banners (info/success/warning/error) ---------- */
.alert, .stAlert {
  background: #efece4 !important;
  border: 1px solid #d9d3c6 !important;
  border-left: 4px solid #101418 !important;
  border-radius: 4px !important;
  color: #101418 !important;
  padding: 10px 14px !important;
  font-size: 14px !important;
}
.alert strong {
  color: #101418 !important;
}

/* ---------- Expanders ---------- */
.expander, .stExpander {
  background: #efece4 !important;
  border: 1px solid #d9d3c6 !important;
  border-radius: 4px !important;
}

/* ---------- Data frame / dataframe cells ---------- */
table[data-testid="stDataFrame"],
.dataframeContainer, .stDataFrame, .stTable {
  background: #f8f6f0 !important;
  color: #101418 !important;
}
table td, table th {
  color: #101418 !important;
}
table tr:nth-child(even) {
  background: #f5f2ea !important;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"],
.stSidebar, .stSidebarContent {
  background: #efece4 !important;
  color: #101418 !important;
}


/* ---------- Scrollbar (thin, neutral) ---------- */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #efece4; }
::-webkit-scrollbar-thumb { background: #b9b1a2; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #9a8f7e; }

/* ---------- Remove placeholder/steps sparklines ---------- */
.css-25ouw4 {
  background: transparent !important;
}
"""

def _inject_css(css: str) -> None:
    """Inject raw CSS into the page as a real <style> element.

    Streamlit's ``st.markdown(unsafe_allow_html=True)`` renders raw HTML as
    actual DOM nodes, so a ``<style>...</style>`` tag IS created as a real
    stylesheet that applies to the page.  The CSS overrides Streamlit's own
    dark-theme emotion-cache rules with ``!important``.
    """
    st.markdown(
        "<style>" + css + "</style>",
        unsafe_allow_html=True,
    )


_inject_css(_CSS)

with st.sidebar:
    st.header("Controls")

    refresh_clicked = st.button("Refresh data now")
    if refresh_clicked:
        # Clear the cached fetch so the next run re-fetches.
        cached_fetch.clear()
        st.rerun()

    st.divider()
    st.caption("Auto-refresh updates the dashboard every 10 minutes.")

    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(
            interval=10 * 60 * 1000,
            key="exoplanet_autorefresh",
        )
    except Exception:  # noqa: BLE001 — package not installed
        st.caption("streamlit-autorefresh not installed — auto-refresh disabled.")

    st.divider()
    st.caption("Filters apply across all charts and KPIs.")


# ---------------------------------------------------------------------------
# Data fetch + preprocessing
# ---------------------------------------------------------------------------

df_raw, meta = cached_fetch()

# Snapshot refresh: after a successful live fetch, keep the fallback current.
if meta.get("source") == "live" and not df_raw.empty:
    try:
        refresh_snapshot(df_raw)
    except Exception:  # noqa: BLE001 — don't crash if the snapshot write fails
        pass

data_source = meta.get("source", "unknown")
fetch_error = meta.get("fetch_error")
last_fetched = meta.get("last_fetched")
count_total_archive = meta.get("count_total", 0)
row_count_loaded = meta.get("row_count", 0)

# Data-status banner.
if data_source == "live":
    st.success("Data status: **Live** — fetched from the NASA Exoplanet Archive.")
else:
    reason = fetch_error or "unknown"
    st.warning(
        f"Data status: **Offline snapshot** — using `data/snapshot.csv`. "
        f"(Live fetch failed: {reason}.)"
    )

# Show last fetched / totals.
total_placeholder = st.empty()
with total_placeholder.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Last fetched at", last_fetched or "—")
    with col2:
        st.metric("Rows loaded", f"{row_count_loaded:,}")
    with col3:
        st.metric(
            "Total confirmed planets (archive count)",
            f"{count_total_archive:,}" if count_total_archive else "—",
        )

# Preprocess.
current_year = pd.Timestamp.now().year
df = preprocess(df_raw, current_year=current_year)

if df.empty:
    st.warning("No planet data available. Check your connection or the snapshot file.")
    st.stop()


# ---------------------------------------------------------------------------
# What's new panel (session_state previous count)
# ---------------------------------------------------------------------------

def _update_whats_new(df: pd.DataFrame) -> None:
    """Show recently-updated planets and count added since previous fetch."""
    if "prev_row_count" not in st.session_state:
        st.session_state["prev_row_count"] = int(df.shape[0])

    prev = st.session_state.get("prev_row_count", 0)
    curr = int(df.shape[0])
    added = curr - prev
    st.session_state["prev_row_count"] = curr

    st.subheader("What's new")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Planets in current dataset", f"{curr:,}")
    with col_b:
        delta_label = "more than last fetch" if added >= 0 else "fewer than last fetch"
        st.metric("Change since last fetch", f"{added:+,}", delta_label)

    # Recently updated rows (last 30 days by discovery publication date).
    if "disc_pubdate" in df.columns:
        updates = df["disc_pubdate"].dropna().astype(str)
        if not updates.empty:
            parsed = pd.to_datetime(updates, errors="coerce")
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
            recent = parsed[parsed >= cutoff]
            if not recent.empty:
                recent_names = (
                    df.loc[recent.index, "pl_name"].tolist()
                    if "pl_name" in df.columns
                    else []
                )
                st.info(
                    f"**{len(recent)}** planets published in the last 30 days. "
                    f"Most recent: {', '.join(recent_names[:8])}"
                    + ("…" if len(recent_names) > 8 else "")
                )
            else:
                st.caption("No planets published in the last 30 days.")
        else:
            st.caption("No `disc_pubdate` data available.")


_update_whats_new(df)


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def _unique_sorted(series: pd.Series) -> list:
    return sorted([str(x) for x in series.dropna().unique() if str(x).strip()])


methods_all = _unique_sorted(df["discoverymethod"]) if "discoverymethod" in df.columns else []
planet_types_all = _unique_sorted(df["planet_type"]) if "planet_type" in df.columns else []
missions_all = _unique_sorted(df["mission"]) if "mission" in df.columns else []

with st.sidebar:
    st.divider()
    st.subheader("Filters")

    year_vals = sorted([int(y) for y in df["disc_year"].dropna().unique()])
    if year_vals:
        year_min, year_max = int(min(year_vals)), int(max(year_vals))
        year_range = st.slider(
            "Discovery year range",
            min_value=year_min,
            max_value=year_max,
            value=(year_min, year_max),
            step=1,
        )
    else:
        year_range = (None, None)

    methods_sel = st.multiselect(
        "Discovery method",
        options=methods_all,
        default=[],
        format_func=lambda x: x,
    )
    if not methods_sel:
        methods_sel = methods_all  # empty selection => include every method

    planet_types_sel = st.multiselect(
        "Planet type",
        options=planet_types_all,
        default=planet_types_all,
    )
    if not planet_types_sel and planet_types_all:
        planet_types_sel = planet_types_all

    temp_vals = sorted([float(t) for t in df["pl_eqt"].dropna().unique()])
    if temp_vals:
        temp_min, temp_max = float(min(temp_vals)), float(max(temp_vals))
        temp_range = st.slider(
            "Equilibrium temperature (K)",
            min_value=temp_min,
            max_value=temp_max,
            value=(temp_min, temp_max),
            step=1.0,
        )
    else:
        temp_range = (None, None)

    dist_vals = sorted([float(d) for d in df["sy_dist_ly"].dropna().unique()])
    if dist_vals:
        dist_min, dist_max = float(min(dist_vals)), float(max(dist_vals))
        distance_range = st.slider(
            "Distance (light-years)",
            min_value=dist_min,
            max_value=dist_max,
            value=(dist_min, dist_max),
            step=1.0,
        )
    else:
        distance_range = (None, None)

    missions_sel = st.multiselect(
        "Mission / facility",
        options=missions_all,
        default=[] if missions_all else [],
    )
    if not missions_sel and missions_all:
        missions_sel = missions_all

    st.divider()
    st.caption(
        "Tip: leaving a filter empty = include all. "
        "See the **Data Quality** tab for missing-value details."
    )


# Convenience: empty multi-select means "all".
def _none_if_empty(lst):
    return lst if lst else None


# ---------------------------------------------------------------------------
# KPI row (top of Overview)
# ---------------------------------------------------------------------------

filtered = df.copy()
if year_range[0] is not None or year_range[1] is not None:
    lo, hi = year_range
    filtered = filtered[filtered["disc_year"].fillna(-np.inf) >= (lo or -np.inf)]
    filtered = filtered[filtered["disc_year"].fillna(np.inf) <= (hi or np.inf)]
if methods_sel:
    filtered = filtered[filtered["discoverymethod"].isin(methods_sel)]
if planet_types_sel:
    filtered = filtered[filtered["planet_type"].isin(planet_types_sel)]
if temp_range[0] is not None or temp_range[1] is not None:
    lo, hi = temp_range
    filtered = filtered[filtered["pl_eqt"].fillna(-np.inf) >= (lo or -np.inf)]
    filtered = filtered[filtered["pl_eqt"].fillna(np.inf) <= (hi or np.inf)]
if distance_range[0] is not None or distance_range[1] is not None:
    lo, hi = distance_range
    filtered = filtered[filtered["sy_dist_ly"].fillna(-np.inf) >= (lo or -np.inf)]
    filtered = filtered[filtered["sy_dist_ly"].fillna(np.inf) <= (hi or np.inf)]
if missions_sel:
    filtered = filtered[filtered["mission"].isin(missions_sel)]

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Total planets (filtered)", f"{filtered.shape[0]:,}")
with kpi2:
    yv = filtered["disc_year"].dropna()
    if not yv.empty:
        st.metric("Earliest / latest year",
                  f"{int(yv.min())} / {int(yv.max())}")
    else:
        st.metric("Earliest / latest year", "—")
with kpi3:
    mc = filtered["discoverymethod"].dropna().value_counts()
    if not mc.empty:
        top_method = mc.index[0]
        st.metric("Most common method", top_method)
    else:
        st.metric("Most common method", "—")
with kpi4:
    dv = filtered["sy_dist_ly"].dropna()
    if not dv.empty:
        st.metric("Median distance (ly)", f"{dv.median():,.1f}")
    else:
        st.metric("Median distance (ly)", "—")


# ---------------------------------------------------------------------------
# Tabbed layout
# ---------------------------------------------------------------------------

tabs = st.tabs([
    "Overview",
    "Discovery Trends",
    "Planet Properties",
    "Forecast",
    "Planet Explorer",
    "Data Quality",
])

# ------------------------------------------------------------------
# Overview
# ------------------------------------------------------------------
with tabs[0]:
    st.header("Overview")

    cumulative_toggle = st.toggle("Cumulative view", value=False)
    fig, insight = chart_discoveries_per_year(
        df,
        cumulative=cumulative_toggle,
        year_range=_none_if_empty(year_range) if not year_range else year_range,
        methods=_none_if_empty(methods_sel),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(insight)

    # Quick stats strip.
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Planets (filtered)", f"{filtered.shape[0]:,}")
    with c2:
        yv = filtered["disc_year"].dropna()
        st.metric("Year span",
                  f"{int(yv.min())}–{int(yv.max())}" if not yv.empty else "—")
    with c3:
        mc = filtered["discoverymethod"].dropna().value_counts()
        st.metric("Top method", mc.index[0] if not mc.empty else "—")
    with c4:
        dv = filtered["sy_dist_ly"].dropna()
        st.metric("Median distance", f"{dv.median():,.1f} ly" if not dv.empty else "—")


# ------------------------------------------------------------------
# Discovery Trends
# ------------------------------------------------------------------
with tabs[1]:
    st.header("Discovery Trends")

    st.subheader("Discoveries per year")
    cumulative2 = st.toggle("Cumulative view", value=False, key="dt_cumulative")
    fig1, ins1 = chart_discoveries_per_year(
        df,
        cumulative=cumulative2,
        year_range=_none_if_empty(year_range) if not year_range else year_range,
        methods=_none_if_empty(methods_sel),
    )
    st.plotly_chart(fig1, key="discoveries_per_year_chart_2", width="stretch")
    st.caption(ins1)

    st.divider()
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Discovery method distribution")
        fig2, ins2 = chart_method_distribution(
            df,
            methods=_none_if_empty(methods_sel),
        )
        st.plotly_chart(fig2, width="stretch")
        st.caption(ins2)
    with c_right:
        st.subheader("Method share (donut)")
        fig2b, ins2b = chart_method_donut(
            df,
            methods=_none_if_empty(methods_sel),
        )
        st.plotly_chart(fig2b, width="stretch")
        st.caption(ins2b)

    st.divider()
    st.subheader("Discoveries by mission / telescope (top 10)")
    fig3, ins3 = chart_mission_top10(
        df,
        methods=_none_if_empty(methods_sel),
    )
    st.plotly_chart(fig3, width="stretch")
    st.caption(ins3)

    st.divider()
    st.subheader("Discovery method stacked area over time")
    fig8, ins8 = chart_method_stacked_area(
        df,
        year_range=_none_if_empty(year_range) if not year_range else year_range,
    )
    st.plotly_chart(fig8, width="stretch")
    st.caption(ins8)


# ------------------------------------------------------------------
# Planet Properties
# ------------------------------------------------------------------
with tabs[2]:
    st.header("Planet Properties")

    st.subheader("Mass vs Radius (log-log)")
    fig4, ins4 = chart_mass_radius(
        df,
        selected_name=st.session_state.get("selected_planet"),
        year_range=_none_if_empty(year_range) if not year_range else year_range,
        methods=_none_if_empty(methods_sel),
        planet_types=_none_if_empty(planet_types_sel),
        temp_range=_none_if_empty(temp_range) if not temp_range else temp_range,
        distance_range=_none_if_empty(distance_range) if not distance_range else distance_range,
        missions=_none_if_empty(missions_sel),
    )
    st.plotly_chart(fig4, width="stretch")
    st.caption(ins4)

    st.divider()
    col_h, col_b = st.columns([1.2, 1])
    with col_h:
        st.subheader("Equilibrium temperature distribution")
        fig5, ins5 = chart_temperature_hist(
            df,
            year_range=_none_if_empty(year_range) if not year_range else year_range,
            methods=_none_if_empty(methods_sel),
            planet_types=_none_if_empty(planet_types_sel),
            temp_range=_none_if_empty(temp_range) if not temp_range else temp_range,
            distance_range=_none_if_empty(distance_range) if not distance_range else distance_range,
            missions=_none_if_empty(missions_sel),
        )
        st.plotly_chart(fig5, width="stretch")
        st.caption(ins5)
    with col_b:
        st.subheader("Temperature by planet type")
        fig5b, ins5b = chart_temperature_box(
            df,
            year_range=_none_if_empty(year_range) if not year_range else year_range,
            methods=_none_if_empty(methods_sel),
            planet_types=_none_if_empty(planet_types_sel),
            temp_range=_none_if_empty(temp_range) if not temp_range else temp_range,
            distance_range=_none_if_empty(distance_range) if not distance_range else distance_range,
            missions=_none_if_empty(missions_sel),
        )
        st.plotly_chart(fig5b, width="stretch")
        st.caption(ins5b)

    st.divider()
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Distance distribution (light-years, log x-axis)")
        fig6, ins6 = chart_distance_hist(
            df,
            year_range=_none_if_empty(year_range) if not year_range else year_range,
            methods=_none_if_empty(methods_sel),
            planet_types=_none_if_empty(planet_types_sel),
            temp_range=_none_if_empty(temp_range) if not temp_range else temp_range,
            distance_range=_none_if_empty(distance_range) if not distance_range else distance_range,
            missions=_none_if_empty(missions_sel),
        )
        st.plotly_chart(fig6, width="stretch")
        st.caption(ins6)
    with c_right:
        st.subheader("Orbital period distribution (log x-axis)")
        fig7, ins7 = chart_period_hist(
            df,
            year_range=_none_if_empty(year_range) if not year_range else year_range,
            methods=_none_if_empty(methods_sel),
            planet_types=_none_if_empty(planet_types_sel),
            temp_range=_none_if_empty(temp_range) if not temp_range else temp_range,
            distance_range=_none_if_empty(distance_range) if not distance_range else distance_range,
            missions=_none_if_empty(missions_sel),
        )
        st.plotly_chart(fig7, width="stretch")
        st.caption(ins7)

    st.divider()
    st.subheader("Correlation heatmap of numeric parameters")
    fig9, ins9 = chart_correlation_heatmap(df)
    st.plotly_chart(fig9, width="stretch")
    st.caption(ins9)

    st.divider()
    st.subheader("Habitable-zone-style view (star temp vs orbital distance)")
    fig10, ins10 = chart_habitable_zone(
        df,
        year_range=_none_if_empty(year_range) if not year_range else year_range,
        methods=_none_if_empty(methods_sel),
        planet_types=_none_if_empty(planet_types_sel),
        temp_range=_none_if_empty(temp_range) if not temp_range else temp_range,
        distance_range=_none_if_empty(distance_range) if not distance_range else distance_range,
        missions=_none_if_empty(missions_sel),
    )
    st.plotly_chart(fig10, width="stretch")
    st.caption(ins10)


# ------------------------------------------------------------------
# Forecast
# ------------------------------------------------------------------
with tabs[3]:
    st.header("Forecast")

    st.caption(
        "Forecasts are trained on **completed years only**; the current partial "
        "year is excluded from training and flagged on the chart."
    )

    forecast_method = st.selectbox(
        "Forecast model",
        options=["holt", "arima", "prophet"],
        format_func=lambda m: {
            "holt": "Holt's linear trend (ExponentialSmoothing)",
            "arima": "ARIMA (statsmodels)",
            "prophet": "Prophet (optional)",
        }[m],
        index=0,
    )

    # Disable Prophet if it isn't installed.
    try:
        import prophet  # noqa: F401
        prophet_available = True
    except Exception:  # noqa: BLE001
        prophet_available = False
    if not prophet_available and forecast_method == "prophet":
        st.warning("Prophet is not installed; switching to Holt.")
        forecast_method = "holt"

    horizon = st.slider("Forecast horizon (years)", 1, 10, 5)

    if st.button("Run forecast", type="primary"):
        result = forecast(
            df,
            method=forecast_method,
            horizon=horizon,
            current_year=current_year,
        )

        if result["figure"] is None:
            st.error(result.get("note") or "Forecast produced no chart.")
            st.stop()

        st.plotly_chart(result["figure"], width="stretch")

        # Backtest metrics.
        st.divider()
        st.subheader("Backtest (train on N−3 years, test on last 3 years)")
        bt = result.get("backtest", {})
        if bt:
            if bt.get("note"):
                st.info(bt["note"])
            if bt.get("train_years") and bt.get("test_years"):
                st.caption(
                    f"Train years: {bt['train_years']} · "
                    f"Test years: {bt['test_years']} · "
                    f"Actual test counts: {bt.get('actual')}"
                )
            for model_key, label in [("holt", "Holt"), ("arima", "ARIMA")]:
                m = bt.get(model_key)
                if m is None:
                    continue
                if "error" in m:
                    st.caption(f"{label}: backtest unavailable ({m['error']})")
                else:
                    st.caption(
                        f"**{label}**: MAE = {m.get('mae')} · "
                        f"RMSE = {m.get('rmse')}"
                        + (f" · order {m.get('order')}" if m.get('order') else "")
                    )
        else:
            st.caption("Not enough history for a 3-year backtest.")

        # Cumulative projection.
        st.divider()
        st.subheader("Cumulative total projection")
        cum = result.get("cumulative", [])
        if cum:
            # Show milestone rows only (not per-year rows) in a tidy table.
            milestone_rows = [r for r in cum if not r.get("_is_year_row")]
            if milestone_rows:
                import pandas as _pd
                tdf = _pd.DataFrame(milestone_rows)
                tdf = tdf.rename(columns={
                    "threshold": "Threshold (planets)",
                    "projected_year": "Projected year",
                    "projected_total": "Projected total",
                })
                tdf = tdf[tdf["Projected year"].notna()]
                if not tdf.empty:
                    st.dataframe(tdf, hide_index=True, width="stretch")
                    top_milestone = milestone_rows[0]
                    if top_milestone.get("projected_year"):
                        st.success(
                            f"Expected to pass **{top_milestone['threshold']:,}** "
                            f"confirmed planets by **{top_milestone['projected_year']}** "
                            f"(projected total ≈ {top_milestone['projected_total']:,})."
                        )
            # Per-year projected cumulative as a small table.
            year_rows = [r for r in cum if r.get("_is_year_row")]
            if year_rows:
                import pandas as _pd2
                ydf = _pd2.DataFrame(year_rows)
                ydf = ydf.rename(columns={
                    "projected_year": "Year",
                    "projected_total": "Projected cumulative total",
                })
                ydf = ydf[["Year", "Projected cumulative total"]]
                st.caption("Projected cumulative total by year:")
                st.dataframe(ydf, hide_index=True, width="stretch")
        else:
            st.caption("No cumulative projection available.")

        # Caveat.
        st.divider()
        st.warning(result.get("insight", ""))
    else:
        st.info("Press **Run forecast** to generate the forecast chart and metrics.")


# ------------------------------------------------------------------
# Planet Explorer
# ------------------------------------------------------------------
with tabs[4]:
    st.header("Planet Explorer")

    st.subheader("Search for a planet")
    planet_names = sorted([str(n) for n in df["pl_name"].dropna().unique()])
    selected_name = st.selectbox(
        "Planet (autocomplete)",
        options=planet_names,
        index=None,
        format_func=lambda x: x,
    )

    if selected_name:
        card = planet_info_card(df, selected_name)
        if card:
            st.divider()
            st.subheader(f"Planet Information Card — {card['pl_name']}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Host star:** {card['hostname'] or '—'}")
                st.markdown(f"**Discovery year:** {int(card['disc_year']) if pd.notna(card['disc_year']) else '—'}")
                st.markdown(f"**Discovery method:** {card['discoverymethod'] or '—'}")
                st.markdown(f"**Mission / facility:** {card['mission'] or '—'}")
                st.markdown(f"**Distance:** {card['sy_dist_ly']:,.1f} light-years" if pd.notna(card['sy_dist_ly']) else "**Distance:** —")
            with c2:
                st.markdown(f"**Planet type:** {card['planet_type'] or '—'}")
                st.markdown(f"**Temperature class:** {card['temp_class'] or '—'}")
                st.markdown(f"**Mass:** {card['pl_bmasse']:,.2f} Earth masses" if pd.notna(card['pl_bmasse']) else "**Mass:** —")
                st.markdown(f"**Orbital period:** {card['pl_orbper']:,.1f} days" if pd.notna(card['pl_orbper']) else "**Orbital period:** —")
                st.markdown(f"**Equilibrium temperature:** {card['pl_eqt']:,.0f} K" if pd.notna(card['pl_eqt']) else "**Temperature:** —")

            st.divider()
            st.markdown("### Comparison with Earth")
            for line in earth_comparison(card):
                st.success(line)

            # Copy of mass-radius scatter with the selected planet highlighted.
            st.divider()
            st.subheader("Position on the mass–radius scatter")
            fig_sel, _ = chart_mass_radius(
                df,
                selected_name=selected_name,
                year_range=_none_if_empty(year_range) if not year_range else year_range,
                methods=_none_if_empty(methods_sel),
                planet_types=_none_if_empty(planet_types_sel),
                temp_range=_none_if_empty(temp_range) if not temp_range else temp_range,
                distance_range=_none_if_empty(distance_range) if not distance_range else distance_range,
                missions=_none_if_empty(missions_sel),
            )
            st.plotly_chart(fig_sel, width="stretch")

    st.divider()
    st.subheader("Compare planets")
    compare_mode = st.toggle("Enable compare mode", value=False)
    if compare_mode:
        max_compare = 3
        compare_names: list[str] = []
        for i in range(max_compare):
            val = st.selectbox(
                f"Planet {i+1}",
                options=["(none)"] + planet_names,
                index=0,
                key=f"cmp_{i}",
            )
            if val and val != "(none)":
                compare_names.append(val)
        if len(compare_names) >= 2:
            rows: list[dict[str, Any]] = []
            for name in compare_names:
                card = planet_info_card(df, name)
                if card:
                    rows.append({
                        "Planet": card["pl_name"],
                        "Host star": card["hostname"] or "—",
                        "Year": int(card['disc_year']) if pd.notna(card['disc_year']) else "—",
                        "Method": card["discoverymethod"] or "—",
                        "Radius (R⊕)": f"{card['pl_rade']:.2f}" if pd.notna(card['pl_rade']) else "—",
                        "Mass (M⊕)": f"{card['pl_bmasse']:.2f}" if pd.notna(card['pl_bmasse']) else "—",
                        "Period (days)": f"{card['pl_orbper']:.1f}" if pd.notna(card['pl_orbper']) else "—",
                        "Distance (ly)": f"{card['sy_dist_ly']:,.1f}" if pd.notna(card['sy_dist_ly']) else "—",
                        "Temp (K)": f"{card['pl_eqt']:,.0f}" if pd.notna(card['pl_eqt']) else "—",
                        "Type": card["planet_type"] or "—",
                    })
            if rows:
                import pandas as _pd
                cdf = _pd.DataFrame(rows)
                st.dataframe(cdf, hide_index=True, width="stretch")
        else:
            st.caption("Select at least 2 planets to compare.")


# ------------------------------------------------------------------
# Data Quality
# ------------------------------------------------------------------
with tabs[5]:
    st.header("Data Quality")

    st.markdown(
        "Each chart drops missing values **per chart** rather than globally, "
        "so the dashboard keeps as many planets visible as possible. Only rows "
        "missing `pl_name` are dropped globally (a planet with no name has no identity)."
    )

    report = build_data_quality_report(df_raw, df)
    with st.expander("Missing-value report (per column)", expanded=True):            st.dataframe(report, hide_index=True, width="stretch")


    st.divider()
    st.markdown("### Cleaning steps")
    st.markdown(f"- **Rows before cleaning:** {len(df_raw):,}")
    st.markdown(f"- **Drop rows missing `pl_name`:** "
                f"{len(df_raw):,} → {len(df):,} "
                f"(-{len(df_raw) - len(df):,})")
    st.markdown("- **Keep NaNs** in all other columns (drop per chart as needed).")
    st.markdown(f"- **Distance conversion:** `sy_dist` (parsecs) → `sy_dist_ly` (light-years, × {3.2616}).")
    st.markdown("- **Planet type** derived from `pl_rade`:")
    st.markdown("  - < 1.25 → Earth-like / Rocky")
    st.markdown("  - 1.25–2 → Super-Earth")
    st.markdown("  - 2–6 → Neptune-like")
    st.markdown("  - > 6 → Jupiter-like / Gas Giant")
    st.markdown("- **Temperature class** derived from `pl_eqt` (K):")
    st.markdown("  - < 250 → Cold · 250–350 → Temperate · 350–1000 → Warm · > 1000 → Hot")
    st.markdown("- **Mission** derived from `disc_facility` (top 10 + Other).")
    st.markdown("- **`disc_year`** coerced to numeric; current partial year flagged for forecasting.")
    st.markdown(f"- **Current (partial) year used for forecast exclusion:** {current_year}.")

    st.divider()
    st.markdown("### Log-transform helpers")
    st.markdown(
        "Charts that use log axes (mass, period, distance) drop non-positive "
        "values to NaN before plotting so the log scale stays valid."
    )


# ---------------------------------------------------------------------------
# Download filtered data
# ---------------------------------------------------------------------------

st.divider()
st.caption("Download the currently filtered dataset as CSV.")
csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download filtered data (CSV)",
    data=csv_bytes,
    file_name="exoplanet_explorer_filtered.csv",
    mime="text/csv",
)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Data: NASA Exoplanet Archive (pscomppars), operated by Caltech under NASA "
    "contract. [Archive citation & terms](https://exoplanetarchive.ipac.caltech.edu/). "
    f"— Exoplanet Explorer · fetched {last_fetched or '—'}."
)
