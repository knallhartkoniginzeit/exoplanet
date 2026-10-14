"""
Plotly chart builders for the Exoplanet Explorer dashboard.

All charts use a consistent dark "space" theme and return Plotly figure
objects. Insight captions are returned alongside figures where a chart has
an associated one-line takeaway.

Charts intentionally drop missing values **per chart** (never globally) so
the dashboard keeps as many planets visible as possible.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

# Consistent dark space palette. Colors are chosen to read well on a dark
# background and to be distinguishable for color-blind viewers where
# practical.
THEME = {
    "bg_fill": "#0b0e17",
    "bg_plot": "#111622",
    "paper_bgcolor": "#0b0e17",
    "font_color": "#d7e0f0",
    "grid_color": "#232a3a",
    "axis_color": "#8a96b0",
    "title_color": "#eef3ff",
}

PLANET_TYPE_COLORS = {
    "Earth-like / Rocky": "#5fd4a0",
    "Super-Earth": "#7fc1ff",
    "Neptune-like": "#ffb347",
    "Jupiter-like / Gas Giant": "#ff7b7b",
}

DEFAULT_PLOTLY_TEMPLATE = "plotly_dark"

# Insight captions (associated with charts; updated as data changes).
INSIGHTS = {
    "discoveries_per_year": "Discoveries peaked after Kepler's 2014 data release and again with TESS; year-to-year counts are lumpy, not smooth.",
    "discoveries_cumulative": "The cumulative confirmed-planet curve steepens sharply after 2014, reflecting Kepler-era transit discoveries.",
    "method_distribution": "Transit dominates the modern catalog; Radial Velocity is the historic backbone; direct imaging and microlensing remain small.",
    "method_donut": "Transit accounts for the majority of confirmed planets, with RV second.",
    "mission_top10": "Kepler, TESS, and ground-based RV facilities drive most discoveries; the long tail is grouped as 'Other'.",
    "mass_radius": "Mass-radius space separates rocky, Neptune-like, and gas-giant populations; selection bias favors large, short-period planets.",
    "temperature_hist": "The temperature distribution is right-skewed toward hot planets; temperate worlds are comparatively rare in the confirmed catalog.",
    "temperature_box": "Median equilibrium temperature rises from rocky to gas-giant classes, but each class spans a wide range.",
    "distance_hist": "Most confirmed planets lie within a few thousand light-years; the long tail reflects nearby ground-based surveys.",
    "period_hist": "Orbital periods cluster at short durations (transit/RV bias); long-period planets are underrepresented.",
    "method_stacked_area": "Kepler (2014/2016) and TESS eras are visible as bright bands in the transit area; RV is steadier across decades.",
    "correlation_heatmap": "Expected correlations appear (e.g. radius vs mass, period vs semi-major axis); missing values reduce the effective sample per pair.",
    "habitable_zone": "Habitable-zone-style view: temperate planets tend to orbit cooler stars at wider separations, but detection bias shapes the pattern.",
}


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Apply the dashboard dark space theme to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor=THEME["paper_bgcolor"],
        plot_bgcolor=THEME["bg_plot"],
        font=dict(color=THEME["font_color"]),
        title=dict(font=dict(color=THEME["title_color"], size=16)),
        legend=dict(font=dict(color=THEME["font_color"]),
                     bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font=dict(color=THEME["font_color"]),
                        bgcolor="#1b2230"),
    )
    fig.update_xaxes(
        gridcolor=THEME["grid_color"],
        zerolinecolor=THEME["grid_color"],
        linecolor=THEME["axis_color"],
        tickfont=dict(color=THEME["font_color"]),
        title_font=dict(color=THEME["font_color"]),
    )
    fig.update_yaxes(
        gridcolor=THEME["grid_color"],
        zerolinecolor=THEME["grid_color"],
        linecolor=THEME["axis_color"],
        tickfont=dict(color=THEME["font_color"]),
        title_font=dict(color=THEME["font_color"]),
    )
    return fig


def _color_map_for_types(df: pd.DataFrame) -> dict[str, str]:
    types_present = []
    if "planet_type" in df.columns:
        types_present = [t for t in df["planet_type"].dropna().unique()
                         if isinstance(t, str)]
    return {t: PLANET_TYPE_COLORS.get(t, "#9aa6c0") for t in types_present}


# ---------------------------------------------------------------------------
# Chart 1: Discoveries per year (line + cumulative toggle)
# ---------------------------------------------------------------------------

def chart_discoveries_per_year(
    df: pd.DataFrame,
    cumulative: bool = False,
    year_range: Tuple[Optional[int], Optional[int]] = (None, None),
    methods: Optional[list[str]] = None,
) -> Tuple[go.Figure, str]:
    """Discoveries per year: line chart with optional cumulative view."""
    sub = df.dropna(subset=["disc_year"]).copy()
    sub["disc_year"] = sub["disc_year"].astype(int)

    if year_range[0] is not None or year_range[1] is not None:
        lo, hi = year_range
        sub = sub[(sub["disc_year"] >= (lo or -np.inf)) &
                  (sub["disc_year"] <= (hi or np.inf))]

    if methods:
        sub = sub[sub["discoverymethod"].isin(methods)]

    yearly = sub.groupby("disc_year").size().reset_index(name="count")
    yearly = yearly.sort_values("disc_year")

    if cumulative:
        yearly["count"] = yearly["count"].cumsum()

    fig = px.line(
        yearly,
        x="disc_year",
        y="count",
        markers=True,
        title="Discoveries per year" + (" (cumulative)" if cumulative else ""),
        labels={"disc_year": "Discovery year", "count": "Confirmed planets"},
    )
    fig.update_traces(line=dict(width=2.5), marker=dict(size=6))
    fig = apply_dark_theme(fig)
    fig.update_xaxes(type="linear", dtick=1)
    insight = INSIGHTS["discoveries_cumulative" if cumulative
                          else "discoveries_per_year"]
    return fig, insight


# ---------------------------------------------------------------------------
# Chart 2: Discovery method distribution (horizontal bar)
# ---------------------------------------------------------------------------

def chart_method_distribution(
    df: pd.DataFrame,
    methods: Optional[list[str]] = None,
) -> Tuple[go.Figure, str]:
    """Discovery method distribution: horizontal bar chart."""
    sub = df.dropna(subset=["discoverymethod"]).copy()
    counts = sub["discoverymethod"].value_counts().reset_index()
    counts.columns = ["method", "count"]
    counts = counts.sort_values("count", ascending=True)

    fig = px.bar(
        counts,
        x="count",
        y="method",
        orientation="h",
        title="Discovery method distribution",
        labels={"method": "Discovery method", "count": "Confirmed planets"},
    )
    fig.update_traces(marker=dict(line=dict(color="#0b0e17", width=1)))
    fig = apply_dark_theme(fig)
    fig.update_layout(xaxis_tickformat=",d")
    insight = INSIGHTS["method_distribution"]
    return fig, insight


# ---------------------------------------------------------------------------
# Chart 2b: Method donut (optional)
# ---------------------------------------------------------------------------

def chart_method_donut(
    df: pd.DataFrame,
    methods: Optional[list[str]] = None,
) -> Tuple[go.Figure, str]:
    """Discovery method donut chart (optional companion to the bar chart)."""
    sub = df.dropna(subset=["discoverymethod"]).copy()
    counts = sub["discoverymethod"].value_counts().reset_index()
    counts.columns = ["method", "count"]

    fig = px.pie(
        counts,
        values="count",
        names="method",
        title="Discovery method share",
        hole=0.45,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label",
                      marker=dict(line=dict(color="#0b0e17", width=1.5)))
    fig = apply_dark_theme(fig)
    fig.update_layout(legend=dict(font=dict(color=THEME["font_color"])))
    insight = INSIGHTS["method_donut"]
    return fig, insight


# ---------------------------------------------------------------------------
# Chart 3: Discoveries by mission/telescope (top 10)
# ---------------------------------------------------------------------------

def chart_mission_top10(
    df: pd.DataFrame,
    methods: Optional[list[str]] = None,
) -> Tuple[go.Figure, str]:
    """Discoveries by mission/telescope: top 10 bar chart."""
    sub = df.dropna(subset=["mission"]).copy()
    if methods:
        sub = sub[sub["discoverymethod"].isin(methods)]
    counts = sub["mission"].value_counts().head(10).reset_index()
    counts.columns = ["mission", "count"]
    counts = counts.sort_values("count", ascending=True)

    fig = px.bar(
        counts,
        x="count",
        y="mission",
        orientation="h",
        title="Discoveries by mission / telescope (top 10)",
        labels={"mission": "Mission / facility", "count": "Confirmed planets"},
    )
    fig.update_traces(marker=dict(line=dict(color="#0b0e17", width=1)))
    fig = apply_dark_theme(fig)
    fig.update_layout(xaxis_tickformat=",d")
    insight = INSIGHTS["mission_top10"]
    return fig, insight


# ---------------------------------------------------------------------------
# Chart 4: Mass vs Radius scatter (log-log, colored by planet_type)
# ---------------------------------------------------------------------------

def chart_mass_radius(
    df: pd.DataFrame,
    selected_name: Optional[str] = None,
    highlight_name: Optional[str] = None,
    year_range: Tuple[Optional[int], Optional[int]] = (None, None),
    methods: Optional[list[str]] = None,
    planet_types: Optional[list[str]] = None,
    temp_range: Tuple[Optional[float], Optional[float]] = (None, None),
    distance_range: Tuple[Optional[float], Optional[float]] = (None, None),
    missions: Optional[list[str]] = None,
) -> Tuple[go.Figure, str]:
    """Mass vs Radius scatter, log-log axes, colored by planet_type, sized by period."""
    sub = _apply_filters(
        df,
        year_range=year_range,
        methods=methods,
        planet_types=planet_types,
        temp_range=temp_range,
        distance_range=distance_range,
        missions=missions,
    )
    sub = sub.dropna(subset=["pl_rade", "pl_bmasse", "pl_orbper",
                             "planet_type"]).copy()
    sub["log_mass"] = np.log10(sub["pl_bmasse"])
    sub["log_radius"] = np.log10(sub["pl_rade"])
    sub["period_display"] = sub["pl_orbper"]

    color_map = _color_map_for_types(sub)

    fig = px.scatter(
        sub,
        x="log_radius",
        y="log_mass",
        color="planet_type",
        size="pl_orbper",
        size_max=14,
        hover_data={
            "pl_name": True,
            "hostname": True,
            "discoverymethod": True,
            "pl_rade": ":,.2f",
            "pl_bmasse": ":,.2f",
            "pl_orbper": ":,.1f",
            "pl_eqt": ":,.0f",
            "sy_dist_ly": ":,.1f",
            "pl_orbper": ":,.1f",
        },
        title="Mass vs Radius (log-log, colored by planet type)",
        labels={
            "log_radius": "log10(radius / Earth radii)",
            "log_mass": "log10(mass / Earth masses)",
            "planet_type": "Planet type",
            "period_display": "Orbital period (days)",
        },
    )
    # Highlight a selected planet.
    if selected_name and selected_name in sub["pl_name"].values:
        row = sub.loc[sub["pl_name"] == selected_name].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[row["log_radius"]],
                y=[row["log_mass"]],
                mode="markers",
                marker=dict(symbol="star", size=16, color="#ffd34d",
                            line=dict(color="#0b0e17", width=1.5)),
                name=f"Selected: {selected_name}",
                hoverinfo="text",
                text=[f"<b>{selected_name}</b><br>{row['hostname']}<br>"
                      f"Radius: {row['pl_rade']:.2f} R⊕ | "
                      f"Mass: {row['pl_bmasse']:.2f} M⊕"],
            )
        )

    fig = apply_dark_theme(fig)
    fig.update_xaxes(title="log10(radius / Earth radii)")
    fig.update_yaxes(title="log10(mass / Earth masses)")

    insight = INSIGHTS["mass_radius"]
    return fig, insight


# ---------------------------------------------------------------------------
# Chart 5: Temperature distribution (histogram) + box by planet_type
# ---------------------------------------------------------------------------

def chart_temperature_hist(
    df: pd.DataFrame,
    year_range: Tuple[Optional[int], Optional[int]] = (None, None),
    methods: Optional[list[str]] = None,
    planet_types: Optional[list[str]] = None,
    temp_range: Tuple[Optional[float], Optional[float]] = (None, None),
    distance_range: Tuple[Optional[float], Optional[float]] = (None, None),
    missions: Optional[list[str]] = None,
) -> Tuple[go.Figure, str]:
    """Temperature distribution: histogram of equilibrium temperature (K)."""
    sub = _apply_filters(
        df,
        year_range=year_range,
        methods=methods,
        planet_types=planet_types,
        temp_range=temp_range,
        distance_range=distance_range,
        missions=missions,
    )
    sub = sub.dropna(subset=["pl_eqt"]).copy()

    fig = px.histogram(
        sub,
        x="pl_eqt",
        nbins=40,
        title="Equilibrium temperature distribution",
        labels={"pl_eqt": "Equilibrium temperature (K)", "count": "Planets"},
    )
    fig.update_traces(marker=dict(line=dict(color="#0b0e17", width=0.5)))
    fig = apply_dark_theme(fig)
    insight = INSIGHTS["temperature_hist"]
    return fig, insight


def chart_temperature_box(
    df: pd.DataFrame,
    year_range: Tuple[Optional[int], Optional[int]] = (None, None),
    methods: Optional[list[str]] = None,
    planet_types: Optional[list[str]] = None,
    temp_range: Tuple[Optional[float], Optional[float]] = (None, None),
    distance_range: Tuple[Optional[float], Optional[float]] = (None, None),
    missions: Optional[list[str]] = None,
) -> Tuple[go.Figure, str]:
    """Temperature box plot by planet_type."""
    sub = _apply_filters(
        df,
        year_range=year_range,
        methods=methods,
        planet_types=planet_types,
        temp_range=temp_range,
        distance_range=distance_range,
        missions=missions,
    )
    sub = sub.dropna(subset=["pl_eqt", "planet_type"]).copy()

    color_map = _color_map_for_types(sub)

    fig = px.box(
        sub,
        x="planet_type",
        y="pl_eqt",
        title="Equilibrium temperature by planet type",
        labels={
            "planet_type": "Planet type",
            "pl_eqt": "Equilibrium temperature (K)",
        },
    )
    fig.update_traces(
        marker=dict(color="#7fc1ff"),
        line=dict(color="#d7e0f0"),
    )
    fig = apply_dark_theme(fig)
    insight = INSIGHTS["temperature_box"]
    return fig, insight


# ---------------------------------------------------------------------------
# Chart 6: Distance distribution (log x-axis, light-years)
# ---------------------------------------------------------------------------

def chart_distance_hist(
    df: pd.DataFrame,
    year_range: Tuple[Optional[int], Optional[int]] = (None, None),
    methods: Optional[list[str]] = None,
    planet_types: Optional[list[str]] = None,
    temp_range: Tuple[Optional[float], Optional[float]] = (None, None),
    distance_range: Tuple[Optional[float], Optional[float]] = (None, None),
    missions: Optional[list[str]] = None,
) -> Tuple[go.Figure, str]:
    """Distance distribution: histogram with log x-axis, in light-years."""
    sub = _apply_filters(
        df,
        year_range=year_range,
        methods=methods,
        planet_types=planet_types,
        temp_range=temp_range,
        distance_range=distance_range,
        missions=missions,
    )
    sub = sub.dropna(subset=["sy_dist_ly"]).copy()

    fig = px.histogram(
        sub,
        x="sy_dist_ly",
        nbins=40,
        title="Distance distribution (light-years, log x-axis)",
        labels={"sy_dist_ly": "Distance (light-years)", "count": "Planets"},
    )
    fig.update_traces(marker=dict(line=dict(color="#0b0e17", width=0.5)))
    fig = apply_dark_theme(fig)
    fig.update_xaxes(type="log", title="Distance (light-years, log scale)")
    insight = INSIGHTS["distance_hist"]
    return fig, insight


# ---------------------------------------------------------------------------
# Chart 7: Orbital period distribution (log x-axis)
# ---------------------------------------------------------------------------

def chart_period_hist(
    df: pd.DataFrame,
    year_range: Tuple[Optional[int], Optional[int]] = (None, None),
    methods: Optional[list[str]] = None,
    planet_types: Optional[list[str]] = None,
    temp_range: Tuple[Optional[float], Optional[float]] = (None, None),
    distance_range: Tuple[Optional[float], Optional[float]] = (None, None),
    missions: Optional[list[str]] = None,
) -> Tuple[go.Figure, str]:
    """Orbital period distribution: histogram with log x-axis."""
    sub = _apply_filters(
        df,
        year_range=year_range,
        methods=methods,
        planet_types=planet_types,
        temp_range=temp_range,
        distance_range=distance_range,
        missions=missions,
    )
    sub = sub.dropna(subset=["pl_orbper"]).copy()
    # Periods are positive; log scale is safe after dropping non-positive.
    sub = sub[sub["pl_orbper"] > 0]

    fig = px.histogram(
        sub,
        x="pl_orbper",
        nbins=40,
        title="Orbital period distribution (log x-axis)",
        labels={"pl_orbper": "Orbital period (days)", "count": "Planets"},
    )
    fig.update_traces(marker=dict(line=dict(color="#0b0e17", width=0.5)))
    fig = apply_dark_theme(fig)
    fig.update_xaxes(type="log", title="Orbital period (days, log scale)")
    insight = INSIGHTS["period_hist"]
    return fig, insight


# ---------------------------------------------------------------------------
# Chart 8 (bonus): Discovery method stacked area over time
# ---------------------------------------------------------------------------

def chart_method_stacked_area(
    df: pd.DataFrame,
    year_range: Tuple[Optional[int], Optional[int]] = (None, None),
) -> Tuple[go.Figure, str]:
    """Discovery method stacked area over time (shows Kepler/TESS eras)."""
    sub = df.dropna(subset=["disc_year", "discoverymethod"]).copy()
    sub["disc_year"] = sub["disc_year"].astype(int)

    if year_range[0] is not None or year_range[1] is not None:
        lo, hi = year_range
        sub = sub[(sub["disc_year"] >= (lo or -np.inf)) &
                  (sub["disc_year"] <= (hi or np.inf))]

    pivot = (
        sub.groupby(["disc_year", "discoverymethod"])
        .size()
        .reset_index(name="count")
        .pivot(index="disc_year", columns="discoverymethod", values="count")
        .fillna(0)
    )
    pivot = pivot.sort_index()

    fig = go.Figure()
    colors = px.colors.qualitative.Pastel
    for i, col in enumerate(pivot.columns):
        fig.add_trace(
            go.Scatter(
                x=pivot.index,
                y=pivot[col].values,
                mode="none",
                stackgroup="one",
                name=str(col),
                line=dict(width=0.8, color=colors[i % len(colors)]),
                fillcolor=colors[i % len(colors)],
            )
        )
    fig.update_layout(
        title="Discovery method stacked area over time",
        xaxis_title="Discovery year",
        yaxis_title="Confirmed planets",
        xaxis=dict(dtick=1, type="linear"),
    )
    fig = apply_dark_theme(fig)
    fig.update_yaxes(tickformat=",d")
    insight = INSIGHTS["method_stacked_area"]
    return fig, insight


# ---------------------------------------------------------------------------
# Chart 9 (bonus): Correlation heatmap of numeric columns
# ---------------------------------------------------------------------------

_NUMERIC_COLUMNS = [
    "pl_rade", "pl_bmasse", "pl_orbper", "sy_dist", "pl_eqt",
    "st_teff", "st_rad", "pl_orbsmax", "sy_pnum",
]


def chart_correlation_heatmap(
    df: pd.DataFrame,
) -> Tuple[go.Figure, str]:
    """Correlation heatmap of numeric columns (Plotly)."""
    num = df[_NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    corr = num.corr(numeric_only=True)

    # Order columns for readability.
    order = ["pl_rade", "pl_bmasse", "pl_orbper", "pl_eqt",
             "st_teff", "st_rad", "pl_orbsmax", "sy_dist", "sy_pnum"]
    corr = corr.reindex(index=order, columns=order)

    text = corr.map(lambda v: f"{v:.2f}" if pd.notna(v) else "")

    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        labels=dict(x="Variable", y="Variable", color="Correlation"),
        title="Correlation heatmap of numeric parameters",
    )
    fig.update_traces(text=text, texttemplate="%{text}", textfont=dict(size=9))
    fig.update_xaxes(tickangle=45)
    fig = apply_dark_theme(fig)
    fig.update_layout(coloraxis_colorbar=dict(
        title="Correlation",
        tickfont=dict(color=THEME["font_color"]),
    ))
    insight = INSIGHTS["correlation_heatmap"]
    return fig, insight


# ---------------------------------------------------------------------------
# Chart 10 (bonus): Habitable-zone-style scatter (star temp vs orbital distance)
# ---------------------------------------------------------------------------

def chart_habitable_zone(
    df: pd.DataFrame,
    year_range: Tuple[Optional[int], Optional[int]] = (None, None),
    methods: Optional[list[str]] = None,
    planet_types: Optional[list[str]] = None,
    temp_range: Tuple[Optional[float], Optional[float]] = (None, None),
    distance_range: Tuple[Optional[float], Optional[float]] = (None, None),
    missions: Optional[list[str]] = None,
) -> Tuple[go.Figure, str]:
    """Habitable-zone-style scatter: star temperature vs orbital distance (AU)."""
    sub = _apply_filters(
        df,
        year_range=year_range,
        methods=methods,
        planet_types=planet_types,
        temp_range=temp_range,
        distance_range=distance_range,
        missions=missions,
    )
    sub = sub.dropna(subset=["st_teff", "pl_orbsmax"]).copy()
    sub = sub[(sub["st_teff"] > 0) & (sub["pl_orbsmax"] > 0)]
    # Marker size comes from orbital period; give missing periods a small
    # constant so Plotly never receives NaN sizes (which raise ValueError).
    sub["marker_size"] = sub["pl_orbper"].fillna(1.0).clip(lower=1.0)

    color_map = _color_map_for_types(sub)

    fig = px.scatter(
        sub,
        x="st_teff",
        y="pl_orbsmax",
        color="planet_type",
        size="marker_size",
        size_max=14,
        hover_data={
            "pl_name": True,
            "hostname": True,
            "discoverymethod": True,
            "pl_rade": ":,.2f",
            "pl_bmasse": ":,.2f",
            "pl_eqt": ":,.0f",
            "sy_dist_ly": ":,.1f",
        },
        title="Habitable-zone-style view: star temperature vs orbital distance",
        labels={
            "st_teff": "Star temperature (K)",
            "pl_orbsmax": "Orbital distance (AU)",
            "planet_type": "Planet type",
        },
    )
    fig.for_each_trace(
        lambda t: t.update(marker=dict(
            color=color_map.get(t.name, "#9aa6c0"),
            line=dict(color="#0b0e17", width=0.6),
            opacity=0.8,
        ))
    )

    # Approximate optimistic habitable-zone band for a Sun-like reference
    # (not a rigorous calculation — illustrative only).
    fig.add_vrect(
        x0=4500, x1=7000,
        y0=0.8, y1=1.8,
        fillcolor="rgba(95, 212, 160, 0.12)",
        line=dict(color="rgba(95, 212, 160, 0.5)", width=1, dash="dot"),
        layer="below",
        annotation_text="Rough HZ reference (Sun-like)",
        annotation_position="top right",
        annotation_font=dict(color="#9be8c4", size=10),
    )

    fig = apply_dark_theme(fig)
    fig.update_xaxes(title="Star temperature (K)")
    fig.update_yaxes(title="Orbital distance (AU)")
    insight = INSIGHTS["habitable_zone"]
    return fig, insight


# ---------------------------------------------------------------------------
# Shared filter application
# ---------------------------------------------------------------------------

def _apply_filters(
    df: pd.DataFrame,
    year_range: Tuple[Optional[int], Optional[int]] = (None, None),
    methods: Optional[list[str]] = None,
    planet_types: Optional[list[str]] = None,
    temp_range: Tuple[Optional[float], Optional[float]] = (None, None),
    distance_range: Tuple[Optional[float], Optional[float]] = (None, None),
    missions: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Apply the dashboard's sidebar filters to a DataFrame copy."""
    out = df.copy()

    if year_range[0] is not None or year_range[1] is not None:
        lo, hi = year_range
        out = out[out["disc_year"].fillna(-np.inf) >= (lo or -np.inf)]
        out = out[out["disc_year"].fillna(np.inf) <= (hi or np.inf)]

    if methods:
        out = out[out["discoverymethod"].isin(methods)]

    if planet_types:
        out = out[out["planet_type"].isin(planet_types)]

    if temp_range[0] is not None or temp_range[1] is not None:
        lo, hi = temp_range
        out = out[out["pl_eqt"].fillna(-np.inf) >= (lo or -np.inf)]
        out = out[out["pl_eqt"].fillna(np.inf) <= (hi or np.inf)]

    if distance_range[0] is not None or distance_range[1] is not None:
        lo, hi = distance_range
        out = out[out["sy_dist_ly"].fillna(-np.inf) >= (lo or -np.inf)]
        out = out[out["sy_dist_ly"].fillna(np.inf) <= (hi or np.inf)]

    if missions:
        out = out[out["mission"].isin(missions)]

    return out


# ---------------------------------------------------------------------------
# Planet information card data
# ---------------------------------------------------------------------------

def planet_info_card(df: pd.DataFrame, pl_name: str) -> dict[str, Any] | None:
    """Return a dict of display fields for a planet, or None if not found."""
    match = df.loc[df["pl_name"] == pl_name]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        "pl_name": row.get("pl_name", ""),
        "hostname": row.get("hostname", ""),
        "disc_year": row.get("disc_year"),
        "discoverymethod": row.get("discoverymethod", ""),
        "pl_rade": row.get("pl_rade"),
        "pl_bmasse": row.get("pl_bmasse"),
        "pl_orbper": row.get("pl_orbper"),
        "sy_dist_ly": row.get("sy_dist_ly"),
        "pl_eqt": row.get("pl_eqt"),
        "disc_facility": row.get("disc_facility", ""),
        "disc_telescope": row.get("disc_telescope", ""),
        "mission": row.get("mission", ""),
        "planet_type": row.get("planet_type"),
        "temp_class": row.get("temp_class"),
        "st_teff": row.get("st_teff"),
        "pl_orbsmax": row.get("pl_orbsmax"),
    }


def earth_comparison(fields: dict[str, Any]) -> list[str]:
    """Build human-readable comparison strings against Earth."""
    lines: list[str] = []
    if fields.get("pl_rade") is not None and pd.notna(fields["pl_rade"]):
        r = float(fields["pl_rade"])
        lines.append(f"Radius: {r:.2f}× Earth's radius")
    if fields.get("pl_bmasse") is not None and pd.notna(fields["pl_bmasse"]):
        m = float(fields["pl_bmasse"])
        lines.append(f"Mass: {m:.2f}× Earth's mass")
    if fields.get("pl_orbper") is not None and pd.notna(fields["pl_orbper"]):
        p = float(fields["pl_orbper"])
        if p > 0:
            lines.append(f"Orbital period: {p:.1f} days")
        else:
            lines.append("Orbital period: n/a")
    if fields.get("sy_dist_ly") is not None and pd.notna(fields["sy_dist_ly"]):
        d = float(fields["sy_dist_ly"])
        lines.append(f"Distance: {d:,.1f} light-years")
    if fields.get("pl_eqt") is not None and pd.notna(fields["pl_eqt"]):
        t = float(fields["pl_eqt"])
        lines.append(f"Equilibrium temperature: {t:,.0f} K")
    if fields.get("planet_type") is not None and pd.notna(fields["planet_type"]):
        lines.append(f"Type: {fields['planet_type']}")
    if fields.get("temp_class") is not None and pd.notna(fields["temp_class"]):
        lines.append(f"Temperature class: {fields['temp_class']}")
    return lines
