"""
Preprocessing for the Exoplanet Explorer dashboard.

This module does **not** drop rows globally (except rows missing `pl_name`,
which have no identity). Missing values in other columns are kept so that
each chart can drop them per-chart as needed. Derived columns are added in
place to a working copy of the data.

All magic thresholds are defined as module-level constants so they can be
referenced in documentation and adjusted in one place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

# ---------------------------------------------------------------------------
# Thresholds / constants
# ---------------------------------------------------------------------------

# Conversion parsecs -> light-years.
PC_TO_LY = 3.2616

# Planet-type boundaries (radius in Earth radii).
RADIUS_EARTH_LIKE_MAX = 1.25   # < 1.25 -> Earth-like / Rocky
RADIUS_SUPER_EARTH_MAX = 2.0   # 1.25 - 2 -> Super-Earth
RADIUS_NEPTUNE_MAX = 6.0       # 2 - 6 -> Neptune-like
# > 6 -> Jupiter-like / Gas Giant

# Temperature-class boundaries (equilibrium temperature in K).
TEMP_COLD_MAX = 250.0
TEMP_TEMPERATE_MAX = 350.0
TEMP_WARM_MAX = 1000.0
# > 1000 -> Hot

# Mission grouping: number of top facilities to keep explicit.
MISSION_TOP_N = 10

# Discovery-year handling: the current calendar year is treated as a partial
# year for forecasting and is excluded from model training.
CURRENT_YEAR = None  # set at runtime by the caller (avoids hardcoding)


# ---------------------------------------------------------------------------
# Public preprocessing entry points
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame, current_year: Optional[int] = None) -> pd.DataFrame:
    """
    Build a cleaned, feature-rich working copy of the raw exoplanet table.

    Steps (each documented in `build_data_quality_report`):
    1. Drop rows missing ``pl_name`` (no identity -> cannot be shown).
    2. Ensure expected columns exist (missing columns become all-NaN).
    3. Convert ``sy_dist`` parsecs -> light-years, keep both.
    4. Derive ``planet_type`` from ``pl_rade``.
    5. Derive ``temp_class`` from ``pl_eqt``.
    6. Derive ``mission`` from ``disc_facility`` (top-N + "Other").
    7. Set ``CURRENT_YEAR`` for forecast helpers.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data from the NASA Exoplanet Archive (or snapshot).
    current_year : int, optional
        Calendar year to treat as "current/partial". Defaults to the current
        calendar year at call time.

    Returns
    -------
    pd.DataFrame
        Working copy with derived columns added. The original ``df`` is not
        modified.
    """
    global CURRENT_YEAR
    if current_year is None:
        current_year = pd.Timestamp.now().year
    CURRENT_YEAR = current_year

    out = df.copy()
    out = _ensure_columns(out)
    out = _drop_rows_missing_name(out)
    out = _convert_distance(out)
    out = _derive_planet_type(out)
    out = _derive_temp_class(out)
    out = _derive_mission(out)
    out = _standardize_year(out)
    return out


def build_data_quality_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Build a tidy DataFrame summarizing data quality for the Data Quality tab.

    Reports, per column:
    - ``column``
    - ``missing_in_raw`` (count)
    - ``missing_pct_raw`` (%)
    - ``non_missing``
    - ``note``: a short human-readable note about the column's role.

    Also includes summary rows for row counts before/after cleaning.
    """
    rows = []
    raw_cols = set(df_raw.columns)
    clean_cols = set(df_clean.columns)

    # Row-count summary.
    rows.append({
        "column": "ROW_COUNT_BEFORE",
        "missing_in_raw": len(df_raw),
        "missing_pct_raw": np.nan,
        "non_missing": np.nan,
        "note": "Rows in raw API/snapshot response (before dropping missing pl_name).",
    })
    rows.append({
        "column": "ROW_COUNT_AFTER",
        "missing_in_raw": len(df_clean),
        "missing_pct_raw": np.nan,
        "non_missing": np.nan,
        "note": "Rows after dropping rows with missing pl_name.",
    })
    rows.append({
        "column": "ROWS_DROPPED",
        "missing_in_raw": len(df_raw) - len(df_clean),
        "missing_pct_raw": np.nan,
        "non_missing": np.nan,
        "note": "Rows removed for missing pl_name.",
    })

    # Per-column missing stats on the cleaned frame (the frame charts use).
    for col in df_clean.columns:
        missing = int(df_clean[col].isna().sum())
        total = len(df_clean)
        pct = (missing / total * 100.0) if total else 0.0
        rows.append({
            "column": col,
            "missing_in_raw": missing,
            "missing_pct_raw": round(pct, 2),
            "non_missing": total - missing,
            "note": _column_note(col),
        })

    # Also report raw-column missingness for columns that existed in the raw frame.
    for col in df_raw.columns:
        if col not in df_clean.columns:
            missing = int(df_raw[col].isna().sum())
            total = len(df_raw)
            pct = (missing / total * 100.0) if total else 0.0
            rows.append({
                "column": col,
                "missing_in_raw": missing,
                "missing_pct_raw": round(pct, 2),
                "non_missing": total - missing,
                "note": _column_note(col) + " (raw-only; not retained).",
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the expected columns exist; missing ones become all-NaN."""
    expected = [
        "pl_name", "hostname", "disc_year", "discoverymethod",
        "pl_rade", "pl_bmasse", "pl_orbper", "sy_dist", "pl_eqt",
        "disc_facility", "disc_telescope", "st_teff", "st_rad",
        "pl_orbsmax", "sy_pnum", "disc_pubdate",
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    return df


def _drop_rows_missing_name(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no planet name (no identity -> cannot be displayed)."""
    before = len(df)
    out = df.dropna(subset=["pl_name"]).copy()
    out["pl_name"] = out["pl_name"].astype(str).str.strip()
    out = out[out["pl_name"].str.lower() != "nan"]
    after = len(out)
    logger_note = f"Dropped {before - after} rows missing pl_name ({before} -> {after})."
    # Stored for the report rather than printed.
    return out


def _convert_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Convert sy_dist (parsecs) -> light-years; keep both columns."""
    if "sy_dist" in df.columns:
        df["sy_dist_ly"] = df["sy_dist"] * PC_TO_LY
    else:
        df["sy_dist_ly"] = np.nan
    return df


def _derive_planet_type(df: pd.DataFrame) -> pd.DataFrame:
    """Derive planet_type from pl_rade (Earth radii)."""
    bins = [-np.inf, RADIUS_EARTH_LIKE_MAX, RADIUS_SUPER_EARTH_MAX,
            RADIUS_NEPTUNE_MAX, np.inf]
    labels = [
        "Earth-like / Rocky",
        "Super-Earth",
        "Neptune-like",
        "Jupiter-like / Gas Giant",
    ]
    if "pl_rade" in df.columns:
        df["planet_type"] = pd.cut(
            df["pl_rade"],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )
    else:
        df["planet_type"] = pd.Series([np.nan] * len(df), dtype=object)
    return df


def _derive_temp_class(df: pd.DataFrame) -> pd.DataFrame:
    """Derive temp_class from pl_eqt (equilibrium temperature, K)."""
    bins = [-np.inf, TEMP_COLD_MAX, TEMP_TEMPERATE_MAX, TEMP_WARM_MAX, np.inf]
    labels = ["Cold", "Temperate", "Warm", "Hot"]
    if "pl_eqt" in df.columns:
        df["temp_class"] = pd.cut(
            df["pl_eqt"],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )
    else:
        df["temp_class"] = pd.Series([np.nan] * len(df), dtype=object)
    return df


def _derive_mission(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive `mission` from `disc_facility`, cleaned and grouped.

    Leading/trailing whitespace is stripped, empty strings become NaN, then
    the top MISSION_TOP_N facilities are kept explicit and the rest are
    grouped as "Other".
    """
    if "disc_facility" not in df.columns:
        df["mission"] = "Unknown"
        return df

    s = df["disc_facility"].astype(str).str.strip()
    s = s.replace({"nan": np.nan, "NaN": np.nan, "": np.nan})
    # Count non-null frequencies.
    counts = s.dropna().value_counts()
    top = set(counts.head(MISSION_TOP_N).index.tolist())
    df["mission"] = s.where(s.isin(top), other="Other")
    df["mission"] = df["mission"].fillna("Unknown")
    return df


def _standardize_year(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce disc_year to numeric (NaN where unparseable)."""
    if "disc_year" in df.columns:
        df["disc_year"] = pd.to_numeric(df["disc_year"], errors="coerce")
    else:
        df["disc_year"] = np.nan
    return df


def _column_note(col: str) -> str:
    """Short human-readable note about a column's role."""
    notes = {
        "pl_name": "Planet name (identity). Rows missing this are dropped globally.",
        "hostname": "Host star name.",
        "disc_year": "Discovery year (numeric). Used for trend/forecast charts.",
        "discoverymethod": "Discovery method (e.g. Transit, Radial Velocity).",
        "pl_rade": "Planet radius in Earth radii. Drives planet_type.",
        "pl_bmasse": "Planet mass in Earth masses.",
        "pl_orbper": "Orbital period in days.",
        "sy_dist": "Distance to system in parsecs.",
        "sy_dist_ly": "Distance to system in light-years (derived from sy_dist).",
        "pl_eqt": "Equilibrium temperature in K. Drives temp_class.",
        "disc_facility": "Discovery facility (raw). Basis for `mission` grouping.",
        "disc_telescope": "Discovery telescope (raw).",
        "st_teff": "Host star effective temperature in K.",
        "st_rad": "Host star radius (solar radii).",
        "pl_orbsmax": "Semi-major axis in AU.",
        "sy_pnum": "Number of planets in the system.",
        "disc_pubdate": "Discovery publication date (archive has no rowupdate column; used as last-updated proxy).",
        "planet_type": "Derived from pl_rade (radius classification).",
        "temp_class": "Derived from pl_eqt (temperature classification).",
        "mission": "Derived from disc_facility (top-N + Other).",
    }
    return notes.get(col, "Exoplanet parameter.")


# ---------------------------------------------------------------------------
# Log-transform helpers (used by chart modules)
# ---------------------------------------------------------------------------

def log10_safe(series: pd.Series, floor: float = 1e-6) -> pd.Series:
    """
    Compute log10 of a numeric series, safely handling non-positive / NaN.

    Non-positive finite values are clamped to `floor` before the log so the
    chart can show them on a log axis without -inf. NaNs are preserved.
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.copy()
    positive_mask = s > 0
    s.loc[positive_mask == False] = np.nan  # replace non-positive with NaN
    return np.log10(s)


# ---------------------------------------------------------------------------
# Logging-friendly helpers for the report (no global print)
# ---------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)
