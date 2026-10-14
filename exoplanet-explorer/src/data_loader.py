"""
Data loader for the Exoplanet Explorer Streamlit dashboard.

Fetches confirmed-planet data from the NASA Exoplanet Archive TAP API
(table `pscomppars`) with a timeout and retry, and falls back to a locally
saved CSV snapshot (`data/snapshot.csv`) when the API is unreachable.

Caching is handled at the Streamlit layer (`st.cache_data`) in `app.py`;
this module exposes the raw fetch logic plus a small wrapper that Streamlit
caches.
"""

from __future__ import annotations

import io
import logging
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
SNAPSHOT_PATH = "data/snapshot.csv"

# Columns requested from the NASA Exoplanet Archive `pscomppars` table.
# These are URL-encoded when building the query.
_COLUMNS = [
    "pl_name",
    "hostname",
    "disc_year",
    "discoverymethod",
    "pl_rade",
    "pl_bmasse",
    "pl_orbper",
    "sy_dist",
    "pl_eqt",
    "disc_facility",
    "disc_telescope",
    "st_teff",
    "st_rad",
    "pl_orbsmax",
    "sy_pnum",
    "disc_pubdate",
]

# Retry configuration for the live API fetch.
FETCH_TIMEOUT_SECONDS = 40
FETCH_MAX_RETRIES = 3
FETCH_BACKOFF_SECONDS = 5

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def _build_query(columns: list[str] | None = None) -> str:
    """Build the TAP SQL query string (without URL encoding)."""
    cols = columns or _COLUMNS
    select_clause = "select " + ", ".join(cols) + " from pscomppars"
    return select_clause


def _urlencode_query(query: str) -> str:
    """URL-encode a TAP query string for use in a query parameter."""
    from urllib.parse import quote_plus

    return quote_plus(query)


# ---------------------------------------------------------------------------
# Core fetch logic (no Streamlit coupling)
# ---------------------------------------------------------------------------

def fetch_exoplanet_data(
    columns: list[str] | None = None,
    snapshot_path: str = SNAPSHOT_PATH,
    timeout: float = FETCH_TIMEOUT_SECONDS,
    max_retries: int = FETCH_MAX_RETRIES,
    backoff: float = FETCH_BACKOFF_SECONDS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Fetch exoplanet data from the NASA Exoplanet Archive.

    Returns a tuple of (dataframe, metadata), where `metadata` describes the
    fetch outcome:

    - ``source``: ``"live"`` if the API succeeded, ``"snapshot"`` if the
      fallback CSV was used.
    - ``last_fetched``: ISO timestamp of the fetch (UTC).
    - ``fetch_error``: error message string if the live fetch failed, else
      ``None``.
    - ``row_count``: number of rows loaded.
    - ``count_total``: total confirmed-planet count from the archive
      (only meaningful when source is ``"live"``; fallback returns the
      snapshot's own row count as an approximation).

    If the live API is unreachable, the function tries to load
    `snapshot_path`. If that also fails, an empty DataFrame is returned with
    an appropriate error in metadata.
    """
    metadata: dict[str, Any] = {
        "source": "offline",
        "last_fetched": datetime.now(timezone.utc).isoformat(),
        "fetch_error": None,
        "row_count": 0,
        "count_total": 0,
    }

    df, meta = _fetch_live(columns=columns, timeout=timeout,
                            max_retries=max_retries, backoff=backoff)
    if df is not None:
        metadata.update(meta)
        return df, metadata

    # Live fetch failed — fall back to snapshot.
    logger.warning("Live fetch failed; falling back to snapshot: %s",
                   meta.get("fetch_error"))
    metadata["source"] = "snapshot"
    metadata["fetch_error"] = meta.get("fetch_error")
    df_snap, meta_snap = _load_snapshot(snapshot_path=snapshot_path)
    metadata.update(meta_snap)
    return df_snap, metadata


def _fetch_live(
    columns: list[str] | None = None,
    timeout: float = FETCH_TIMEOUT_SECONDS,
    max_retries: int = FETCH_MAX_RETRIES,
    backoff: float = FETCH_BACKOFF_SECONDS,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """
    Attempt to fetch data from the live NASA Exoplanet Archive TAP API.

    Returns ``(None, error_meta)`` on failure so the caller can fall back
    to the snapshot.
    """
    query = _build_query(columns=columns)
    encoded_query = _urlencode_query(query)
    url = f"{BASE_URL}?query={encoded_query}&format=csv"

    last_error: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            text = resp.text
            if not text or not text.strip():
                raise ValueError("Empty response from NASA Exoplanet Archive")
            df = pd.read_csv(io.StringIO(text))
            count_total = _fetch_total_count(timeout=timeout,
                                             max_retries=max_retries,
                                             backoff=backoff)
            metadata = {
                "source": "live",
                "last_fetched": datetime.now(timezone.utc).isoformat(),
                "fetch_error": None,
                "row_count": len(df),
                "count_total": count_total,
            }
            return df, metadata
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Live fetch attempt %d/%d failed: %s",
                attempt,
                max_retries,
                last_error,
            )
            if attempt < max_retries:
                time.sleep(backoff)
    return None, {
        "source": "offline",
        "last_fetched": datetime.now(timezone.utc).isoformat(),
        "fetch_error": last_error,
        "row_count": 0,
        "count_total": 0,
    }


def _fetch_total_count(
    timeout: float = FETCH_TIMEOUT_SECONDS,
    max_retries: int = FETCH_MAX_RETRIES,
    backoff: float = FETCH_BACKOFF_SECONDS,
) -> int:
    """
    Query ``select count(*) from pscomppars`` to get the current total.

    Returns the integer count, or 0 if the query fails.
    """
    query = "select count(*) as cnt from pscomppars"
    encoded_query = _urlencode_query(query)
    url = f"{BASE_URL}?query={encoded_query}&format=csv"
    last_error: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            if "cnt" in df.columns and len(df) > 0:
                return int(df["cnt"].iloc[0])
            return 0
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Total-count query attempt %d/%d failed: %s",
                attempt,
                max_retries,
                last_error,
            )
            if attempt < max_retries:
                time.sleep(backoff)
    logger.warning("Total-count query failed; returning 0. Error: %s",
                   last_error)
    return 0


def _load_snapshot(snapshot_path: str = SNAPSHOT_PATH) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the offline snapshot CSV, or return an empty DataFrame on failure."""
    metadata: dict[str, Any] = {
        "source": "snapshot",
        "last_fetched": datetime.now(timezone.utc).isoformat(),
        "fetch_error": None,
        "row_count": 0,
        "count_total": 0,
    }
    try:
        df = pd.read_csv(snapshot_path)
        metadata["row_count"] = len(df)
        metadata["count_total"] = len(df)
        metadata["fetch_error"] = None
        return df, metadata
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error("Failed to load snapshot '%s': %s", snapshot_path, error_msg)
        metadata["fetch_error"] = error_msg
        return pd.DataFrame(), metadata


# ---------------------------------------------------------------------------
# Snapshot maintenance
# ---------------------------------------------------------------------------

def refresh_snapshot(df: pd.DataFrame, snapshot_path: str = SNAPSHOT_PATH) -> None:
    """
    Persist a freshly fetched DataFrame to the snapshot CSV.

    This keeps the offline fallback up to date after a successful live fetch.
    """
    df.to_csv(snapshot_path, index=False)
    logger.info("Snapshot updated: %s (%d rows)", snapshot_path, len(df))
