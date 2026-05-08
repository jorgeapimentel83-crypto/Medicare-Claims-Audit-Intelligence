"""
Medicare Claims Audit — Data Loading
======================================
GPU-first data loading with automatic fallback to pandas.
Handles dtype optimization, memory management, and data validation.

Usage:
    from src.load_data import load_provider_service, load_leie, GPU_AVAILABLE
    df = load_provider_service(year="2022", nrows=500_000)  # dev sample
    df = load_provider_service(year="2022")                 # full dataset
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

# GPU-first: try RAPIDS cuDF, fall back to pandas
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
    _default_engine = cudf
    _gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
except ImportError:
    import pandas
    GPU_AVAILABLE = False
    _default_engine = None
    _gpu_name = None

import pandas as pd  # Always available

from config import PATHS, CMS_DTYPE_MAP, DEFAULT_YEAR, LEIE_FILENAME

log = logging.getLogger(__name__)


def _engine_info() -> str:
    """Return a one-line string describing the active data engine."""
    if GPU_AVAILABLE:
        return f"cuDF (GPU: {_gpu_name})"
    return "pandas (CPU)"


def load_csv(
    filepath: Path,
    dtype_map: Optional[dict] = None,
    nrows: Optional[int] = None,
    use_gpu: bool = True,
) -> pd.DataFrame:
    """
    Load a CSV with GPU (cuDF) or CPU (pandas).

    Parameters
    ----------
    filepath : Path
        Full path to the CSV file.
    dtype_map : dict, optional
        Column name → dtype mapping. Uses CMS_DTYPE_MAP by default.
    nrows : int, optional
        Limit rows for development iteration. None = full file.
    use_gpu : bool
        If True and RAPIDS available, load with cuDF.
        Result is always converted to pandas for downstream compatibility.

    Returns
    -------
    pd.DataFrame
        Loaded data as a pandas DataFrame.
    """
    if not filepath.exists():
        log.error(f"File not found: {filepath}")
        log.error(f"Run first: python src/download_data.py")
        raise FileNotFoundError(f"{filepath} does not exist")

    size_mb = filepath.stat().st_size / (1024 ** 2)
    engine_name = "cuDF" if (use_gpu and GPU_AVAILABLE) else "pandas"
    sample_note = f", nrows={nrows:,}" if nrows else ""
    log.info(f"Loading {filepath.name} ({size_mb:.0f} MB) via {engine_name}{sample_note}")

    dtypes = dtype_map or CMS_DTYPE_MAP

    try:
        if use_gpu and GPU_AVAILABLE:
            # cuDF load → convert to pandas at the end
            df = cudf.read_csv(filepath, dtype=dtypes, nrows=nrows)
            df = df.to_pandas()
        else:
            df = pd.read_csv(filepath, dtype=dtypes, nrows=nrows, low_memory=False)
    except Exception as e:
        log.warning(f"Typed load failed ({e}), retrying without dtype map...")
        if use_gpu and GPU_AVAILABLE:
            df = cudf.read_csv(filepath, nrows=nrows).to_pandas()
        else:
            df = pd.read_csv(filepath, nrows=nrows, low_memory=False)

    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    log.info(f"  ✓ {len(df):,} rows × {len(df.columns)} cols ({mem_mb:.1f} MB in memory)")
    return df


# ---------------------------------------------------------------------------
# Convenience loaders for each CMS dataset
# ---------------------------------------------------------------------------
def load_provider_service(
    year: str = DEFAULT_YEAR,
    nrows: Optional[int] = None,
    use_gpu: bool = True,
) -> pd.DataFrame:
    """
    Load Medicare Physician & Other Practitioners — by Provider and Service.

    This is the primary audit dataset: one row per (NPI, HCPCS, Place of Service).
    ~10M rows for a full year. Set nrows=500_000 for quick dev iteration.
    """
    filepath = PATHS["data_raw"] / f"provider_service_{year}.csv"
    return load_csv(filepath, nrows=nrows, use_gpu=use_gpu)


def load_provider_agg(
    year: str = DEFAULT_YEAR,
    nrows: Optional[int] = None,
    use_gpu: bool = True,
) -> pd.DataFrame:
    """
    Load Medicare Physician & Other Practitioners — by Provider (aggregate).
    One row per NPI. ~1.2M rows.
    """
    filepath = PATHS["data_raw"] / f"provider_agg_{year}.csv"
    return load_csv(filepath, nrows=nrows, use_gpu=use_gpu)


def load_geo_service(
    year: str = DEFAULT_YEAR,
    nrows: Optional[int] = None,
    use_gpu: bool = True,
) -> pd.DataFrame:
    """
    Load Medicare Physician & Other Practitioners — by Geography and Service.
    State/national benchmarks by HCPCS code for peer comparison features.
    """
    filepath = PATHS["data_raw"] / f"geo_service_{year}.csv"
    return load_csv(filepath, nrows=nrows, use_gpu=use_gpu)


def load_leie(use_gpu: bool = False) -> pd.DataFrame:
    """
    Load OIG List of Excluded Individuals/Entities.

    Used for weak supervision: providers on LEIE have been excluded from
    federal healthcare programs based on OIG exclusion records.
    Join on NPI to create pseudo-labels for model training.

    Note: LEIE is small (~75K rows), so GPU loading isn't necessary.
    """
    filepath = PATHS["data_raw"] / LEIE_FILENAME
    return load_csv(filepath, dtype_map={}, nrows=None, use_gpu=use_gpu)


# ---------------------------------------------------------------------------
# Data quality checks
# ---------------------------------------------------------------------------
def quality_report(df: pd.DataFrame, name: str = "Dataset") -> dict:
    """
    Generate a concise data quality summary for a CMS dataset.

    Returns a dict of quality metrics for programmatic use.
    """
    report = {
        "name": name,
        "rows": len(df),
        "cols": len(df.columns),
        "memory_mb": df.memory_usage(deep=True).sum() / (1024 ** 2),
        "missing": {},
        "numeric_ranges": {},
    }

    print(f"\n{'='*60}")
    print(f"  DATA QUALITY: {name}")
    print(f"{'='*60}")
    print(f"  Shape: {report['rows']:,} × {report['cols']}")
    print(f"  Memory: {report['memory_mb']:.1f} MB")

    # Missing values
    missing = df.isnull().sum()
    has_missing = missing[missing > 0].sort_values(ascending=False)
    if len(has_missing) > 0:
        print(f"\n  Missing values ({len(has_missing)} cols):")
        for col in has_missing.head(8).index:
            pct = has_missing[col] / len(df) * 100
            print(f"    {col}: {has_missing[col]:,} ({pct:.1f}%)")
            report["missing"][col] = {"count": int(has_missing[col]), "pct": round(pct, 1)}
    else:
        print(f"\n  Missing values: None ✓")

    # Key numeric columns
    numeric = df.select_dtypes(include=[np.number])
    if len(numeric.columns) > 0:
        print(f"\n  Numeric ranges:")
        for col in numeric.columns[:6]:
            lo, hi = numeric[col].min(), numeric[col].max()
            med = numeric[col].median()
            print(f"    {col}: [{lo:,.2f} → {hi:,.2f}] median={med:,.2f}")
            report["numeric_ranges"][col] = {
                "min": float(lo), "max": float(hi), "median": float(med)
            }

    return report


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Data engine: {_engine_info()}")
    print(f"GPU available: {GPU_AVAILABLE}")
    print(f"Raw data dir: {PATHS['data_raw']}")
    print(f"\nFiles in data/raw/:")
    for f in sorted(PATHS["data_raw"].glob("*.csv")):
        print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
