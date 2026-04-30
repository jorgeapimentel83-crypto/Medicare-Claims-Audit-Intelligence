"""
Medicare Claims Audit — CMS Data Download Pipeline
====================================================
Downloads public Medicare Part B datasets from data.cms.gov.

All data is de-identified Public Use File (PUF) data:
  - No PHI, no PII, no DUA required for public versions
  - Provider-level aggregates with small-cell suppression (<11 benes)

Usage:
    python src/download_data.py                        # Download all, default year
    python src/download_data.py --dataset provider_service --year 2022
    python src/download_data.py --list                 # Show available datasets
"""

import sys
import time
import logging
from pathlib import Path

import requests
from tqdm import tqdm

from config import CMS_DATASETS, LEIE_URL, LEIE_FILENAME, PATHS, DEFAULT_YEAR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Download utilities
# ---------------------------------------------------------------------------
def download_file(
    url: str,
    dest: Path,
    description: str = "",
    chunk_size: int = 8192,
    timeout: int = 600,
    retries: int = 3,
) -> Path:
    """
    Stream-download a file with progress bar, retry logic, and size validation.

    CMS data.cms.gov serves CSV when the Accept header includes text/csv.
    The /data/{year} endpoint redirects to the actual CSV file.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 1000:
        size_mb = dest.stat().st_size / (1024 ** 2)
        log.info(f"  ✓ Already exists: {dest.name} ({size_mb:.1f} MB) — skipping")
        return dest

    headers = {
        "Accept": "text/csv",
        "User-Agent": "MedicareAuditIntelligence/0.1 (research; HHS-OIG-auditor)",
    }

    for attempt in range(1, retries + 1):
        try:
            log.info(f"  ↓ Downloading: {description or dest.name} (attempt {attempt}/{retries})")
            resp = requests.get(url, headers=headers, stream=True, timeout=timeout)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))

            with open(dest, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=dest.name[:45],
                disable=(total == 0),
            ) as bar:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    bar.update(len(chunk))

            size_mb = dest.stat().st_size / (1024 ** 2)
            log.info(f"  ✓ Saved: {dest.name} ({size_mb:.1f} MB)")
            return dest

        except requests.RequestException as e:
            log.warning(f"  ✗ Attempt {attempt} failed: {e}")
            if dest.exists():
                dest.unlink()
            if attempt < retries:
                wait = 2 ** attempt
                log.info(f"    Retrying in {wait}s...")
                time.sleep(wait)

    log.error(f"  ✗ FAILED after {retries} attempts: {url}")
    log.error(f"    If the URL changed, check the dataset page at data.cms.gov")
    return None


def validate_csv(filepath: Path, min_rows: int = 100) -> bool:
    """Quick sanity check: is this a real CSV with a header and data?"""
    if filepath is None or not filepath.exists():
        return False
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline()
            if not header or "," not in header:
                log.warning(f"  ⚠ {filepath.name}: not a valid CSV (no comma in header)")
                return False
            n_rows = sum(1 for _ in f)
            if n_rows < min_rows:
                log.warning(f"  ⚠ {filepath.name}: only {n_rows} data rows (expected {min_rows}+)")
                return False
        n_cols = len(header.split(","))
        log.info(f"  ✓ Validated: {filepath.name} → {n_rows:,} rows × {n_cols} cols")
        return True
    except Exception as e:
        log.error(f"  ✗ Validation error on {filepath.name}: {e}")
        return False


# ---------------------------------------------------------------------------
# Download orchestration
# ---------------------------------------------------------------------------
def download_cms_dataset(dataset_key: str, year: str = DEFAULT_YEAR) -> Path:
    """Download a specific CMS dataset for a given year."""
    if dataset_key not in CMS_DATASETS:
        log.error(f"Unknown dataset '{dataset_key}'. Available: {list(CMS_DATASETS.keys())}")
        return None

    ds = CMS_DATASETS[dataset_key]
    if year not in ds["urls"]:
        log.error(f"Year {year} not available for '{dataset_key}'. Available: {list(ds['urls'].keys())}")
        return None

    url = ds["urls"][year]
    filename = ds["filename"].format(year=year)
    dest = PATHS["data_raw"] / filename

    log.info(f"\n{'─'*60}")
    log.info(f"  {ds['name']}")
    log.info(f"  Year: {year}")
    log.info(f"─{'─'*59}")

    result = download_file(url, dest, description=ds["name"])
    if result:
        validate_csv(result)
    return result


def download_leie() -> Path:
    """Download OIG LEIE exclusion list (weak supervision labels)."""
    dest = PATHS["data_raw"] / LEIE_FILENAME
    log.info(f"\n{'─'*60}")
    log.info(f"  OIG LEIE — List of Excluded Individuals/Entities")
    log.info(f"  Purpose: weak supervision labels for model validation")
    log.info(f"─{'─'*59}")
    result = download_file(LEIE_URL, dest, description="LEIE Exclusion List")
    if result:
        validate_csv(result, min_rows=50)
    return result


def download_all(year: str = DEFAULT_YEAR) -> dict:
    """Download all datasets needed for the audit platform."""
    log.info("=" * 60)
    log.info("  MEDICARE CLAIMS AUDIT — DATA DOWNLOAD PIPELINE")
    log.info(f"  Target year: {year}")
    log.info(f"  Output: {PATHS['data_raw']}")
    log.info("=" * 60)

    results = {}

    for key, ds in CMS_DATASETS.items():
        if year in ds["urls"]:
            results[key] = download_cms_dataset(key, year)

    results["leie"] = download_leie()

    # Summary
    log.info(f"\n{'='*60}")
    log.info("  DOWNLOAD SUMMARY")
    log.info(f"{'='*60}")
    for key, path in results.items():
        status = "✓" if path and path.exists() else "✗ FAILED"
        size = f"({path.stat().st_size / 1e6:.1f} MB)" if path and path.exists() else ""
        log.info(f"  {status}  {key} {size}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download CMS Medicare Part B public use files for audit analysis"
    )
    parser.add_argument(
        "--dataset",
        choices=list(CMS_DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    parser.add_argument(
        "--year",
        default=DEFAULT_YEAR,
        help=f"Calendar year (default: {DEFAULT_YEAR})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable CMS datasets:")
        for key, ds in CMS_DATASETS.items():
            years = ", ".join(ds["urls"].keys())
            print(f"  {key:20s}  years: [{years}]  {ds['name']}")
        print(f"\n  {'leie':20s}  LEIE exclusion list (no year filter)")
        sys.exit(0)

    if args.dataset == "all":
        download_all(args.year)
    else:
        download_cms_dataset(args.dataset, args.year)

    log.info("\n✓ Done. Next step: python -c \"from src.load_data import load_provider_service; df = load_provider_service()\"")
