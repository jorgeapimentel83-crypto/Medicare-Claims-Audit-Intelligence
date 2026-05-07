"""
Production-safe feature rebuild for the Medicare Claims Audit Intelligence Platform.

Rebuilds provider-service features from the raw CMS Provider-Service CSV using
pandas (CPU). The development feature parquet was generated from a 500K-row
sample; this script supports either a sized sample or the full population, and
writes to distinct output paths so the development file is never overwritten.

Usage
-----
    python scripts/build_full_features.py --sample 500000
    python scripts/build_full_features.py --sample 2000000 --year 2022
    python scripts/build_full_features.py --full --year 2022

Notes
-----
- Pure pandas pipeline. The previous GPU (cuDF) path produced a segmentation
  fault on the full file, so GPU is intentionally not used here.
- Imports the existing feature builders from src/features.py without
  modification.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Path wiring: make src/ importable so we can reuse the existing builders.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features import build_all_features, get_feature_metadata  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Columns the feature pipeline (src/features.py) reads directly from raw input.
REQUIRED_COLUMNS = [
    "Rndrng_NPI",
    "Rndrng_Prvdr_Type",
    "Rndrng_Prvdr_State_Abrvtn",
    "Rndrng_Prvdr_Ent_Cd",
    "Rndrng_Prvdr_Mdcr_Prtcptg_Ind",
    "HCPCS_Cd",
    "HCPCS_Drug_Ind",
    "Place_Of_Srvc",
    "Tot_Benes",
    "Tot_Srvcs",
    "Avg_Sbmtd_Chrg",
    "Avg_Mdcr_Alowd_Amt",
    "Avg_Mdcr_Pymt_Amt",
    "Avg_Mdcr_Stdzd_Amt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild provider-service features from raw CMS CSV (pandas, CPU).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Build features from the first N rows of the raw CSV.",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Build features from the entire raw CSV.",
    )
    parser.add_argument(
        "--year",
        type=str,
        default="2022",
        help="CMS data year (default: 2022).",
    )
    args = parser.parse_args()

    # Default behavior when neither flag is supplied: 500K-row sample.
    if not args.full and args.sample is None:
        args.sample = 500_000
    return args


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    """Resolve input CSV path and (sample- or full-specific) output paths."""
    input_path = RAW_DIR / f"provider_service_{args.year}.csv"

    if args.full:
        output_path = PROCESSED_DIR / f"features_provider_service_{args.year}_full.parquet"
        metadata_path = PROCESSED_DIR / f"feature_metadata_{args.year}_full.json"
    else:
        output_path = (
            PROCESSED_DIR
            / f"features_provider_service_{args.year}_sample_{args.sample}.parquet"
        )
        metadata_path = None

    # Defensive: never overwrite the development feature parquet.
    dev_path = PROCESSED_DIR / f"features_provider_service_{args.year}.parquet"
    if output_path.resolve() == dev_path.resolve():
        raise RuntimeError(
            f"Refusing to overwrite development feature file: {dev_path}"
        )

    return input_path, output_path, metadata_path


def verify_input(input_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw CSV not found: {input_path}\n"
            f"Expected file under {RAW_DIR}. Run src/download_data.py first."
        )


def verify_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Raw CSV is missing required columns for the feature pipeline:\n"
            f"  missing: {missing}\n"
            f"  present: {list(df.columns)}"
        )


def add_compatibility_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns the feature pipeline expects but that aren't in the raw CSV.

    The CMS Provider-Service file reports an *average* Medicare payment per
    (NPI, HCPCS, POS) row (Avg_Mdcr_Pymt_Amt) and the count of services
    (Tot_Srvcs); it does not include a row-level total payment column.
    Several provider-level aggregates downstream (HHI revenue shares, drug
    revenue share) are computed against Tot_Mdcr_Pymt_Amt, so we synthesize
    an estimated row-level total here as Avg_Mdcr_Pymt_Amt * Tot_Srvcs. The
    estimate is exact enough for share-of-revenue aggregations.
    """
    if "Tot_Mdcr_Pymt_Amt" not in df.columns:
        df = df.copy()
        df["Tot_Mdcr_Pymt_Amt"] = df["Avg_Mdcr_Pymt_Amt"] * df["Tot_Srvcs"]
    return df


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    val = float(num_bytes)
    for unit in units:
        if val < 1024 or unit == units[-1]:
            return f"{val:,.1f} {unit}"
        val /= 1024
    return f"{val:,.1f} {units[-1]}"


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    input_path, output_path, metadata_path = resolve_paths(args)

    mode_label = "FULL" if args.full else f"SAMPLE n={args.sample:,}"

    print("=" * 72)
    print("  Medicare Claims Audit — Feature Rebuild")
    print("=" * 72)
    print(f"  project root : {PROJECT_ROOT}")
    print(f"  input path   : {input_path}")
    print(f"  output path  : {output_path}")
    if metadata_path is not None:
        print(f"  metadata path: {metadata_path}")
    print(f"  mode         : {mode_label}")
    print(f"  year         : {args.year}")
    print(f"  engine       : pandas (CPU)  [GPU/cuDF disabled — known segfault]")
    print("-" * 72)

    verify_input(input_path)

    nrows = None if args.full else args.sample
    print(f"Reading CSV{'' if nrows is None else f' (first {nrows:,} rows)'}...")
    df = pd.read_csv(input_path, nrows=nrows, low_memory=False)
    verify_columns(df)

    raw_mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"  raw shape          : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  memory pre-features: {raw_mem_mb:,.1f} MB")

    df = add_compatibility_columns(df)

    print("Building features...")
    df = build_all_features(df)

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    print(f"  feature shape      : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  feat_ columns      : {len(feat_cols)}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Writing parquet → {output_path}")
    df.to_parquet(output_path, engine="pyarrow", index=False)

    out_size = output_path.stat().st_size
    print(f"  output file size   : {human_size(out_size)}")

    if metadata_path is not None:
        meta = {
            "year": args.year,
            "source_csv": str(input_path),
            "output_parquet": str(output_path),
            "mode": "full",
            "row_count": int(df.shape[0]),
            "column_count": int(df.shape[1]),
            "feature_columns": sorted(feat_cols),
            "feature_count": len(feat_cols),
            "feature_descriptions": get_feature_metadata(),
        }
        print(f"Writing metadata → {metadata_path}")
        with open(metadata_path, "w") as f:
            json.dump(meta, f, indent=2)

    print("=" * 72)
    print("  ✓ Done.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
