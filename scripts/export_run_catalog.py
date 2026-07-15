#!/usr/bin/env python3
"""Export the normalized public run catalog."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.public_data.catalog import build_run_catalog


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    Path(os.environ.get("HFIRBG_ANALYSIS", REPO_ROOT / "analysis")) / "catalog"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        help="canonical SQLite database (defaults to $HFIRBG_CALDB)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="directory for run_catalog.csv and, when supported, Parquet",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    catalog = build_run_catalog(args.db)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "run_catalog.csv"
    catalog.to_csv(csv_path, index=False)
    print(f"Wrote {len(catalog)} runs to {csv_path}")

    parquet_path = args.output_dir / "run_catalog.parquet"
    try:
        catalog.to_parquet(parquet_path, index=False)
    except (ImportError, ModuleNotFoundError) as error:
        print(f"Parquet export skipped (no supported engine): {error}")
    else:
        print(f"Wrote {len(catalog)} runs to {parquet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
