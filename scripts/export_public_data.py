#!/usr/bin/env python3
"""Export public HFIR run metadata, file metadata, or calibrated spectra."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.public_data.browser import (  # noqa: E402
    load_spectrum,
    query_file_metadata,
    rebin_by_factor,
    resolve_browser_paths,
    spectrum_dataframe,
)
from src.public_data.catalog import build_run_catalog  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "exports"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", help="SQLite database (default: env/public bundle)")
    parser.add_argument("--data-root", help="spectrum directory (default: env/public bundle)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog = subparsers.add_parser("catalog", help="export one row per run")
    catalog.add_argument("--cycle", action="append", help="official cycle; repeatable")
    catalog.add_argument(
        "--calendar-state",
        action="append",
        choices=("operating", "outage", "mixed", "unknown"),
        help="official calendar classification; repeatable",
    )
    catalog.add_argument("--shield", action="append", help="shield name; repeatable")
    catalog.add_argument("--contains", help="plain-text run-name/description filter")
    catalog.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "run_catalog.csv")

    files = subparsers.add_parser("files", help="export calibrated file metadata")
    files.add_argument("--run-id", type=int, action="append", help="run ID; repeatable")
    files.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "file_catalog.csv")

    spectra = subparsers.add_parser("spectra", help="export calibrated spectrum bins")
    spectra.add_argument("--file-id", type=int, action="append", help="file ID; repeatable")
    spectra.add_argument("--run-id", type=int, action="append", help="include every file in this run; repeatable")
    spectra.add_argument(
        "--normalization",
        choices=("counts", "counts/s", "counts/s/keV"),
        default="counts/s/keV",
    )
    spectra.add_argument("--rebin", type=int, default=1, help="adjacent bins to combine")
    spectra.add_argument("--emin", type=float, help="minimum bin-center energy in keV")
    spectra.add_argument("--emax", type=float, help="maximum bin-center energy in keV")
    spectra.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "spectra.csv")
    return parser.parse_args(argv)


def write_csv(frame: pd.DataFrame, output: Path) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame):,} rows to {output}")


def filter_run_catalog(catalog: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    result = catalog
    for column, values in (
        ("calendar_cycle", args.cycle),
        ("calendar_reactor_state", args.calendar_state),
        ("shield_name", args.shield),
    ):
        if values:
            result = result[result[column].fillna("unknown").astype(str).isin(values)]
    if args.contains:
        query = args.contains.strip()
        text = (
            result["run_name"].fillna("").astype(str)
            + " "
            + result["run_description"].fillna("").astype(str)
        )
        result = result[text.str.contains(query, case=False, regex=False)]
    return result.reset_index(drop=True)


def selected_file_ids(
    metadata: pd.DataFrame,
    file_ids: list[int] | None,
    run_ids: list[int] | None,
) -> list[int]:
    selected = set(file_ids or [])
    if run_ids:
        selected.update(
            metadata.loc[metadata["run_id"].isin(run_ids), "file_id"].astype(int)
        )
    known = set(metadata["file_id"].astype(int))
    missing = sorted(selected - known)
    if missing:
        raise SystemExit(f"Unknown calibrated file ID(s): {', '.join(map(str, missing))}")
    if not selected:
        raise SystemExit("spectra requires at least one --file-id or --run-id")
    return sorted(selected)


def export_spectra(
    file_ids: list[int],
    db_path: str,
    data_root: str | None,
    normalization: str,
    rebin: int,
    emin: float | None,
    emax: float | None,
) -> pd.DataFrame:
    if rebin <= 0:
        raise SystemExit("--rebin must be a positive integer")
    if emin is not None and emax is not None and emin > emax:
        raise SystemExit("--emin must be less than or equal to --emax")

    exported: list[pd.DataFrame] = []
    for file_id in file_ids:
        spectrum = rebin_by_factor(
            load_spectrum(file_id, db_path=db_path, data_root=data_root), rebin
        )
        frame = spectrum_dataframe(spectrum, normalization)  # type: ignore[arg-type]
        if emin is not None:
            frame = frame[frame["energy_keV"] >= emin]
        if emax is not None:
            frame = frame[frame["energy_keV"] <= emax]
        frame = frame.copy()
        frame.insert(0, "file_id", spectrum.file_id)
        frame.insert(1, "file_name", spectrum.file_name)
        frame.insert(2, "run_id", spectrum.run_id)
        frame.insert(3, "run_name", spectrum.run_name)
        frame.insert(4, "live_time_s", spectrum.live_time)
        frame.insert(5, "calibration_A0_keV", spectrum.calibration_A0)
        frame.insert(6, "calibration_A1_keV_per_channel", spectrum.calibration_A1)
        exported.append(frame)
    return pd.concat(exported, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = resolve_browser_paths(db_path=args.db, data_root=args.data_root)
    db_path = str(paths.db_path)
    data_root = str(paths.data_root) if paths.data_root is not None else None

    if args.command == "catalog":
        write_csv(filter_run_catalog(build_run_catalog(db_path), args), args.output)
        return 0

    metadata = query_file_metadata(db_path)
    if args.command == "files":
        if args.run_id:
            metadata = metadata[metadata["run_id"].isin(args.run_id)]
        write_csv(metadata.reset_index(drop=True), args.output)
        return 0

    ids = selected_file_ids(metadata, args.file_id, args.run_id)
    frame = export_spectra(
        ids,
        db_path,
        data_root,
        args.normalization,
        args.rebin,
        args.emin,
        args.emax,
    )
    write_csv(frame, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
