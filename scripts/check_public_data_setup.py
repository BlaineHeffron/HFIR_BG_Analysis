#!/usr/bin/env python3
"""Check the public spectrum-directory and canonical-database configuration.

This intentionally uses only the Python standard library, so it can verify the
download before the optional ROOT/PyROOT analysis dependency is installed.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path, PureWindowsPath


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    return 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sanitize-database-path",
        action="store_true",
        help=(
            "replace directory paths stored in the canonical database with the "
            "bundle-relative spectra directory"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_value = os.environ.get("HFIRBGDATA")
    db_value = os.environ.get("HFIRBG_CALDB")
    if not data_value:
        return fail("HFIRBGDATA is not set")
    if not db_value:
        candidates = [
            Path(__file__).resolve().parents[1] / "db" / "HFIRBG.db",
            Path(data_value).expanduser().resolve().parent / "HFIRBG.db",
        ]
        db_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if db_path is None:
            return fail("HFIRBG_CALDB is not set and no bundled/default HFIRBG.db was found")
        db_value = str(db_path)

    data_dir = Path(data_value).expanduser().resolve()
    db_path = Path(db_value).expanduser().resolve()
    if not data_dir.is_dir():
        return fail(f"spectrum directory does not exist: {data_dir}")
    if not db_path.is_file():
        return fail(f"SQLite database does not exist: {db_path}")

    txt_names = {path.stem for path in data_dir.glob("*.txt")}

    try:
        mode = "rw" if args.sanitize_database_path else "ro"
        connection = sqlite3.connect(f"file:{db_path}?mode={mode}", uri=True)
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            return fail(f"database integrity check failed: {integrity}")
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        required = {"datafile", "directory", "calibration_group", "file_calibration_group"}
        missing_tables = required - tables
        if missing_tables:
            return fail("database is missing tables: " + ", ".join(sorted(missing_tables)))

        stored_directories = [row[0] for row in connection.execute("SELECT path FROM directory")]
        db_names = [row[0] for row in connection.execute("SELECT name FROM datafile")]
        missing_files = [name for name in db_names if name not in txt_names]
        if args.sanitize_database_path:
            if missing_files:
                return fail(
                    "refusing to change database paths while file records are missing "
                    "from HFIRBGDATA"
                )
            relative_data_dir = os.path.relpath(data_dir, db_path.parent)
            if relative_data_dir == os.pardir or relative_data_dir.startswith(os.pardir + os.sep):
                return fail(
                    "refusing to store a path outside the database bundle; "
                    "place the database beside the spectra directory"
                )
            connection.execute("UPDATE directory SET path = ?", (relative_data_dir,))
            connection.commit()
            stored_directories = [relative_data_dir]

        calibration_links = connection.execute(
            "SELECT COUNT(DISTINCT file_id) FROM file_calibration_group"
        ).fetchone()[0]
    except sqlite3.Error as error:
        return fail(f"could not read database: {error}")
    finally:
        if "connection" in locals():
            connection.close()

    print(f"HFIRBGDATA:   {data_dir}")
    print(f"HFIRBG_CALDB: {db_path}")
    print(f"Spectrum files found: {len(txt_names)}")
    print(f"Database file records: {len(db_names)}")
    print(f"Files with calibration assignments: {calibration_links}")
    print("Stored spectrum directories: " + ", ".join(stored_directories))
    absolute_directories = [
        path
        for path in stored_directories
        if Path(path).is_absolute() or PureWindowsPath(path).is_absolute()
    ]
    if absolute_directories:
        print(
            "WARNING: database contains machine-specific absolute paths; "
            "HFIRBGDATA overrides them. Run again with --sanitize-database-path "
            "to make the database itself portable."
        )
    if missing_files:
        print(f"WARNING: {len(missing_files)} database records have no matching .txt file")
        print("First missing names: " + ", ".join(missing_files[:10]))
        return 2

    print("Setup looks good: every database file record has a matching spectrum.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
