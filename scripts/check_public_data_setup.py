#!/usr/bin/env python3
"""Check the public spectrum-directory and canonical-database configuration.

This intentionally uses only the Python standard library, so it can verify the
download before the optional ROOT/PyROOT analysis dependency is installed.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    return 1


def main():
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

    try:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        required = {"datafile", "directory", "calibration_group", "file_calibration_group"}
        missing_tables = required - tables
        if missing_tables:
            return fail("database is missing tables: " + ", ".join(sorted(missing_tables)))

        db_names = [row[0] for row in connection.execute("SELECT name FROM datafile")]
        calibration_links = connection.execute(
            "SELECT COUNT(DISTINCT file_id) FROM file_calibration_group"
        ).fetchone()[0]
    except sqlite3.Error as error:
        return fail(f"could not read database: {error}")
    finally:
        if "connection" in locals():
            connection.close()

    txt_names = {path.stem for path in data_dir.glob("*.txt")}
    missing_files = [name for name in db_names if name not in txt_names]

    print(f"HFIRBGDATA:   {data_dir}")
    print(f"HFIRBG_CALDB: {db_path}")
    print(f"Spectrum files found: {len(txt_names)}")
    print(f"Database file records: {len(db_names)}")
    print(f"Files with calibration assignments: {calibration_links}")
    if missing_files:
        print(f"WARNING: {len(missing_files)} database records have no matching .txt file")
        print("First missing names: " + ", ".join(missing_files[:10]))
        return 2

    print("Setup looks good: every database file record has a matching spectrum.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
