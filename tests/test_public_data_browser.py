"""Focused tests for the ROOT-free public spectrum browser data layer."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.public_data.browser import (
    load_spectrum,
    query_file_metadata,
    rebin_by_factor,
    resolve_browser_paths,
    spectrum_dataframe,
)


SCHEMA = """
CREATE TABLE directory (id INTEGER PRIMARY KEY, path TEXT NOT NULL);
CREATE TABLE datafile (
    id INTEGER PRIMARY KEY, name TEXT, directory_id INTEGER NOT NULL,
    start_time INTEGER, live_time REAL, run_number INTEGER,
    FOREIGN KEY(directory_id) REFERENCES directory(id)
);
CREATE TABLE detector_coordinates (
    id INTEGER PRIMARY KEY, Rx REAL, Rz REAL, Lx REAL, Lz REAL,
    angle REAL, track INTEGER
);
CREATE TABLE detector (id INTEGER PRIMARY KEY, type TEXT, description TEXT);
CREATE TABLE shield_configuration (
    id INTEGER PRIMARY KEY, name TEXT, description TEXT
);
CREATE TABLE detector_configuration (
    id INTEGER PRIMARY KEY, detector INTEGER, detector_settings INTEGER,
    acquisition_settings INTEGER, shield INTEGER,
    FOREIGN KEY(detector) REFERENCES detector(id),
    FOREIGN KEY(shield) REFERENCES shield_configuration(id)
);
CREATE TABLE runs (
    id INTEGER PRIMARY KEY, description TEXT, name TEXT,
    detector_configuration INTEGER, detector_coordinates INTEGER,
    FOREIGN KEY(detector_configuration) REFERENCES detector_configuration(id),
    FOREIGN KEY(detector_coordinates) REFERENCES detector_coordinates(id)
);
CREATE TABLE run_file_list (
    id INTEGER PRIMARY KEY, file_id INTEGER, run_id INTEGER,
    FOREIGN KEY(file_id) REFERENCES datafile(id),
    FOREIGN KEY(run_id) REFERENCES runs(id)
);
CREATE TABLE calibration_group (
    id INTEGER PRIMARY KEY, name TEXT, A0 REAL, A1 REAL
);
CREATE TABLE file_calibration_group (
    id INTEGER PRIMARY KEY, group_id INTEGER, det INTEGER, file_id INTEGER,
    FOREIGN KEY(group_id) REFERENCES calibration_group(id),
    FOREIGN KEY(det) REFERENCES detector(id),
    FOREIGN KEY(file_id) REFERENCES datafile(id)
);
"""


class TemporaryPublicBundleTests(unittest.TestCase):
    def setUp(self):
        # A developer may have sourced the repository .env before running the
        # suite. Keep temporary fixtures independent of that canonical bundle.
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.temporary = tempfile.TemporaryDirectory()
        self.bundle = Path(self.temporary.name)
        self.spectra = self.bundle / "spectra"
        self.spectra.mkdir()
        self.db = self.bundle / "HFIRBG.db"
        with sqlite3.connect(self.db) as connection:
            connection.executescript(SCHEMA)
            connection.executescript(
                """
                INSERT INTO directory VALUES (1, 'spectra');
                INSERT INTO datafile VALUES (7, '00000007', 1, 1000, 10.0, 7);
                INSERT INTO detector_coordinates VALUES
                    (3, 21.0, 94.5, 38.0, 90.5, 46.5, 0);
                INSERT INTO detector VALUES (1, 'HPGe', 'test detector');
                INSERT INTO shield_configuration VALUES (2, 'none', 'no shield');
                INSERT INTO detector_configuration VALUES (4, 1, 1, 1, 2);
                INSERT INTO runs VALUES (5, 'test run', 'Cycle490A_test', 4, 3);
                INSERT INTO run_file_list VALUES (1, 7, 5);
                INSERT INTO calibration_group VALUES (6, 'test calibration', 0.5, 2.0);
                INSERT INTO file_calibration_group VALUES (1, 6, 1, 7);
                """
            )
        (self.spectra / "00000007.txt").write_text(
            "# n\tenergy(keV)\tcounts\trate(1/s)\n"
            "1\t999.0\t0\t0\n"
            "2\t999.0\t4\t0.4\n"
            "3\t999.0\t9\t0.9\n"
            "4\t999.0\t16\t1.6\n"
            "5\t999.0\t25\t2.5\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()
        self.environment.stop()

    def test_relative_database_directory_and_canonical_calibration(self):
        spectrum = load_spectrum(7, db_path=self.db)
        self.assertEqual(spectrum.file_id, 7)
        self.assertEqual(spectrum.run_id, 5)
        self.assertEqual(spectrum.metadata["coordinate_angle"], 46.5)
        np.testing.assert_allclose(spectrum.counts, [0, 4, 9, 16, 25])
        # The deliberately incorrect energy column in the text file is ignored.
        np.testing.assert_allclose(spectrum.energy_keV, [2.5, 4.5, 6.5, 8.5, 10.5])

    def test_explicit_data_root_precedes_environment(self):
        missing = self.bundle / "missing"
        with patch.dict(os.environ, {"HFIRBGDATA": str(missing)}):
            spectrum = load_spectrum(7, db_path=self.db, data_root=self.spectra)
        self.assertEqual(spectrum.counts.sum(), 54)

    def test_environment_precedes_adjacent_standard_layout(self):
        alternate = self.bundle / "alternate"
        alternate.mkdir()
        (alternate / "00000007.txt").write_text(
            "1\t1\t1\t0.1\n", encoding="utf-8"
        )
        with patch.dict(os.environ, {"HFIRBGDATA": str(alternate)}):
            spectrum = load_spectrum(7, db_path=self.db)
        np.testing.assert_allclose(spectrum.counts, [1])

    def test_metadata_joins_and_filters(self):
        metadata = query_file_metadata(self.db, run_id=5)
        self.assertEqual(len(metadata), 1)
        row = metadata.iloc[0]
        self.assertEqual(row["file_id"], 7)
        self.assertEqual(row["calibration_A1"], 2.0)
        self.assertEqual(row["coordinate_Rx"], 21.0)
        self.assertEqual(row["coordinate_track"], 0)
        self.assertTrue(query_file_metadata(self.db, run_id=999).empty)

    def test_rebin_and_normalizations(self):
        spectrum = rebin_by_factor(load_spectrum(7, db_path=self.db), 2)
        np.testing.assert_allclose(spectrum.counts, [4, 25, 25])
        np.testing.assert_allclose(spectrum.energy_keV, [3.5, 7.5, 10.5])
        np.testing.assert_allclose(spectrum.bin_width_keV, [4, 4, 2])

        counts = spectrum_dataframe(spectrum, "counts")
        np.testing.assert_allclose(counts["statistical_error"], [2, 5, 5])
        per_second = spectrum_dataframe(spectrum, "counts/s")
        np.testing.assert_allclose(per_second["value"], [0.4, 2.5, 2.5])
        density = spectrum_dataframe(spectrum, "counts/s/keV")
        np.testing.assert_allclose(density["value"], [0.1, 0.625, 1.25])

    def test_malformed_channel_rows_are_rejected(self):
        (self.spectra / "00000007.txt").write_text(
            "1\t2.5\t1\t0.1\n3\t6.5\t2\t0.2\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "contiguous and 1-based"):
            load_spectrum(7, db_path=self.db)

    def test_missing_files_and_invalid_options_are_clear(self):
        (self.spectra / "00000007.txt").unlink()
        with self.assertRaises(FileNotFoundError):
            load_spectrum(7, db_path=self.db)
        with self.assertRaises(ValueError):
            query_file_metadata(self.db, file_id=True)

    def test_read_only_connection_cannot_mutate_database(self):
        metadata = query_file_metadata(self.db)
        self.assertEqual(len(metadata), 1)
        # The query did not create journal/WAL sidecars or alter the source.
        self.assertFalse(Path(f"{self.db}-journal").exists())
        self.assertFalse(Path(f"{self.db}-wal").exists())


@unittest.skipUnless(
    os.environ.get("HFIRBG_CALDB"),
    "HFIRBG_CALDB is not set; canonical integration test skipped",
)
class CanonicalBrowserIntegrationTests(unittest.TestCase):
    def test_load_first_canonical_spectrum(self):
        metadata = query_file_metadata()
        self.assertFalse(metadata.empty)
        db_path = Path(os.environ["HFIRBG_CALDB"]).expanduser().resolve()
        with sqlite3.connect(db_path) as connection:
            expected_files = connection.execute(
                "SELECT COUNT(*) FROM datafile"
            ).fetchone()[0]
        self.assertEqual(len(metadata), expected_files)
        self.assertTrue(metadata["file_id"].is_unique)
        first_file_id = int(metadata.iloc[0]["file_id"])
        spectrum = load_spectrum(first_file_id)
        self.assertGreater(spectrum.counts.size, 0)
        self.assertGreater(spectrum.live_time, 0)
        self.assertTrue(np.all(np.diff(spectrum.energy_keV) > 0))
        self.assertEqual(spectrum.file_id, first_file_id)


if __name__ == "__main__":
    unittest.main()
