"""Focused regressions for the bounded HPGe cosmic-pilot comparison."""

from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np

from src.public_data.cosmic_pilot import (
    area_convergence,
    binomial_clopper_pearson,
    check_sato_runtime,
    combine_identical_spectra,
    fixed_edges,
    load_simulation_batch,
    select_measured_spectra,
    semantic_replay_equal,
    shape_distances,
    sideband_diagnostic,
    unit_area_histogram,
    values_outside_masks,
)


EVT_DTYPE = np.dtype(
    [("evt", "<i8"), ("t", "<f8"), ("ct", "<f8"), ("flg", "<i4")]
)
PRIM_DTYPE = np.dtype(
    [
        ("PID", "<i4"),
        ("x", "<f4", (3,)),
        ("p", "<f4", (3,)),
        ("E", "<f4"),
        ("t", "<f4"),
        ("evt", "<i8"),
        ("vol", "<i4"),
    ]
)
IONI_DTYPE = np.dtype(
    [
        ("E", "<f4"),
        ("t", "<f4"),
        ("x", "<f4", (3,)),
        ("EdEdx", "<f4"),
        ("Eq", "<f4"),
        ("vol", "<i4"),
        ("PID", "<i4"),
        ("evt", "<i8"),
    ]
)


class CosmicPilotTests(unittest.TestCase):
    def make_hdf(
        self,
        path: Path,
        *,
        foreign_ionization: bool = False,
        compression: str | None = None,
        cpu_time_offset: float = 0.0,
    ) -> None:
        event_ids = np.asarray([42_100_000, 42_100_001, 42_100_002])
        evt = np.zeros(3, dtype=EVT_DTYPE)
        evt["evt"] = event_ids
        evt["t"] = [1.0e9, 2.0e9, 3.0e9]
        evt["ct"] = [0.01 + cpu_time_offset, 0.02, 0.03]
        prim = np.zeros(3, dtype=PRIM_DTYPE)
        prim["evt"] = event_ids
        prim["PID"] = 2112
        prim["E"] = [1.0, 2.0, 3.0]
        ioni = np.zeros(3, dtype=IONI_DTYPE)
        ioni["evt"] = [event_ids[0], event_ids[0], event_ids[2]]
        if foreign_ionization:
            ioni["evt"][-1] = 9
        ioni["E"] = [1.0, 0.5, 12.0]
        ioni["PID"] = [11, 22, 11]
        with h5py.File(path, "w") as handle:
            for name, values in (("evt", evt), ("prim", prim), ("ioni", ioni)):
                dataset = handle.create_dataset(name, data=values, compression=compression)
                dataset.attrs["nprim"] = 3.0
                dataset.attrs["runtime"] = 3.0

    def make_bundle(self, root: Path) -> tuple[dict, Path]:
        spectra = root / "spectra"
        spectra.mkdir()
        (spectra / "00000001.txt").write_text(
            "1 0 2\n2 1 3\n3 2 5\n4 3 7\n", encoding="utf-8"
        )
        db = root / "HFIRBG.db"
        with sqlite3.connect(db) as connection:
            connection.executescript(
                """
                CREATE TABLE runs (id INTEGER PRIMARY KEY, name TEXT, description TEXT,
                    detector_configuration INTEGER, detector_coordinates INTEGER);
                CREATE TABLE detector_configuration (id INTEGER PRIMARY KEY, detector INTEGER,
                    detector_settings INTEGER, acquisition_settings INTEGER, shield INTEGER);
                CREATE TABLE acquisition_settings (id INTEGER PRIMARY KEY, coarse_gain REAL,
                    PUR_guard REAL, offset REAL, fine_gain REAL, LLD REAL, LTC_mode INTEGER,
                    memory_group INTEGER);
                CREATE TABLE detector_coordinates (id INTEGER PRIMARY KEY, Rx REAL, Rz REAL,
                    Lx REAL, Lz REAL, angle REAL, track INTEGER);
                CREATE TABLE shield_configuration (id INTEGER PRIMARY KEY, name TEXT,
                    description TEXT);
                CREATE TABLE detector (id INTEGER PRIMARY KEY, type TEXT, description TEXT);
                CREATE TABLE directory (id INTEGER PRIMARY KEY, path TEXT);
                CREATE TABLE datafile (id INTEGER PRIMARY KEY, name TEXT, directory_id INTEGER,
                    start_time INTEGER, live_time REAL, run_number INTEGER);
                CREATE TABLE run_file_list (run_id INTEGER, file_id INTEGER);
                CREATE TABLE calibration_group (id INTEGER PRIMARY KEY, name TEXT, A0 REAL, A1 REAL);
                CREATE TABLE file_calibration_group (file_id INTEGER, group_id INTEGER, det INTEGER);
                INSERT INTO detector VALUES (1, 'Germanium', 'test');
                INSERT INTO shield_configuration VALUES (2, 'RD', 'test shield');
                INSERT INTO detector_coordinates VALUES (3, 33.5, 200, 33.5, 200, 180, 0);
                INSERT INTO detector_configuration VALUES (16, 1, 1, 5, 2);
                INSERT INTO acquisition_settings VALUES (5, 2.0, 1.1, 0.0, 1.02, 0.1, 1, 1);
                INSERT INTO runs VALUES (326, 'PreCycle494_RD_low_gain', 'off', 16, 3);
                INSERT INTO directory VALUES (1, 'spectra');
                INSERT INTO datafile VALUES (10, '00000001', 1, 1000, 20.0, 1);
                INSERT INTO run_file_list VALUES (326, 10);
                INSERT INTO calibration_group VALUES (9, 'test', 0.0, 1.0);
                INSERT INTO file_calibration_group VALUES (10, 9, 1);
                """
            )
        digest = hashlib.sha256(db.read_bytes()).hexdigest()
        config = {
            "public_release": {"database_sha256": digest},
            "measured_selection": {
                "run_name": "PreCycle494_RD_low_gain",
                "file_names_chronological": ["00000001"],
                "detector_configuration_id": 16,
                "acquisition_settings_id": 5,
                "acquisition_settings": {
                    "coarse_gain": 2.0,
                    "PUR_guard": 1.1,
                    "offset": 0.0,
                    "fine_gain": 1.02,
                    "LLD": 0.1,
                    "LTC_mode": 1,
                    "memory_group": 1,
                },
                "shield_id": 2,
                "shield_name": "RD",
                "orientation_angle_deg": 180.0,
                "coordinates": {"Rx": 33.5, "Rz": 200.0, "Lx": 33.5, "Lz": 200.0},
                "calibration_A0_keV": 0.0,
                "calibration_A1_keV_per_channel": 1.0,
            },
        }
        return config, db

    def test_hdf_event_closure_overflow_and_semantic_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_path = root / "first.h5"
            second_path = root / "second.h5"
            changed_path = root / "changed.h5"
            self.make_hdf(first_path)
            self.make_hdf(second_path, compression="gzip", cpu_time_offset=0.2)
            self.make_hdf(changed_path)
            with h5py.File(changed_path, "r+") as handle:
                changed = handle["prim"][()]
                changed["E"][0] += 0.25
                handle["prim"][...] = changed
            first = load_simulation_batch(first_path, "sato_nonthermal", [2112])
            second = load_simulation_batch(second_path, "sato_nonthermal", [2112])
            changed = load_simulation_batch(changed_path, "sato_nonthermal", [2112])
            np.testing.assert_allclose(first.deposition_keV, [1500.0, 0.0, 12000.0])
            self.assertEqual(first.deposited_event_count, 2)
            self.assertTrue(semantic_replay_equal(first, second))
            self.assertFalse(semantic_replay_equal(first, changed))

            xml = root / "first.h5.xml"
            xml.write_text(
                '<AnalysisStep><CosmicNeutron flux="2 Hz/cm2" scale_T="0"/>'
                '<CosineThrower s_area="3 cm2" nAttempts="18"/></AnalysisStep>\n',
                encoding="utf-8",
            )
            closure = check_sato_runtime(first, xml)
            self.assertEqual(closure["status"], "pass")
            self.assertEqual(closure["xml_runtime_s"], 3.0)

    def test_hdf_foreign_event_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.h5"
            self.make_hdf(path, foreign_ionization=True)
            with self.assertRaisesRegex(ValueError, "foreign event"):
                load_simulation_batch(path, "sato_nonthermal", [2112])

    def test_public_selection_is_exact_and_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, db = self.make_bundle(root)
            before = db.read_bytes()
            selected = select_measured_spectra(config, root)
            combined = combine_identical_spectra(selected, root / "spectra")
            self.assertEqual(combined.file_ids, (10,))
            self.assertEqual(combined.live_time_s, 20.0)
            np.testing.assert_array_equal(combined.counts, [2, 3, 5, 7])
            shifted = replace(selected[0], energy_keV=selected[0].energy_keV + 0.1)
            with self.assertRaisesRegex(ValueError, "identical calibrated grid"):
                combine_identical_spectra((selected[0], shifted), root / "spectra")
            self.assertEqual(db.read_bytes(), before)
            self.assertFalse(Path(f"{db}-journal").exists())
            self.assertFalse(Path(f"{db}-wal").exists())

    def test_frozen_shape_masks_diagnostics_and_convergence(self):
        edges = fixed_edges(3000.0, 5600.0, 100.0)
        masks = [{"center": 5433.1, "half_width": 20.0}]
        first, raw = unit_area_histogram(
            np.asarray([3050.0, 3150.0, 5433.1]), edges, masks
        )
        np.testing.assert_array_equal(
            values_outside_masks(np.asarray([3050.0, 5433.1]), masks),
            [3050.0],
        )
        second, _ = unit_area_histogram(np.asarray([3050.0, 3050.0]), edges)
        self.assertEqual(raw, 2.0)
        self.assertAlmostEqual(first.sum(), 1.0)
        self.assertAlmostEqual(shape_distances(first, second)["total_variation"], 0.5)
        local = sideband_diagnostic(
            np.asarray([100.0, 100.0, 150.0, 50.0]), 100.0, 30.0, [40.0, 100.0]
        )
        self.assertEqual(local["signal"], 2.0)
        self.assertEqual(local["sidebands"], 2.0)
        settings = {
            "minimum_in_window_depositions_per_area": 2,
            "bootstrap_seed": 7,
            "bootstrap_replicates": 10,
            "two_sided_ks_p_threshold": 0.05,
        }
        result = area_convergence(
            np.asarray([1.0, 2.0, 3.0, 4.0]),
            np.asarray([1.0, 2.0, 3.0, 4.0]),
            settings,
        )
        self.assertEqual(result["status"], "not_rejected")
        self.assertEqual(result["ks_D"], 0.0)
        interval = binomial_clopper_pearson(1, 100)
        self.assertLess(interval[0], 0.01)
        self.assertGreater(interval[1], 0.01)


if __name__ == "__main__":
    unittest.main()
