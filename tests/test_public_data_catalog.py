"""Focused tests for the normalized public run catalog."""

from __future__ import annotations

import os
import sqlite3
import unittest
from pathlib import Path

from src.public_data.catalog import (
    build_run_catalog,
    derive_reactor_cycle,
    derive_reactor_state,
)


class ReactorMetadataDerivationTests(unittest.TestCase):
    def test_cycle_from_name_with_letter_suffix(self):
        self.assertEqual(derive_reactor_cycle("Cycle490BRampTo100"), "490B")
        self.assertEqual(derive_reactor_cycle("Cycle496b_RD"), "496B")

    def test_cycle_falls_back_to_description(self):
        self.assertEqual(derive_reactor_cycle("HB4_overnight", "Cycle 491 HB4"), "491")

    def test_cycle_is_unknown_without_explicit_token(self):
        self.assertEqual(derive_reactor_cycle("position_scan_4"), "unknown")

    def test_explicit_states(self):
        self.assertEqual(derive_reactor_state("MIF_reactor_on"), "on")
        self.assertEqual(derive_reactor_state("PreCycle496_RD"), "off")
        self.assertEqual(derive_reactor_state("Cycle496Outage_RD"), "off")
        self.assertEqual(derive_reactor_state("Cycle491_10to30pwr"), "transition")
        self.assertEqual(derive_reactor_state("Cycle493Shutdown_RD"), "transition")

    def test_generic_cycle_and_conflicts_are_unknown(self):
        self.assertEqual(derive_reactor_state("Cycle495_RD"), "unknown")
        self.assertEqual(
            derive_reactor_state("reactor_on_and_reactor_off"),
            "unknown",
        )


@unittest.skipUnless(
    os.environ.get("HFIRBG_CALDB"),
    "HFIRBG_CALDB is not set; canonical-database integration test skipped",
)
class CanonicalDatabaseIntegrationTests(unittest.TestCase):
    def test_catalog_matches_run_and_file_aggregates(self):
        catalog = build_run_catalog()
        db_path = Path(os.environ["HFIRBG_CALDB"]).expanduser().resolve()
        with sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True) as connection:
            expected_runs = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            expected_files = connection.execute(
                "SELECT COUNT(*) FROM run_file_list"
            ).fetchone()[0]

        self.assertFalse(catalog.empty)
        self.assertTrue(catalog["run_id"].is_unique)
        self.assertEqual(len(catalog), expected_runs)
        self.assertEqual(int(catalog["file_count"].sum()), expected_files)
        self.assertTrue(
            {
                "start_time",
                "end_time",
                "total_live_time",
                "orientation_angle",
                "detector_type",
                "acquisition_coarse_gain",
                "shield_name",
            }.issubset(catalog.columns)
        )
        populated = catalog[catalog["file_count"] > 0]
        self.assertTrue((populated["end_time"] >= populated["start_time"]).all())


if __name__ == "__main__":
    unittest.main()
