"""Focused tests for the normalized public run catalog."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.public_data.catalog import (
    CALENDAR_TIMEZONE,
    build_run_catalog,
    classify_run_interval,
    derive_reactor_cycle,
    derive_reactor_state,
    load_cycle_calendar,
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


def _local_timestamp(year, month, day, hour=0, minute=0):
    return datetime(
        year, month, day, hour, minute, tzinfo=ZoneInfo(CALENDAR_TIMEZONE)
    ).timestamp()


class OfficialCycleCalendarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calendar = load_cycle_calendar()

    def test_calendar_is_complete_and_authoritative(self):
        self.assertEqual(self.calendar.iloc[0]["cycle"], "490A")
        self.assertEqual(self.calendar.iloc[-1]["cycle"], "499")
        self.assertEqual(len(self.calendar), 15)
        self.assertTrue(self.calendar["date_precision"].eq("day").all())
        self.assertTrue(self.calendar["schedule_basis"].eq("actual").all())
        self.assertTrue(self.calendar["record_status"].eq("complete").all())

    def test_operating_cycle_including_day_boundaries(self):
        result = classify_run_interval(
            _local_timestamp(2021, 3, 2),
            _local_timestamp(2021, 3, 28),
            self.calendar,
        )
        self.assertEqual(result["calendar_cycle"], "490C")
        self.assertEqual(result["calendar_reactor_state"], "operating")
        self.assertEqual(result["calendar_period"], "Cycle 490C")
        self.assertEqual(result["calendar_date_precision"], "day")

    def test_whole_interval_in_gap_is_labeled_outage(self):
        result = classify_run_interval(
            _local_timestamp(2021, 2, 24, 8),
            _local_timestamp(2021, 2, 24, 9),
            self.calendar,
        )
        self.assertEqual(result["calendar_cycle"], "unknown")
        self.assertEqual(result["calendar_reactor_state"], "outage")
        self.assertEqual(result["calendar_period"], "EOC 490A / pre-cycle 490B")

    def test_interval_crossing_cycle_and_gap_is_mixed(self):
        result = classify_run_interval(
            _local_timestamp(2021, 2, 23, 23, 59),
            _local_timestamp(2021, 2, 24, 0, 1),
            self.calendar,
        )
        self.assertEqual(result["calendar_cycle"], "unknown")
        self.assertEqual(result["calendar_reactor_state"], "mixed")
        self.assertEqual(result["calendar_period"], "mixed")

    def test_invalid_and_out_of_range_intervals_are_unknown(self):
        for start, end in (
            (None, None),
            (-1, 0),
            (20, 10),
            (_local_timestamp(2020, 1, 1), _local_timestamp(2020, 1, 2)),
        ):
            with self.subTest(start=start, end=end):
                result = classify_run_interval(start, end, self.calendar)
                self.assertEqual(result["calendar_cycle"], "unknown")
                self.assertEqual(result["calendar_reactor_state"], "unknown")
                self.assertEqual(result["calendar_period"], "unknown")

    def test_calendar_validation_rejects_overlap(self):
        invalid = self.calendar.copy()
        invalid["start_date"] = invalid["start_date"].astype(str)
        invalid["end_date"] = invalid["end_date"].astype(str)
        invalid.loc[1, "start_date"] = invalid.loc[0, "end_date"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlap.csv"
            invalid.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "overlapping"):
                load_cycle_calendar(path)


@unittest.skipUnless(
    os.environ.get("HFIRBG_CALDB"),
    "HFIRBG_CALDB is not set; canonical-database integration test skipped",
)
class CanonicalDatabaseIntegrationTests(unittest.TestCase):
    def test_cycle492_ne_east_uses_the_corrected_cart_coordinate(self):
        catalog = build_run_catalog()
        row = catalog.loc[catalog["run_name"] == "Cycle492_NE_East"]
        self.assertEqual(len(row), 1)
        record = row.iloc[0]
        self.assertEqual(record["coordinate_Rx"], 32.0)
        self.assertEqual(record["coordinate_Rz"], 221.5)
        self.assertEqual(record["coordinate_Lx"], 32.0)
        self.assertEqual(record["coordinate_Lz"], 204.0)
        self.assertEqual(record["orientation_angle"], 90.0)

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
                "calendar_cycle",
                "calendar_reactor_state",
                "calendar_period",
                "calendar_date_precision",
                "calendar_source_url",
                "calendar_source_doi",
            }.issubset(catalog.columns)
        )
        populated = catalog[catalog["file_count"] > 0]
        self.assertTrue((populated["end_time"] >= populated["start_time"]).all())
        self.assertTrue(catalog["calendar_date_precision"].eq("day").all())
        self.assertTrue(
            catalog["calendar_reactor_state"].isin(
                {"operating", "outage", "mixed", "unknown"}
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
