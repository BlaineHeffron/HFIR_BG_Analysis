"""Unit tests for the supported public-data export command."""

from __future__ import annotations

import unittest

import pandas as pd

from scripts.export_public_data import filter_run_catalog, parse_args, selected_file_ids


class PublicDataExportTests(unittest.TestCase):
    def setUp(self):
        self.catalog = pd.DataFrame(
            [
                {
                    "run_name": "MIF reactor on",
                    "run_description": "baseline",
                    "calendar_cycle": "491",
                    "calendar_reactor_state": "operating",
                    "shield_name": "none",
                },
                {
                    "run_name": "shield center",
                    "run_description": "lead layer",
                    "calendar_cycle": "unknown",
                    "calendar_reactor_state": "outage",
                    "shield_name": "lead",
                },
            ]
        )
        self.metadata = pd.DataFrame(
            {"file_id": [10, 11, 12], "run_id": [1, 1, 2]}
        )

    def test_catalog_filters_are_combined(self):
        args = parse_args(
            ["catalog", "--cycle", "491", "--calendar-state", "operating", "--contains", "REACTOR"]
        )
        result = filter_run_catalog(self.catalog, args)
        self.assertEqual(result["run_name"].tolist(), ["MIF reactor on"])

    def test_file_and_run_selections_are_deduplicated(self):
        self.assertEqual(selected_file_ids(self.metadata, [10, 12], [1]), [10, 11, 12])

    def test_empty_and_unknown_spectrum_selections_fail_cleanly(self):
        with self.assertRaises(SystemExit):
            selected_file_ids(self.metadata, None, None)
        with self.assertRaises(SystemExit):
            selected_file_ids(self.metadata, [99], None)


if __name__ == "__main__":
    unittest.main()
