"""Tests for the ROOT-free Figure 7 floor-scan analysis."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.analyze_floor_scan_statistics import (
    fixed_width_histogram,
    representative_point,
    select_floor_scan_files,
)
from src.public_data.browser import PublicSpectrum


class FloorScanStatisticsTests(unittest.TestCase):
    def test_selection_excludes_later_monitoring_and_flags_extended_runs(self):
        common = {
            "shield_name": "collimator30",
            "coordinate_angle": 0,
            "coordinate_track": 0,
            "coordinate_Rx": 10.0,
            "coordinate_Rz": 20.0,
            "run_description": "ordinary scan",
            "start_time": 1,
            "calibration_A0": 0.0,
            "calibration_A1": 0.7,
        }
        metadata = pd.DataFrame(
            [
                dict(common, file_id=1, run_name="position_scan_7_POINT", live_time=240),
                dict(common, file_id=2, run_name="position_scan_6_LONG", live_time=800),
                dict(common, file_id=3, run_name="Cycle492_MIF_down", live_time=900),
                dict(common, file_id=4, run_name="position_scan_9_ANGLE", live_time=240),
            ]
        )
        selected = select_floor_scan_files(metadata)
        self.assertEqual(selected["file_id"].tolist(), [1, 2])
        self.assertEqual(selected["routine_point"].tolist(), [True, False])

    def test_fixed_width_histogram_preserves_counts_and_errors(self):
        spectrum = PublicSpectrum(
            file_id=1,
            run_id=2,
            file_name="test",
            run_name="test run",
            live_time=10.0,
            calibration_A0=0.0,
            calibration_A1=1.0,
            counts=np.asarray([1.0, 4.0, 9.0, 16.0]),
            energy_keV=np.asarray([1.0, 2.0, 3.0, 4.0]),
            bin_width_keV=np.ones(4),
            metadata={},
        )
        frame = fixed_width_histogram(spectrum, 0, 4, 2)
        np.testing.assert_allclose(frame["counts"], [1, 29])
        np.testing.assert_allclose(frame["statistical_error_counts"], [1, np.sqrt(29)])
        self.assertEqual(frame["counts"].sum(), 30)

    def test_representative_point_uses_live_counts_and_rate_medians(self):
        points = pd.DataFrame(
            {
                "file_id": [1, 2, 3],
                "live_time_s": [100.0, 200.0, 900.0],
                "counts_50_11400_keV": [1000.0, 4000.0, 90000.0],
                "rate_50_11400_counts_per_s": [10.0, 20.0, 100.0],
            }
        )
        self.assertEqual(int(representative_point(points)["file_id"]), 2)


if __name__ == "__main__":
    unittest.main()
