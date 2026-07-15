"""Tests for the browser's ROOT- and Streamlit-free UI helpers."""

import unittest

import pandas as pd

from webapp.helpers import (
    DEFAULT_MAP_X_RANGE, DEFAULT_MAP_Z_RANGE, cart_azimuth_degrees,
    detector_face_position, filter_catalog, location_catalog,
    measurement_map_ranges,
)


class WebappHelperTests(unittest.TestCase):
    def setUp(self):
        self.catalog = pd.DataFrame([
            dict(run_id=1, run_name="MIF on", run_description="alpha", calendar_cycle="490A", calendar_reactor_state="operating", shield_name="none", detector_coordinates_id=8, coordinate_Rx=10.0, coordinate_Rz=20.0, coordinate_Lx=10.0, coordinate_Lz=10.0, orientation_angle=90.0, file_count=2),
            dict(run_id=2, run_name="MIF off", run_description="beta", calendar_cycle="unknown", calendar_reactor_state="outage", shield_name="lead", detector_coordinates_id=9, coordinate_Rx=20.0, coordinate_Rz=20.0, coordinate_Lx=10.0, coordinate_Lz=20.0, orientation_angle=0.0, file_count=3),
        ])

    def test_filter_catalog_combines_filters_and_plain_text(self):
        result = filter_catalog(self.catalog, cycles=["490A"], states=["operating"], text="ALPHA")
        self.assertEqual(result["run_id"].tolist(), [1])
        self.assertEqual(filter_catalog(self.catalog, shields=["lead"], coordinate_id=9)["run_id"].tolist(), [2])

    def test_public_coordinate_conventions(self):
        self.assertAlmostEqual(cart_azimuth_degrees(10, 20, 10, 10), 0.0)
        z, x = detector_face_position(10, 20, 10, 10, 90)
        self.assertAlmostEqual(z, 19.2)
        self.assertAlmostEqual(x, 26.0)

    def test_location_catalog_aggregates_files(self):
        locations = location_catalog(self.catalog)
        self.assertEqual(len(locations), 2)
        self.assertEqual(int(locations["file_count"].sum()), 5)
        self.assertIn("cart_azimuth", locations)

    def test_measurement_map_ranges_include_faces_and_cart_corners(self):
        locations = location_catalog(self.catalog)
        z_range, x_range = measurement_map_ranges(locations)
        self.assertLess(z_range[0], locations[["map_z", "coordinate_Rz", "coordinate_Lz"]].min().min())
        self.assertGreater(z_range[1], locations[["map_z", "coordinate_Rz", "coordinate_Lz"]].max().max())
        self.assertGreater(x_range[0], locations[["map_x", "coordinate_Rx", "coordinate_Lx"]].max().max())
        self.assertLess(x_range[1], locations[["map_x", "coordinate_Rx", "coordinate_Lx"]].min().min())
        self.assertGreater(x_range[1], -50.0)

    def test_empty_measurement_map_uses_released_region_default(self):
        self.assertEqual(
            measurement_map_ranges(pd.DataFrame()),
            (DEFAULT_MAP_Z_RANGE, DEFAULT_MAP_X_RANGE),
        )


if __name__ == "__main__":
    unittest.main()
