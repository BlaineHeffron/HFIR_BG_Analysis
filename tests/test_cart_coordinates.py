"""Check vertical signs, point identity, and the historical horizontal API."""
import unittest

from src.database.CartScanFiles import convert_cart_coord_to_det_coord


class CartCoordinateTests(unittest.TestCase):
    def test_rotation_and_legacy_projection(self):
        # Synthetic 50-inch pivot; this is not an HFIR height estimate.
        for corners in [(30.5, 216.5, 30.5, 199.5), (0, 0, 17, 0)]:
            for angle, height in [(0, 42), (90, 50), (180, 58)]:
                with self.subTest(corners=corners, angle=angle):
                    horizontal = convert_cart_coord_to_det_coord(*corners, angle)
                    xyz = convert_cart_coord_to_det_coord(
                        *corners, angle, axis_height_inches=50)
                    self.assertEqual(horizontal, [xyz[0], xyz[2]])
                    self.assertAlmostEqual(xyz[1], height)
        self.assertEqual(convert_cart_coord_to_det_coord(
            30.5, 216.5, 30.5, 199.5, 0), [46.5, 207.7])

    def test_point_behind_face_and_missing_height(self):
        corners = (30.5, 216.5, 30.5, 199.5)
        for angle, expected in [(0, [46.5, 44, 207.7]),
                                (90, [46.5, 50, 213.7]),
                                (180, [46.5, 56, 207.7])]:
            xyz = convert_cart_coord_to_det_coord(
                *corners, angle, axis_height_inches=50, face_to_point_inches=2)
            for actual, target in zip(xyz, expected):
                self.assertAlmostEqual(actual, target)
        with self.assertRaises(ValueError):
            convert_cart_coord_to_det_coord(*corners, 0, face_to_point_inches=2)


if __name__ == "__main__":
    unittest.main()
