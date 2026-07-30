"""Tests for the ROOT-free Poisson parent-group fitter."""

from __future__ import annotations

import importlib
import sys
import unittest
from unittest.mock import patch

import numpy as np

from src.public_data.browser import PublicSpectrum, rebin_by_factor
from src.public_data.parent_group_fit import (
    AffineCalibrationConstraint,
    FitWindow,
    ParentGroupSpec,
    PeakComponent,
    fit_parent_group,
    gaussian_bin_probabilities,
    poisson_nll,
)


def synthetic_spectrum(
    *,
    seed: int = 731,
    include_contaminant: bool = False,
) -> tuple[PublicSpectrum, ParentGroupSpec, AffineCalibrationConstraint]:
    rng = np.random.default_rng(seed)
    channel_count = 640
    A0 = 0.0
    A1 = 1.0
    channels = np.arange(1, channel_count + 1, dtype=np.float64)
    low_edges = channels - 0.5
    high_edges = channels + 0.5
    expected = np.full(channel_count, 2.0)
    components = [
        PeakComponent("parent", "fep", 500.0, "fep", 2.2, 8.0, 0.8, 5.0),
        PeakComponent("single_escape", "sep", 300.0, "sep", 2.0, 8.0, 0.8, 5.0),
        PeakComponent("double_escape", "dep", 100.0, "dep", 1.8, 8.0, 0.8, 5.0),
    ]
    truths = (
        (500.4, 2.2, 2400.0),
        (300.3, 2.0, 1200.0),
        (100.2, 1.8, 600.0),
    )
    for centroid, sigma, area in truths:
        expected += area * gaussian_bin_probabilities(
            low_edges, high_edges, centroid, sigma
        )
    if include_contaminant:
        components.append(
            PeakComponent(
                "known_contaminant",
                "contaminant",
                308.0,
                "sep",
                1.7,
                4.0,
                0.8,
                4.0,
            )
        )
        expected += 500.0 * gaussian_bin_probabilities(
            low_edges, high_edges, 308.1, 1.7
        )
    counts = rng.poisson(expected).astype(np.float64)
    spectrum = PublicSpectrum(
        file_id=1,
        run_id=1,
        file_name="synthetic",
        run_name="synthetic",
        live_time=100.0,
        calibration_A0=A0,
        calibration_A1=A1,
        counts=counts,
        energy_keV=A0 + A1 * channels,
        bin_width_keV=np.full(channel_count, A1),
        metadata={"synthetic": True},
    )
    spec = ParentGroupSpec(
        "synthetic-parent",
        (
            FitWindow("dep", 80.0, 120.0),
            FitWindow("sep", 280.0, 320.0),
            FitWindow("fep", 480.0, 520.0),
        ),
        tuple(components),
    )
    calibration = AffineCalibrationConstraint(
        A0,
        A1,
        np.diag([0.01**2, 1e-5**2]),
    )
    return spectrum, spec, calibration


class PrimitiveTests(unittest.TestCase):
    def test_gaussian_is_integrated_over_bin_edges(self):
        edges = np.linspace(-12.0, 12.0, 241)
        probabilities = gaussian_bin_probabilities(
            edges[:-1], edges[1:], 0.0, 1.0
        )
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=12)
        np.testing.assert_allclose(probabilities, probabilities[::-1], atol=1e-15)

    def test_sparse_poisson_nll_handles_zero_counts(self):
        value = poisson_nll(np.asarray([0, 1, 4]), np.asarray([0.0, 1.0, 3.5]))
        self.assertTrue(np.isfinite(value))
        self.assertEqual(poisson_nll(np.asarray([1]), np.asarray([0.0])), np.inf)
        with self.assertRaisesRegex(ValueError, "integer counts"):
            poisson_nll(np.asarray([0.5]), np.asarray([1.0]))

    def test_import_does_not_require_root(self):
        module = sys.modules["src.public_data.parent_group_fit"]
        with patch.dict(sys.modules, {"ROOT": None}):
            importlib.reload(module)


class ParentGroupFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spectrum, cls.spec, cls.calibration = synthetic_spectrum()
        cls.result = fit_parent_group(cls.spectrum, cls.spec, cls.calibration)

    def test_recovers_three_target_lines_and_covariance(self):
        result = self.result
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.target_labels, ("fep", "sep", "dep"))
        np.testing.assert_allclose(
            result.target_areas_counts,
            [2400.0, 1200.0, 600.0],
            rtol=0.16,
        )
        np.testing.assert_allclose(result.ratios, [0.5, 0.25], atol=0.08)
        self.assertEqual(result.target_area_covariance.shape, (3, 3))
        self.assertEqual(result.ratio_covariance.shape, (2, 2))
        self.assertTrue(np.isfinite(result.covariance).all())
        self.assertEqual(result.covariance_rank, len(result.parameter_names))
        self.assertGreater(result.degrees_of_freedom, 0)
        self.assertGreaterEqual(result.poisson_deviance, 0)

    def test_recovers_centroids_and_widths(self):
        result = self.result
        for name, shift, sigma in (
            ("parent", 0.4, 2.2),
            ("single_escape", 0.3, 2.0),
            ("double_escape", 0.2, 1.8),
        ):
            self.assertAlmostEqual(
                result.parameter(f"line.{name}.centroid_shift_keV"),
                shift,
                delta=0.5,
            )
            self.assertAlmostEqual(
                result.parameter(f"line.{name}.sigma_keV"),
                sigma,
                delta=0.5,
            )

    def test_optional_contaminant_is_fitted_jointly(self):
        spectrum, spec, calibration = synthetic_spectrum(
            seed=801, include_contaminant=True
        )
        result = fit_parent_group(spectrum, spec, calibration)
        self.assertTrue(result.success, result.message)
        self.assertGreater(result.parameter("line.known_contaminant.area_counts"), 250)
        self.assertAlmostEqual(result.ratios[0], 0.5, delta=0.1)

    def test_calibration_uncertainty_propagates_to_centroid_covariance(self):
        tight = self.result
        broad_calibration = AffineCalibrationConstraint(
            0.0,
            1.0,
            np.diag([0.5**2, 5e-4**2]),
        )
        broad = fit_parent_group(self.spectrum, self.spec, broad_calibration)
        name = "line.parent.centroid_shift_keV"
        tight_index = tight.parameter_names.index(name)
        broad_index = broad.parameter_names.index(name)
        self.assertGreater(
            broad.covariance[broad_index, broad_index],
            tight.covariance[tight_index, tight_index],
        )

    def test_invalid_shapes_and_overlapping_windows_are_rejected(self):
        rebinned = rebin_by_factor(self.spectrum, 2)
        with self.assertRaisesRegex(ValueError, "unrebinned"):
            fit_parent_group(rebinned, self.spec, self.calibration)

        overlapping = ParentGroupSpec(
            "overlap",
            (
                FitWindow("dep", 80.0, 120.0),
                FitWindow("sep", 110.0, 320.0),
                FitWindow("fep", 480.0, 520.0),
            ),
            self.spec.components,
        )
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            fit_parent_group(self.spectrum, overlapping, self.calibration)


if __name__ == "__main__":
    unittest.main()
