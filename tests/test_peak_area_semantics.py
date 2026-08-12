"""Focused unit checks for HPGe peak-window estimands."""

from __future__ import annotations

import unittest
from math import erf, sqrt
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.rd_peak_fitter import fit_relative_areas
from src.analysis.Spectrum import (
    MultiPeakFit,
    PEAK_AREA_LEGACY_DENSITY,
    PEAK_AREA_NET_COUNTS,
    PeakFit,
    ge_multi_peak_function,
    ge_peak_function,
)


def _single_peak(bin_width, sigma, net_counts=10_000.0):
    centroid = 100.0
    xs = np.arange(70.0, 130.0 + bin_width / 2.0, bin_width)
    height_density = net_counts / (np.sqrt(2.0 * np.pi) * sigma)
    parameters = np.array(
        [height_density * bin_width, 0.0, centroid, sigma, sigma, 2.0 * bin_width, 0.0, 0.0]
    )
    ys = ge_peak_function(xs, *parameters)
    return PeakFit(
        parameters,
        np.ones(parameters.size),
        np.eye(parameters.size),
        xs,
        ys,
    )


def _two_peaks(bin_width):
    xs = np.arange(70.0, 140.0 + bin_width / 2.0, bin_width)
    parameters = np.array(
        [
            10_000.0 / np.sqrt(2.0 * np.pi) * bin_width,
            0.0,
            90.0,
            1.0,
            1.0,
            5_000.0 / (2.0 * np.sqrt(2.0 * np.pi)) * bin_width,
            0.0,
            120.0,
            2.0,
            2.0,
            2.0 * bin_width,
            0.0,
            0.0,
        ]
    )
    ys = ge_multi_peak_function(xs, *parameters)
    return MultiPeakFit(
        parameters,
        np.ones(parameters.size),
        np.eye(parameters.size),
        xs,
        ys,
    )


class PeakAreaUnitTests(unittest.TestCase):
    def test_refutes_material_bin_width_dependence(self):
        results = [
            _single_peak(width, 2.0).area()[0]
            for width in (0.25, 0.5, 1.0, 2.0)
        ]
        expected = 10_000.0 * erf(3.5 / sqrt(2.0))
        self.assertLess((max(results) - min(results)) / expected, 2e-3)

    def test_net_counts_recover_truncated_gaussian_area(self):
        expected = 10_000.0 * erf(3.5 / sqrt(2.0))
        for bin_width, sigma in ((0.125, 1.5), (0.25, 2.0), (0.5, 1.5), (1.0, 1.5)):
            result = _single_peak(bin_width, sigma).area()[0]
            self.assertAlmostEqual(result / expected, 1.0, delta=1e-3)

    def test_legacy_value_is_density_not_area(self):
        fit = _single_peak(0.25, 2.0)
        net_value, net_error = fit.area()
        density_value, density_error = fit.area(
            mode=PEAK_AREA_LEGACY_DENSITY
        )
        self.assertAlmostEqual(density_value, net_value / (7.0 * fit.sigma))
        self.assertAlmostEqual(density_error, net_error / (7.0 * fit.sigma))

    def test_legacy_ratio_has_hidden_resolution_dependence(self):
        narrow = _single_peak(0.125, 1.0)
        wide = _single_peak(0.125, 2.0)
        narrow_net = narrow.area()[0]
        wide_net = wide.area()[0]
        narrow_density = narrow.area(mode=PEAK_AREA_LEGACY_DENSITY)[0]
        wide_density = wide.area(mode=PEAK_AREA_LEGACY_DENSITY)[0]
        self.assertAlmostEqual(narrow_net / wide_net, 1.0, places=4)
        self.assertAlmostEqual(narrow_density / wide_density, 2.0, places=4)

    def test_multi_peak_modes_have_the_same_units(self):
        fit = _two_peaks(0.25)
        net = fit.area(mode=PEAK_AREA_NET_COUNTS)
        density = fit.area(mode=PEAK_AREA_LEGACY_DENSITY)
        truncated_fraction = erf(3.5 / sqrt(2.0))
        self.assertAlmostEqual(net[0][0], 10_000.0 * truncated_fraction, delta=2.0)
        self.assertAlmostEqual(net[1][0], 5_000.0 * truncated_fraction, delta=2.0)
        for index, sigma in enumerate(fit.sigmas):
            self.assertAlmostEqual(density[index][0], net[index][0] / (7.0 * sigma))
            self.assertAlmostEqual(density[index][1], net[index][1] / (7.0 * sigma))

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "peak area mode"):
            _single_peak(0.25, 1.0).area(mode="area-ish")


class _FakePeakFit:
    def __init__(self, net_value, net_error, width):
        self.net_value = net_value
        self.net_error = net_error
        self.width = width

    def area(self, mode=PEAK_AREA_NET_COUNTS):
        if mode == PEAK_AREA_NET_COUNTS:
            return self.net_value, self.net_error
        if mode == PEAK_AREA_LEGACY_DENSITY:
            return self.net_value / self.width, self.net_error / self.width
        raise AssertionError(mode)


class RelativeAreaUncertaintyTests(unittest.TestCase):
    def setUp(self):
        self.spectrum = SimpleNamespace(fname="test.txt", live=10.0)
        self.fits = {
            100.0: _FakePeakFit(50.0, 5.0, 5.0),
            200.0: _FakePeakFit(25.0, 2.5, 10.0),
        }

    def test_corrected_reference_self_ratio_has_zero_uncertainty(self):
        with patch("scripts.rd_peak_fitter.fit_spectra", return_value=self.fits):
            result = fit_relative_areas([self.spectrum], [100.0, 200.0], 100.0)
        self.assertEqual(result["100.00"], (1.0, 0.0))
        self.assertAlmostEqual(result["200.00"][0], 0.5)

    def test_paper_legacy_path_preserves_density_and_self_uncertainty(self):
        with patch("scripts.rd_peak_fitter.fit_spectra", return_value=self.fits):
            result = fit_relative_areas(
                [self.spectrum],
                [100.0, 200.0],
                100.0,
                area_mode=PEAK_AREA_LEGACY_DENSITY,
                preserve_legacy_reference_uncertainty=True,
            )
        self.assertGreater(result["100.00"][1], 0.0)
        self.assertAlmostEqual(result["200.00"][0], 0.25)


if __name__ == "__main__":
    unittest.main()
