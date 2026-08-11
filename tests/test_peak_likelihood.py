"""Manufactured tests for the paper-facing joint Poisson peak likelihood."""

from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

import src.public_data.peak_likelihood as likelihood
from src.public_data.browser import PublicSpectrum, rebin_by_factor
from src.public_data.peak_likelihood import (
    CalibrationConstraint,
    FitWindow,
    JointPeakSpec,
    LineComponent,
    LinearResolution,
    RatioDefinition,
    fit_joint_peak_model,
    gaussian_bin_probabilities,
    parametric_bootstrap,
    peak_shape_bin_probabilities,
    profile_ratio_interval,
    ratio_values_and_covariance,
    ratios_from_fit,
    stable_seed,
)
from scripts.reanalyze_paper_peak_statistics import _fit_diagnostics


def manufactured_case(
    *,
    seed: int = 417,
    live_times: tuple[float, ...] = (500.0,),
    signal_scales: tuple[float, ...] | None = None,
    background_scales: tuple[float, ...] | None = None,
    calibration_offsets_keV: tuple[float, ...] | None = None,
    resolution_scales: tuple[float, ...] | None = None,
    component_origin_classes: dict[str, str] | None = None,
    run_origin_scales: tuple[dict[str, float], ...] | None = None,
    weak_rate: float = 0.035,
    poisson: bool = True,
) -> tuple[
    tuple[PublicSpectrum, ...],
    JointPeakSpec,
    CalibrationConstraint,
    LinearResolution,
    dict[str, float],
]:
    rng = np.random.default_rng(seed)
    signal_scales = signal_scales or tuple(1.0 for _ in live_times)
    background_scales = background_scales or tuple(1.0 for _ in live_times)
    calibration_offsets_keV = calibration_offsets_keV or tuple(
        0.0 for _ in live_times
    )
    resolution_scales = resolution_scales or tuple(1.0 for _ in live_times)
    run_origin_scales = run_origin_scales or tuple({} for _ in live_times)
    if not (
        len(live_times)
        == len(signal_scales)
        == len(background_scales)
        == len(calibration_offsets_keV)
        == len(resolution_scales)
        == len(run_origin_scales)
    ):
        raise ValueError("manufactured run controls must have matching lengths")
    channel_count = 720
    channels = np.arange(1, channel_count + 1, dtype=np.float64)
    low_edges = channels - 0.5
    high_edges = channels + 0.5
    windows = (
        FitWindow("low", 80.0, 135.0),
        FitWindow("middle", 275.0, 335.0),
        FitWindow("high", 475.0, 535.0),
    )
    components = (
        LineComponent("weak", 105.0, "low"),
        LineComponent("middle_a", 300.0, "middle"),
        LineComponent("middle_b", 310.0, "middle"),
        LineComponent("reference", 505.0, "high"),
    )
    if component_origin_classes is not None:
        components = tuple(
            replace(
                component,
                origin_class=component_origin_classes[component.name],
            )
            for component in components
        )
    spec = JointPeakSpec("manufactured-joint-case", windows, components)
    truth = {
        "weak": weak_rate,
        "middle_a": 0.8,
        "middle_b": 0.42,
        "reference": 1.15,
    }
    resolution = LinearResolution(
        0.85,
        0.0011,
        0.18,
        1.7,
        intercept_bounds_keV=(0.3, 2.0),
        slope_bounds=(0.0, 0.003),
        tail_fraction_bounds=(0.0, 0.5),
        tail_scale_bounds_in_sigma=(0.3, 6.0),
    )
    spectra: list[PublicSpectrum] = []
    for run_index, (
        live_time,
        signal_scale,
        background_scale,
        calibration_offset_keV,
        resolution_scale,
        origin_scales,
    ) in enumerate(
        zip(
            live_times,
            signal_scales,
            background_scales,
            calibration_offsets_keV,
            resolution_scales,
            run_origin_scales,
        )
    ):
        background_density = background_scale * (
            0.18 + 0.00012 * channels
        )
        expected = live_time * background_density
        for component in components:
            sigma = resolution_scale * (
                resolution.intercept_keV
                + resolution.slope * component.energy_keV
            )
            expected += (
                live_time
                * origin_scales.get(component.origin_class, signal_scale)
                * truth[component.name]
                * peak_shape_bin_probabilities(
                    low_edges + calibration_offset_keV,
                    high_edges + calibration_offset_keV,
                    component.energy_keV,
                    sigma,
                    resolution.tail_fraction,
                    resolution.tail_scale_in_sigma,
                )
            )
        counts = (
            rng.poisson(expected).astype(np.float64)
            if poisson
            else np.rint(expected).astype(np.float64)
        )
        spectra.append(
            PublicSpectrum(
                file_id=run_index + 1,
                run_id=10,
                file_name=f"manufactured_{run_index}",
                run_name="manufactured",
                live_time=live_time,
                calibration_A0=0.0,
                calibration_A1=1.0,
                counts=counts,
                energy_keV=channels.copy(),
                bin_width_keV=np.ones(channel_count),
                metadata={"manufactured": True},
            )
        )
    calibration = CalibrationConstraint(
        0.0,
        0.0,
        np.diag([0.08**2, 8e-5**2]),
        (-0.5, 0.5),
        (-5e-4, 5e-4),
        (
            np.diag([0.25**2, 1e-4**2])
            if any(value != 0.0 for value in calibration_offsets_keV)
            or any(value != 1.0 for value in resolution_scales)
            else None
        ),
    )
    if any(value != 1.0 for value in resolution_scales):
        resolution = replace(
            resolution,
            per_run_scale_sigma=0.15,
            per_run_scale_bounds=(0.6, 1.4),
        )
    return tuple(spectra), spec, calibration, resolution, truth


class PrimitiveAndCovarianceTests(unittest.TestCase):
    def test_bin_integrated_shapes_are_normalized(self):
        edges = np.linspace(-80.0, 80.0, 32001)
        gaussian = gaussian_bin_probabilities(edges[:-1], edges[1:], 0.0, 1.2)
        tailed = peak_shape_bin_probabilities(
            edges[:-1], edges[1:], 0.0, 1.2, 0.25, 2.5
        )
        self.assertAlmostEqual(float(gaussian.sum()), 1.0, places=12)
        self.assertAlmostEqual(float(tailed.sum()), 1.0, places=10)
        self.assertTrue(np.all(tailed >= 0.0))

    def test_general_ratio_covariance_uses_covariance_terms(self):
        names = ("a", "b", "c")
        rates = np.asarray([10.0, 5.0, 2.0])
        covariance = np.asarray(
            [[4.0, 1.2, 0.3], [1.2, 2.0, 0.4], [0.3, 0.4, 1.0]]
        )
        definitions = (
            RatioDefinition("a/c", "a", "c"),
            RatioDefinition("b/c", "b", "c"),
            RatioDefinition("c/c", "c", "c"),
        )
        result = ratio_values_and_covariance(
            names, rates, covariance, definitions
        )
        expected_variance = (
            covariance[0, 0] / rates[2] ** 2
            + rates[0] ** 2 * covariance[2, 2] / rates[2] ** 4
            - 2.0 * rates[0] * covariance[0, 2] / rates[2] ** 3
        )
        self.assertAlmostEqual(result.covariance[0, 0], expected_variance)
        self.assertNotEqual(result.covariance[0, 1], 0.0)
        self.assertEqual(result.values[2], 1.0)
        np.testing.assert_array_equal(result.jacobian[2], np.zeros(3))
        np.testing.assert_array_equal(result.covariance[2], np.zeros(3))
        np.testing.assert_array_equal(result.covariance[:, 2], np.zeros(3))

    def test_stable_seed_depends_only_on_case_identity(self):
        self.assertEqual(stable_seed("case-a"), stable_seed("case-a"))
        self.assertNotEqual(stable_seed("case-a"), stable_seed("case-b"))

    def test_normal_and_chi_square_diagnostics_exclude_low_expected_bins(self):
        result = SimpleNamespace(
            covariance=np.ones((1, 1)),
            poisson_deviance=100.0,
            degrees_of_freedom=3,
            expected_counts=np.asarray([1.0, 4.9, 5.0, 100.0]),
            observed_counts=np.asarray([25.0, 25.0, 5.0, 100.0]),
            observation_window_names=("one", "one", "one", "one"),
            parameter_names=("parameter",),
            parameter_values=np.asarray([1.0]),
            success=True,
            message="ok",
            optimizer_iterations=1,
            optimizer_evaluations=2,
            data_poisson_nll=50.0,
            gaussian_nuisance_prior_deviance=0.0,
            calibration_prior_deviance=0.0,
            penalized_nll=50.0,
            penalized_degrees_of_freedom=5,
            fisher_rank=1,
            fisher_condition=1.0,
            fisher_covariance_valid=True,
            active_bounds=(),
        )
        diagnostics = _fit_diagnostics(
            result,
            {
                "chi_square_minimum_expected_counts_per_bin": 5.0,
                "per_bin_outlier_sigma_thresholds": [4.0, 5.0],
                "global_deviance_p_value_minimum": 0.001,
                "familywise_diagnostic_alpha": 0.01,
            },
        )
        self.assertEqual(diagnostics["chi_square_reference_bin_count"], 2)
        self.assertEqual(diagnostics["excluded_low_expected_bin_count"], 2)
        self.assertEqual(
            diagnostics["per_bin_outlier_diagnostics"][0][
                "observed_outlier_count"
            ],
            0,
        )
        self.assertEqual(
            diagnostics["window_deviance_diagnostics"][0][
                "chi_square_reference_bin_count"
            ],
            2,
        )


class JointFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spectra, cls.spec, cls.calibration, cls.resolution, cls.truth = (
            manufactured_case(
                live_times=(500.0, 350.0),
                signal_scales=(1.0, 0.82),
                background_scales=(1.0, 1.18),
            )
        )
        cls.result = fit_joint_peak_model(
            cls.spectra, cls.spec, cls.calibration, cls.resolution
        )

    def test_recovers_isolated_and_multiplet_rates_in_simultaneous_runs(self):
        self.assertTrue(self.result.success, self.result.message)
        for name in ("middle_a", "middle_b", "reference"):
            self.assertAlmostEqual(
                self.result.line_rate(name), self.truth[name], delta=0.16 * self.truth[name]
            )
        self.assertAlmostEqual(
            self.result.parameter("spectrum.1.signal_scale_relative_to_spectrum_0"),
            0.82,
            delta=0.12,
        )
        self.assertAlmostEqual(
            self.result.parameter("spectrum.1.background_scale_relative_to_spectrum_0"),
            1.18,
            delta=0.12,
        )

    def test_full_covariance_retains_background_and_shared_denominator_correlations(self):
        rate_index = self.result.parameter_names.index(
            "line.middle_a.rate_counts_per_s"
        )
        background_index = self.result.parameter_names.index(
            "background.middle.low_counts_per_s_per_keV"
        )
        self.assertNotEqual(
            self.result.covariance[rate_index, background_index], 0.0
        )
        ratios = ratios_from_fit(
            self.result,
            (
                RatioDefinition("a/ref", "middle_a", "reference"),
                RatioDefinition("b/ref", "middle_b", "reference"),
            ),
        )
        self.assertNotEqual(ratios.covariance[0, 1], 0.0)

    def test_count_deviance_and_calibration_prior_conventions_are_separate(self):
        self.assertAlmostEqual(
            self.result.data_poisson_nll,
            0.5 * self.result.poisson_deviance,
            places=8,
        )
        self.assertGreaterEqual(self.result.calibration_prior_deviance, 0.0)
        self.assertAlmostEqual(
            self.result.penalized_nll,
            self.result.data_poisson_nll
            + 0.5 * self.result.calibration_prior_deviance,
            places=10,
        )
        self.assertEqual(
            self.result.penalized_degrees_of_freedom,
            self.result.degrees_of_freedom + 2,
        )

    def test_raw_bin_requirement_rejects_rebinned_input(self):
        rebinned = rebin_by_factor(self.spectra[0], 2)
        with self.assertRaisesRegex(ValueError, "unrebinned"):
            fit_joint_peak_model(
                (rebinned,), self.spec, self.calibration, self.resolution
            )

    def test_analytic_gradient_matches_finite_difference(self):
        problem = likelihood._prepare_problem(
            self.spectra,
            self.spec,
            self.calibration,
            self.resolution,
            dict(zip(self.result.parameter_names, self.result.parameter_values)),
        )
        parameters = problem.initial.copy()
        _, analytic = likelihood._objective_and_gradient(problem, parameters)
        for index in (0, 1, 2, 3, 4, 5, 6, parameters.size - 1):
            step = max(abs(parameters[index]) * 2e-6, problem.scales[index] * 2e-6, 1e-8)
            plus = parameters.copy()
            minus = parameters.copy()
            plus[index] += step
            minus[index] -= step
            numerical = (
                likelihood._objective_and_gradient(problem, plus)[0]
                - likelihood._objective_and_gradient(problem, minus)[0]
            ) / (2.0 * step)
            self.assertAlmostEqual(
                analytic[index],
                numerical,
                delta=max(2e-3, 2e-3 * abs(numerical)),
                msg=f"gradient mismatch for {problem.parameter_names[index]}",
            )

    def test_deterministic_bootstrap_reports_coverage_diagnostics(self):
        first = parametric_bootstrap(
            self.spectra,
            self.spec,
            self.calibration,
            self.resolution,
            self.result,
            case_identity="manufactured-bootstrap",
            replicates=5,
        )
        second = parametric_bootstrap(
            self.spectra,
            self.spec,
            self.calibration,
            self.resolution,
            self.result,
            case_identity="manufactured-bootstrap",
            replicates=5,
        )
        self.assertEqual(first.seed, second.seed)
        self.assertEqual(first.successful_replicates, 5)
        np.testing.assert_allclose(
            first.empirical_standard_deviation,
            second.empirical_standard_deviation,
        )
        self.assertTrue(np.all((first.fisher_95_percent_coverage >= 0.0)))
        self.assertTrue(np.all((first.fisher_95_percent_coverage <= 1.0)))

    def test_sqrt_resolution_without_tail_omits_unidentified_tail_parameters(self):
        sqrt_resolution = LinearResolution(
            0.8,
            0.001,
            0.0,
            2.0,
            form="sqrt",
            tail_model="none",
            intercept_bounds_keV=(0.2, 2.0),
            slope_bounds=(0.0, 0.02),
        )
        sigma, derivative_intercept, derivative_slope = (
            likelihood.resolution_sigma_and_derivatives(
                sqrt_resolution, 500.0
            )
        )
        self.assertAlmostEqual(sigma**2, 0.8**2 + 0.001 * 500.0)
        self.assertAlmostEqual(derivative_intercept, 0.8 / sigma)
        self.assertAlmostEqual(derivative_slope, 500.0 / (2.0 * sigma))
        result = fit_joint_peak_model(
            self.spectra, self.spec, self.calibration, sqrt_resolution
        )
        self.assertTrue(result.success, result.message)
        self.assertIn("resolution.variance_slope_keV", result.parameter_names)
        self.assertFalse(
            any(name.startswith("shape.") for name in result.parameter_names)
        )

    def test_quadratic_background_uses_calibrated_fraction_and_valid_gradient(self):
        quadratic_spec = replace(
            self.spec,
            name="manufactured-quadratic-background",
            windows=tuple(
                replace(window, background_model="quadratic")
                for window in self.spec.windows
            ),
        )
        result = fit_joint_peak_model(
            self.spectra, quadratic_spec, self.calibration, self.resolution
        )
        self.assertTrue(result.success, result.message)
        middle_name = "background.middle.middle_counts_per_s_per_keV"
        self.assertIn(middle_name, result.parameter_names)
        problem = likelihood._prepare_problem(
            self.spectra,
            quadratic_spec,
            self.calibration,
            self.resolution,
            dict(zip(result.parameter_names, result.parameter_values)),
        )
        parameters = problem.initial.copy()
        _, analytic = likelihood._objective_and_gradient(problem, parameters)
        for index in (0, 1, problem.parameter_names.index(middle_name)):
            step = max(
                abs(parameters[index]) * 2e-6,
                problem.scales[index] * 2e-6,
                1e-8,
            )
            plus = parameters.copy()
            minus = parameters.copy()
            plus[index] += step
            minus[index] -= step
            numerical = (
                likelihood._objective_and_gradient(problem, plus)[0]
                - likelihood._objective_and_gradient(problem, minus)[0]
            ) / (2.0 * step)
            self.assertAlmostEqual(
                analytic[index],
                numerical,
                delta=max(4e-3, 3e-3 * abs(numerical)),
                msg=f"gradient mismatch for {problem.parameter_names[index]}",
            )

    def test_per_run_calibration_and_resolution_nuisances_recover_drift(self):
        spectra, spec, calibration, resolution, _ = manufactured_case(
            seed=27,
            live_times=(1800.0, 1600.0),
            calibration_offsets_keV=(0.0, 0.18),
            resolution_scales=(1.0, 1.12),
            poisson=False,
        )
        result = fit_joint_peak_model(spectra, spec, calibration, resolution)
        self.assertTrue(result.success, result.message)
        offset_name = "spectrum.1.calibration_offset_deviation_keV"
        stretch_name = "spectrum.1.fractional_gain_stretch_deviation"
        scale_name = "spectrum.1.resolution_scale_relative_to_spectrum_0"
        self.assertAlmostEqual(result.parameter(offset_name), 0.18, delta=0.08)
        self.assertAlmostEqual(result.parameter(scale_name), 1.12, delta=0.08)
        self.assertIn(stretch_name, result.parameter_names)
        self.assertEqual(
            result.penalized_degrees_of_freedom,
            result.degrees_of_freedom + 5,
        )

        problem = likelihood._prepare_problem(
            spectra,
            spec,
            calibration,
            resolution,
            dict(zip(result.parameter_names, result.parameter_values)),
        )
        parameters = problem.initial.copy()
        _, analytic = likelihood._objective_and_gradient(problem, parameters)
        for name in (offset_name, stretch_name, scale_name):
            index = problem.parameter_names.index(name)
            step = max(problem.scales[index] * 2e-6, 1e-8)
            plus = parameters.copy()
            minus = parameters.copy()
            plus[index] += step
            minus[index] -= step
            numerical = (
                likelihood._objective_and_gradient(problem, plus)[0]
                - likelihood._objective_and_gradient(problem, minus)[0]
            ) / (2.0 * step)
            self.assertAlmostEqual(
                analytic[index],
                numerical,
                delta=max(5e-3, 4e-3 * abs(numerical)),
                msg=f"gradient mismatch for {name}",
            )

    def test_declared_origin_classes_receive_independent_run_scales(self):
        origin_classes = {
            "weak": "radioactive_decay",
            "middle_a": "radioactive_decay",
            "middle_b": "neutron_capture",
            "reference": "neutron_capture",
        }
        spectra, spec, calibration, resolution, _ = manufactured_case(
            seed=91,
            live_times=(1600.0, 1500.0),
            component_origin_classes=origin_classes,
            run_origin_scales=(
                {"radioactive_decay": 1.0, "neutron_capture": 1.0},
                {"radioactive_decay": 1.12, "neutron_capture": 0.78},
            ),
            poisson=False,
        )
        result = fit_joint_peak_model(spectra, spec, calibration, resolution)
        self.assertTrue(result.success, result.message)
        decay_name = (
            "spectrum.1.signal_scale.radioactive_decay_relative_to_spectrum_0"
        )
        capture_name = (
            "spectrum.1.signal_scale.neutron_capture_relative_to_spectrum_0"
        )
        self.assertAlmostEqual(result.parameter(decay_name), 1.12, delta=0.08)
        self.assertAlmostEqual(result.parameter(capture_name), 0.78, delta=0.08)
        self.assertGreater(
            result.parameter(decay_name) - result.parameter(capture_name),
            0.2,
        )


class BoundaryAndCoverageTests(unittest.TestCase):
    def test_absent_line_gets_profile_upper_limit(self):
        spectra, spec, calibration, resolution, _ = manufactured_case(
            seed=71, weak_rate=0.0, poisson=False
        )
        result = fit_joint_peak_model(spectra, spec, calibration, resolution)
        self.assertTrue(result.success, result.message)
        interval = profile_ratio_interval(
            spectra,
            spec,
            calibration,
            resolution,
            result,
            RatioDefinition("weak/reference", "weak", "reference"),
            confidence_level=0.95,
            max_evaluations=30,
        )
        self.assertEqual(interval.kind, "upper_limit")
        self.assertEqual(interval.lower, 0.0)
        self.assertGreater(interval.upper, 0.0)

    def test_injected_rate_has_reasonable_fisher_coverage(self):
        covered = 0
        successful = 0
        truth = 0.8
        for seed in range(900, 912):
            spectra, spec, calibration, resolution, _ = manufactured_case(
                seed=seed,
                live_times=(300.0,),
            )
            result = fit_joint_peak_model(spectra, spec, calibration, resolution)
            if not result.success:
                continue
            index = result.line_names.index("middle_a")
            error = float(np.sqrt(result.line_rate_covariance[index, index]))
            covered += abs(result.line_rates_counts_per_s[index] - truth) <= 1.96 * error
            successful += 1
        self.assertGreaterEqual(successful, 10)
        self.assertGreaterEqual(covered / successful, 0.7)


if __name__ == "__main__":
    unittest.main()
