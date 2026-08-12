"""Manufactured tests for the paper-facing joint Poisson peak likelihood."""

from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

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
    profile_linear_ratio_interval,
    profile_ratio_interval,
    ratio_values_and_covariance,
    ratios_from_fit,
    stable_seed,
)
from scripts.reanalyze_paper_peak_statistics import (
    _bootstrap_provenance,
    _bootstrap_rows,
    _fit_diagnostics,
)
from src.public_data.run_estimands import aggregate_independent_run_rates


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
    middle_b_energy_keV: float = 310.0,
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
        LineComponent("middle_b", middle_b_energy_keV, "middle"),
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


def manufactured_quadratic_valley_case() -> tuple[
    tuple[PublicSpectrum, ...],
    JointPeakSpec,
    CalibrationConstraint,
    LinearResolution,
    dict[str, float],
]:
    """Return a deterministic positive quadratic with a negative middle term."""

    channel_count = 140
    channels = np.arange(1, channel_count + 1, dtype=np.float64)
    low_keV = 20.0
    high_keV = 120.0
    line_energy_keV = 70.0
    live_time = 4000.0
    truth = {
        "low": 0.60,
        "middle": -0.05,
        "high": 0.60,
        "line": 0.90,
        "anchor": 1.30,
    }
    fraction = (channels - low_keV) / (high_keV - low_keV)
    background_density = (
        truth["low"] * (1.0 - fraction) ** 2
        + 2.0 * truth["middle"] * fraction * (1.0 - fraction)
        + truth["high"] * fraction**2
    )
    line_probability = peak_shape_bin_probabilities(
        channels - 0.5,
        channels + 0.5,
        line_energy_keV,
        2.0,
        0.0,
        2.0,
    )
    anchor_probability = peak_shape_bin_probabilities(
        channels - 0.5,
        channels + 0.5,
        40.0,
        2.0,
        0.0,
        2.0,
    )
    counts = np.rint(
        live_time
        * (
            background_density
            + truth["line"] * line_probability
            + truth["anchor"] * anchor_probability
        )
    )
    spectrum = PublicSpectrum(
        file_id=1,
        run_id=1,
        file_name="manufactured_quadratic_valley",
        run_name="manufactured",
        live_time=live_time,
        calibration_A0=0.0,
        calibration_A1=1.0,
        counts=counts,
        energy_keV=channels,
        bin_width_keV=np.ones(channel_count),
        metadata={"manufactured": True},
    )
    spec = JointPeakSpec(
        "manufactured-quadratic-valley",
        (FitWindow("valley", low_keV, high_keV, "quadratic"),),
        (
            LineComponent("line", line_energy_keV, "valley"),
            LineComponent("anchor", 40.0, "valley"),
        ),
    )
    calibration = CalibrationConstraint(
        0.0,
        0.0,
        np.diag([1.0e-8, 1.0e-12]),
        (-1.0e-3, 1.0e-3),
        (-1.0e-6, 1.0e-6),
    )
    resolution = LinearResolution(
        2.0,
        0.0,
        0.0,
        2.0,
        tail_model="none",
        intercept_bounds_keV=(1.99, 2.01),
        slope_bounds=(0.0, 1.0e-6),
    )
    return (spectrum,), spec, calibration, resolution, truth


def optimizer_result(
    parameters,
    message: str,
    *,
    status: int = 0,
    success: bool = True,
    objective: float | None = None,
) -> SimpleNamespace:
    fields = {
        "x": np.asarray(parameters).copy(),
        "success": success,
        "status": status,
        "message": message,
        "nit": 0,
        "nfev": 1,
    }
    if objective is not None:
        fields["fun"] = objective
    return SimpleNamespace(**fields)


class PrimitiveAndCovarianceTests(unittest.TestCase):
    def test_quadratic_bernstein_exact_cone_matches_analytic_minimum(self):
        rng = np.random.default_rng(1807)
        cases = [
            (0.0, 0.0, 0.0),
            (0.0, 0.2, 0.7),
            (0.0, -1.0e-12, 0.7),
            (0.8, 0.0, 0.0),
            (0.8, -1.0e-12, 0.0),
        ]
        for _ in range(200):
            low, high = rng.uniform(0.0, 2.0, size=2)
            boundary = -np.sqrt(low * high)
            middle = boundary + rng.uniform(-0.5, 0.5)
            cases.append((low, middle, high))
        grid = np.linspace(0.0, 1.0, 20001)
        for low, middle, high in cases:
            with self.subTest(low=low, middle=middle, high=high):
                margin = likelihood.quadratic_bernstein_cone_margin(
                    low, middle, high
                )
                analytic_minimum = likelihood.quadratic_bernstein_minimum(
                    low, middle, high
                )
                sampled = (
                    low * (1.0 - grid) ** 2
                    + 2.0 * middle * grid * (1.0 - grid)
                    + high * grid**2
                )
                self.assertAlmostEqual(
                    analytic_minimum,
                    float(np.min(sampled)),
                    delta=2.0e-8,
                )
                self.assertEqual(margin >= 0.0, analytic_minimum >= 0.0)

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

    def test_relative_poisson_objective_matches_long_double_without_cancellation(self):
        observed = np.full(5000, 1_000_000_000.0, dtype=np.float64)
        expected = observed + np.linspace(-0.25, 0.25, observed.size)
        value = likelihood._poisson_nll_relative_to_saturated(
            observed, expected
        )
        observed_high = observed.astype(np.longdouble)
        expected_high = expected.astype(np.longdouble)
        difference_high = observed_high - expected_high
        reference = np.sum(
            -difference_high
            + observed_high
            * np.log1p(difference_high / expected_high),
            dtype=np.longdouble,
        )
        self.assertLess(abs(np.longdouble(value) - reference), 1.0e-12)
        cancellation_prone = likelihood.poisson_nll(
            observed, expected
        ) - likelihood.poisson_nll(observed, observed)
        self.assertGreater(value, 1.0e-9)
        self.assertGreater(abs(value - cancellation_prone), 1.0e-9)

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
            fisher_covariance_valid=False,
            active_bounds=(),
            quadratic_background_cone_diagnostics={
                "one": {
                    "low_counts_per_s_per_keV": 1.0,
                    "middle_counts_per_s_per_keV": -1.0,
                    "high_counts_per_s_per_keV": 1.0,
                    "cone_margin_counts_per_s_per_keV": 0.0,
                    "normalized_cone_margin": 0.0,
                    "minimum_density_counts_per_s_per_keV": 0.0,
                }
            },
            quadratic_background_exact_cone_valid=True,
            quadratic_background_cone_active_windows=("one",),
            quadratic_background_constrained_fallback_used=True,
            quadratic_background_cone_feasibility_relative_tolerance=1.0e-10,
            quadratic_background_cone_activity_relative_tolerance=1.0e-6,
        )
        diagnostics = _fit_diagnostics(
            result,
            {
                "chi_square_minimum_expected_counts_per_bin": 5.0,
                "per_bin_outlier_sigma_thresholds": [4.0, 5.0],
                "global_deviance_p_value_minimum": 0.001,
                "familywise_diagnostic_alpha": 0.01,
                "quadratic_background_cone_feasibility_relative_tolerance": 1.0e-10,
                "quadratic_background_cone_activity_relative_tolerance": 1.0e-6,
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
        self.assertTrue(
            diagnostics["quadratic_background_constrained_fallback_used"]
        )
        self.assertIn(
            "quadratic background exact cone is active and the fit is nonregular",
            diagnostics["applicability_reasons"],
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

    def _near_stationary_problem_and_candidate(self):
        initial = dict(
            zip(self.result.parameter_names, self.result.parameter_values)
        )
        problem = likelihood._prepare_problem(
            self.spectra,
            self.spec,
            self.calibration,
            self.resolution,
            initial,
            "shared_origin_scales",
        )
        candidate = problem.initial.copy()
        offset_index = problem.parameter_names.index("calibration.offset_keV")
        candidate[offset_index] += 1.0e-3 * problem.scales[offset_index]
        objective, projected_gradient = likelihood._stationarity_at(
            problem, candidate
        )
        self.assertGreater(
            projected_gradient, likelihood._STATIONARITY_TOLERANCE
        )
        return problem, candidate, objective

    def test_recovers_isolated_and_multiplet_rates_in_simultaneous_runs(self):
        self.assertTrue(self.result.success, self.result.message)
        self.assertTrue(self.result.optimizer_stationarity_valid)
        self.assertLessEqual(
            self.result.scaled_projected_gradient_inf_norm,
            self.result.stationarity_tolerance,
        )
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

    def test_fisher_covariance_inverts_scaled_coordinates_then_transforms_back(self):
        inverse_inputs: list[np.ndarray] = []
        original_inverse = np.linalg.inv

        def recording_inverse(matrix):
            inverse_inputs.append(np.asarray(matrix).copy())
            return original_inverse(matrix)

        with patch.object(np.linalg, "inv", side_effect=recording_inverse):
            result = fit_joint_peak_model(
                self.spectra,
                self.spec,
                self.calibration,
                self.resolution,
            )

        problem = likelihood._prepare_problem(
            self.spectra,
            self.spec,
            self.calibration,
            self.resolution,
            dict(zip(result.parameter_names, result.parameter_values)),
        )
        expected, jacobian = likelihood._model_and_jacobian(
            problem, result.parameter_values, with_jacobian=True
        )
        self.assertIsNotNone(jacobian)
        raw_fisher = jacobian.T @ (jacobian / expected[:, np.newaxis])
        raw_fisher[
            np.ix_(
                problem.prior_parameter_indices,
                problem.prior_parameter_indices,
            )
        ] += problem.prior_precision
        raw_fisher = (raw_fisher + raw_fisher.T) / 2.0
        scaled_fisher = (
            problem.scales[:, np.newaxis]
            * raw_fisher
            * problem.scales[np.newaxis, :]
        )
        full_size_inverse_inputs = [
            matrix
            for matrix in inverse_inputs
            if matrix.shape == scaled_fisher.shape
        ]
        self.assertEqual(len(full_size_inverse_inputs), 1)
        np.testing.assert_array_equal(
            full_size_inverse_inputs[0], scaled_fisher
        )
        scaled_covariance = original_inverse(scaled_fisher)
        expected_covariance = (
            problem.scales[:, np.newaxis]
            * scaled_covariance
            * problem.scales[np.newaxis, :]
        )
        expected_covariance = (
            expected_covariance + expected_covariance.T
        ) / 2.0
        np.testing.assert_allclose(
            result.covariance,
            expected_covariance,
            rtol=2.0e-12,
            atol=1.0e-14,
        )
        self.assertEqual(
            result.fisher_covariance_inversion_coordinates,
            "scaled_parameter_coordinates_then_transformed_to_physical",
        )

    def test_fit_result_warm_start_maps_only_shared_parameters_and_records_source(self):
        no_tail = replace(self.resolution, tail_model="none")
        result = fit_joint_peak_model(
            self.spectra,
            self.spec,
            self.calibration,
            no_tail,
            warm_start=self.result,
            warm_start_source="manufactured canonical fit",
        )
        self.assertTrue(result.success, result.message)
        configuration = result.optimizer_configuration
        self.assertEqual(
            configuration["initialization_source"],
            "manufactured canonical fit",
        )
        self.assertGreater(
            configuration["warm_start_shared_parameter_count"], 0
        )
        self.assertNotIn(
            "shape.low_energy_tail_fraction",
            configuration["warm_start_shared_parameter_names"],
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            fit_joint_peak_model(
                self.spectra,
                self.spec,
                self.calibration,
                no_tail,
                initial={"calibration.offset_keV": 0.0},
                warm_start=self.result,
            )

    def test_stationarity_gate_rejects_a_deliberately_stalled_solver(self):
        def stalled_minimize(function, initial, **_kwargs):
            value, gradient = function(initial)
            self.assertTrue(np.isfinite(value))
            self.assertTrue(np.isfinite(gradient).all())
            return optimizer_result(
                initial, "deliberately stalled", objective=value
            )

        with patch.object(likelihood, "minimize", side_effect=stalled_minimize):
            result = fit_joint_peak_model(
                self.spectra,
                self.spec,
                self.calibration,
                self.resolution,
            )
        self.assertTrue(result.optimizer_converged)
        self.assertFalse(result.optimizer_stationarity_valid)
        self.assertFalse(result.success)
        self.assertGreater(
            result.scaled_projected_gradient_inf_norm,
            result.stationarity_tolerance,
        )

    def test_scoring_repairs_residual_stationarity_monotonically_and_deterministically(self):
        problem, candidate, _ = self._near_stationary_problem_and_candidate()
        candidate_scaled = (
            candidate - problem.initial
        ) / problem.scales

        def stalled_near_solution(_function, _initial, **_kwargs):
            return optimizer_result(
                candidate_scaled,
                "deliberately left above stationarity gate",
            )

        with patch.object(
            likelihood, "minimize", side_effect=stalled_near_solution
        ):
            first = likelihood._optimize_problem(problem)
            second = likelihood._optimize_problem(problem)

        self.assertTrue(first.stationarity_valid)
        self.assertEqual(first.scoring_iterations, 1)
        self.assertLessEqual(
            first.stages[1]["stable_difference_identity_error"], 1.0e-9
        )
        self.assertLessEqual(
            first.scaled_projected_gradient_inf_norm,
            likelihood._STATIONARITY_TOLERANCE,
        )
        accepted_nlls = [
            stage["nll"] for stage in first.stages if stage["accepted"]
        ]
        self.assertTrue(
            all(
                later <= earlier
                for earlier, later in zip(accepted_nlls, accepted_nlls[1:])
            )
        )
        np.testing.assert_array_equal(first.parameters, second.parameters)
        self.assertEqual(first.stages, second.stages)

    def test_scoring_accepts_bounded_numerically_flat_stationarity_repair(self):
        problem, candidate, objective = (
            self._near_stationary_problem_and_candidate()
        )
        with patch.object(
            likelihood,
            "_penalized_nll_difference",
            return_value=(
                0.5 * likelihood._SCORING_NUMERICALLY_FLAT_NLL_TOLERANCE
            ),
        ):
            scoring = likelihood._fisher_scoring_step(
                problem, candidate, objective
            )
        self.assertTrue(scoring.accepted)
        self.assertLessEqual(
            scoring.projected_gradient, likelihood._STATIONARITY_TOLERANCE
        )
        self.assertIn("numerically flat", scoring.message)

    def test_fixed_reference_objective_matches_stable_total_at_displaced_points(self):
        problem, _, _ = self._near_stationary_problem_and_candidate()
        reference = problem.initial.copy()
        reference_nll, _, reference_expected = (
            likelihood._objective_gradient_and_expected(problem, reference)
        )
        displacements = (
            {"calibration.offset_keV": 0.15},
            {"calibration.fractional_gain_stretch": -0.1},
            {
                "line.weak.rate_counts_per_s": 0.2,
                "background.middle.high_counts_per_s_per_keV": 0.1,
            },
        )
        for displacement in displacements:
            with self.subTest(displacement=displacement):
                candidate = reference.copy()
                for name, scaled_step in displacement.items():
                    index = problem.parameter_names.index(name)
                    candidate[index] += scaled_step * problem.scales[index]
                candidate_nll, candidate_gradient, candidate_expected = (
                    likelihood._objective_gradient_and_expected(
                        problem, candidate
                    )
                )
                delta = likelihood._penalized_nll_difference(
                    problem,
                    reference,
                    candidate,
                    reference_expected,
                    candidate_expected,
                )
                self.assertLessEqual(
                    abs(delta - (candidate_nll - reference_nll)), 1.0e-9
                )
                _, direct_gradient = likelihood._objective_and_gradient(
                    problem, candidate
                )
                np.testing.assert_array_equal(
                    candidate_gradient, direct_gradient
                )

    def test_polish_basin_guard_failure_fails_closed(self):
        problem, candidate, _ = self._near_stationary_problem_and_candidate()
        candidate_scaled = (
            candidate - problem.initial
        ) / problem.scales

        def stalled_near_solution(_function, _initial, **_kwargs):
            return optimizer_result(
                candidate_scaled,
                "deliberately outside declared polish basin",
            )

        with patch.object(
            likelihood, "minimize", side_effect=stalled_near_solution
        ), patch.object(
            likelihood,
            "_polish_basin_metrics",
            return_value=(11.0, 0.1, False),
        ):
            outcome = likelihood._optimize_problem(problem)
        self.assertFalse(outcome.polish_basin_valid)
        self.assertFalse(outcome.stationarity_valid)
        self.assertEqual(outcome.scoring_iterations, 0)
        self.assertFalse(outcome.stages[-1]["accepted"])

    def test_guarded_full_chain_restart_validates_and_is_deterministic(self):
        problem, candidate, _ = self._near_stationary_problem_and_candidate()
        candidate_scaled = (
            candidate - problem.initial
        ) / problem.scales

        def run_once():
            call_count = 0

            def staged_minimize(_function, initial, **_kwargs):
                nonlocal call_count
                call_count += 1
                returned = (
                    candidate_scaled.copy()
                    if call_count == 1
                    else np.zeros_like(initial)
                )
                return optimizer_result(
                    returned,
                    f"manufactured chain call {call_count}",
                    status=17 if call_count == 2 else 0,
                )

            with patch.object(
                likelihood, "minimize", side_effect=staged_minimize
            ), patch.object(
                likelihood,
                "_polish_basin_metrics",
                return_value=(11.0, 0.1, False),
            ):
                return likelihood._optimize_problem(
                    problem,
                    allow_basin_restart=True,
                    basin_restart_source="manufactured deterministic candidate",
                )

        first = run_once()
        second = run_once()
        self.assertTrue(first.stationarity_valid)
        self.assertTrue(first.basin_restart_used)
        self.assertTrue(first.polish_basin_valid)
        self.assertEqual(
            [stage["optimizer_chain"] for stage in first.stages],
            [1, 1, "restart", 2],
        )
        self.assertEqual(first.stages[2]["status"], 17)
        self.assertEqual(
            first.stages[2]["solver_success"],
            first.stages[1]["solver_success"],
        )
        np.testing.assert_array_equal(first.parameters, second.parameters)
        self.assertEqual(first.stages, second.stages)

    def test_guarded_restart_chain_failure_still_fails_closed(self):
        problem, candidate, _ = self._near_stationary_problem_and_candidate()
        candidate_scaled = (
            candidate - problem.initial
        ) / problem.scales
        call_count = 0

        def stalled_minimize(_function, initial, **_kwargs):
            nonlocal call_count
            call_count += 1
            return optimizer_result(
                candidate_scaled,
                f"manufactured stalled restart call {call_count}",
            )

        with patch.object(
            likelihood, "minimize", side_effect=stalled_minimize
        ), patch.object(
            likelihood,
            "_polish_basin_metrics",
            side_effect=(
                (11.0, 0.1, False),
                (0.0, 0.0, True),
                (0.0, 0.0, True),
            ),
        ), patch.object(
            likelihood,
            "cho_factor",
            side_effect=np.linalg.LinAlgError("deliberate restart failure"),
        ):
            outcome = likelihood._optimize_problem(
                problem,
                allow_basin_restart=True,
                basin_restart_source="manufactured failing candidate",
            )
        self.assertTrue(outcome.basin_restart_used)
        self.assertFalse(outcome.stationarity_valid)

    def test_scoring_cholesky_and_fixed_damping_failure_fails_closed(self):
        problem, candidate, objective = (
            self._near_stationary_problem_and_candidate()
        )
        with patch.object(
            likelihood,
            "cho_factor",
            side_effect=np.linalg.LinAlgError("deliberate factorization failure"),
        ) as factor:
            scoring = likelihood._fisher_scoring_step(
                problem, candidate, objective
            )
        self.assertFalse(scoring.accepted)
        self.assertEqual(
            factor.call_count, len(likelihood._SCORING_DAMPING_LADDER)
        )
        self.assertIn("fixed damping ladder", scoring.message)
        np.testing.assert_array_equal(scoring.parameters, candidate)

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

    def test_overlapping_declared_components_retain_anticorrelation(self):
        spectra, spec, calibration, resolution, _ = manufactured_case(
            seed=771,
            live_times=(1800.0,),
            middle_b_energy_keV=302.0,
            poisson=False,
        )
        result = fit_joint_peak_model(spectra, spec, calibration, resolution)
        self.assertTrue(result.success, result.message)
        first = result.line_names.index("middle_a")
        second = result.line_names.index("middle_b")
        covariance = result.line_rate_covariance[first, second]
        correlation = covariance / np.sqrt(
            result.line_rate_covariance[first, first]
            * result.line_rate_covariance[second, second]
        )
        self.assertLess(correlation, -0.2)

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
        self.assertEqual(
            first.gaussian_pseudo_observation_seed,
            second.gaussian_pseudo_observation_seed,
        )
        self.assertNotEqual(
            first.seed, first.gaussian_pseudo_observation_seed
        )
        self.assertEqual(first.successful_replicates, 5)
        np.testing.assert_allclose(
            first.empirical_standard_deviation,
            second.empirical_standard_deviation,
        )
        self.assertTrue(np.all((first.fisher_95_percent_coverage >= 0.0)))
        self.assertTrue(np.all((first.fisher_95_percent_coverage <= 1.0)))
        self.assertEqual(
            first.quadratic_background_constrained_fallback_replicates, 0
        )
        self.assertEqual(
            first.quadratic_background_exact_cone_invalid_replicates, 0
        )
        self.assertEqual(first.quadratic_background_cone_active_replicates, 0)
        self.assertTrue(
            np.isnan(
                first.minimum_quadratic_background_normalized_cone_margin
            )
        )
        self.assertEqual(
            first.gaussian_pseudo_observation_parameter_names,
            (
                "calibration.offset_keV",
                "calibration.fractional_gain_stretch",
            ),
        )
        np.testing.assert_allclose(
            first.gaussian_pseudo_observation_generating_values,
            self.result.parameter_values[:2],
        )
        np.testing.assert_allclose(
            first.gaussian_pseudo_observation_covariance,
            self.calibration.covariance,
        )
        provenance = _bootstrap_provenance(first)
        self.assertEqual(
            provenance["gaussian_pseudo_observation_parameter_names"],
            list(first.gaussian_pseudo_observation_parameter_names),
        )
        np.testing.assert_allclose(
            provenance["gaussian_pseudo_observation_covariance"],
            self.calibration.covariance,
        )
        rows = _bootstrap_rows(first, 200)
        self.assertTrue(rows)
        self.assertEqual(rows[0]["poisson_count_seed"], first.seed)
        self.assertEqual(
            rows[0]["gaussian_pseudo_observation_seed"],
            first.gaussian_pseudo_observation_seed,
        )
        self.assertIn(
            "draw the complete Gaussian constraint pseudo-observation",
            rows[0]["semantics"],
        )

    def test_bootstrap_redraws_every_gaussian_pseudo_observation(self):
        spectra, spec, calibration, resolution, _ = manufactured_case(
            seed=5150,
            live_times=(700.0, 650.0),
            calibration_offsets_keV=(0.0, 0.2),
            resolution_scales=(1.0, 1.1),
        )
        result = fit_joint_peak_model(
            spectra, spec, calibration, resolution
        )
        self.assertTrue(result.success, result.message)
        initial = dict(zip(result.parameter_names, result.parameter_values))
        problem = likelihood._prepare_problem(
            spectra,
            spec,
            calibration,
            resolution,
            initial,
        )
        prior_names = tuple(
            problem.parameter_names[index]
            for index in problem.prior_parameter_indices
        )
        self.assertEqual(
            prior_names,
            (
                "calibration.offset_keV",
                "calibration.fractional_gain_stretch",
                "spectrum.1.calibration_offset_deviation_keV",
                "spectrum.1.fractional_gain_stretch_deviation",
                "spectrum.1.resolution_scale_relative_to_spectrum_0",
            ),
        )
        with patch.object(
            likelihood, "fit_joint_peak_model", return_value=result
        ) as mocked_fit:
            summary = parametric_bootstrap(
                spectra,
                spec,
                calibration,
                resolution,
                result,
                case_identity="bootstrap-pseudo-observation-draws",
                replicates=3,
            )
        expected_covariance = np.linalg.inv(problem.prior_precision)
        expected_generating = result.parameter_values[
            problem.prior_parameter_indices
        ]
        np.testing.assert_allclose(
            summary.gaussian_pseudo_observation_covariance,
            expected_covariance,
        )
        np.testing.assert_allclose(
            summary.gaussian_pseudo_observation_generating_values,
            expected_generating,
        )
        self.assertEqual(
            summary.gaussian_pseudo_observation_parameter_names,
            prior_names,
        )
        gaussian_rng = np.random.default_rng(
            summary.gaussian_pseudo_observation_seed
        )
        poisson_rng = np.random.default_rng(summary.seed)
        cholesky = np.linalg.cholesky(expected_covariance)
        for call in mocked_fit.call_args_list:
            expected_counts = poisson_rng.poisson(result.expected_counts)
            expected_draw = expected_generating + cholesky @ (
                gaussian_rng.standard_normal(len(prior_names))
            )
            replica_spectra = call.args[0]
            for observation_index, expected_count in enumerate(
                expected_counts
            ):
                run_index = int(
                    result.observation_spectrum_indices[observation_index]
                )
                channel_index = int(
                    result.observation_channel_indices[observation_index]
                )
                self.assertEqual(
                    replica_spectra[run_index].counts[channel_index],
                    expected_count,
                )
            observed = call.kwargs["gaussian_constraint_observations"]
            self.assertEqual(tuple(observed), prior_names)
            np.testing.assert_allclose(
                [observed[name] for name in prior_names], expected_draw
            )

    def test_gaussian_constraint_override_is_exact_and_fail_closed(self):
        spectra, spec, calibration, resolution, _ = manufactured_case(
            seed=5150,
            live_times=(700.0, 650.0),
            calibration_offsets_keV=(0.0, 0.2),
            resolution_scales=(1.0, 1.1),
            poisson=False,
        )
        base = likelihood._prepare_problem(
            spectra, spec, calibration, resolution, None
        )
        names = tuple(
            base.parameter_names[index]
            for index in base.prior_parameter_indices
        )
        observations = {
            name: float(base.prior_mean[index] + (index + 1) * 1e-6)
            for index, name in enumerate(names)
        }
        overridden = likelihood._prepare_problem(
            spectra,
            spec,
            calibration,
            resolution,
            None,
            gaussian_constraint_observations=observations,
        )
        np.testing.assert_allclose(
            overridden.prior_mean,
            [observations[name] for name in names],
        )
        np.testing.assert_allclose(
            overridden.prior_precision, base.prior_precision
        )
        missing = dict(observations)
        missing.pop(names[-1])
        with self.assertRaisesRegex(ValueError, "missing:"):
            likelihood._prepare_problem(
                spectra,
                spec,
                calibration,
                resolution,
                None,
                gaussian_constraint_observations=missing,
            )
        unknown = dict(observations)
        unknown["not.a.constrained.nuisance"] = 0.0
        with self.assertRaisesRegex(ValueError, "unknown:"):
            likelihood._prepare_problem(
                spectra,
                spec,
                calibration,
                resolution,
                None,
                gaussian_constraint_observations=unknown,
            )

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

    def test_inactive_exact_quadratic_cone_matches_legacy_coefficient_box(self):
        quadratic_spec = replace(
            self.spec,
            name="manufactured-inactive-quadratic-cone",
            windows=tuple(
                replace(window, background_model="quadratic")
                for window in self.spec.windows
            ),
        )
        exact = fit_joint_peak_model(
            self.spectra,
            quadratic_spec,
            self.calibration,
            self.resolution,
        )
        with patch.object(
            likelihood, "_QUADRATIC_MIDDLE_LOWER_BOUND", 1.0e-15
        ):
            legacy_box = fit_joint_peak_model(
                self.spectra,
                quadratic_spec,
                self.calibration,
                self.resolution,
            )
        self.assertTrue(exact.success, exact.message)
        self.assertTrue(legacy_box.success, legacy_box.message)
        self.assertFalse(exact.quadratic_background_constrained_fallback_used)
        self.assertFalse(exact.quadratic_background_cone_active_windows)
        np.testing.assert_allclose(
            exact.parameter_values,
            legacy_box.parameter_values,
            rtol=5.0e-6,
            atol=2.0e-9,
        )
        self.assertAlmostEqual(
            exact.penalized_nll, legacy_box.penalized_nll, places=9
        )

    def test_negative_middle_quadratic_valley_is_recovered_and_old_box_fails(self):
        spectra, spec, calibration, resolution, truth = (
            manufactured_quadratic_valley_case()
        )
        exact = fit_joint_peak_model(spectra, spec, calibration, resolution)
        with patch.object(
            likelihood, "_QUADRATIC_MIDDLE_LOWER_BOUND", 1.0e-15
        ):
            legacy_box = fit_joint_peak_model(
                spectra, spec, calibration, resolution
            )
        middle_name = "background.valley.middle_counts_per_s_per_keV"
        self.assertTrue(exact.success, exact.message)
        self.assertTrue(exact.quadratic_background_exact_cone_valid)
        self.assertFalse(exact.quadratic_background_cone_active_windows)
        self.assertTrue(exact.fisher_covariance_valid)
        self.assertGreater(
            exact.line_rate_covariance[0, 0], 0.0
        )
        self.assertAlmostEqual(
            exact.parameter(middle_name), truth["middle"], delta=2.0e-3
        )
        self.assertAlmostEqual(
            exact.line_rate("line"), truth["line"], delta=2.0e-3
        )
        self.assertGreaterEqual(
            exact.quadratic_background_cone_diagnostics["valley"][
                "minimum_density_counts_per_s_per_keV"
            ],
            0.0,
        )
        self.assertGreaterEqual(legacy_box.parameter(middle_name), 0.0)
        self.assertGreater(
            abs(legacy_box.line_rate("line") - truth["line"]), 0.05
        )
        self.assertGreater(
            legacy_box.poisson_deviance - exact.poisson_deviance, 50.0
        )

        problem = likelihood._prepare_problem(
            spectra,
            spec,
            calibration,
            resolution,
            dict(zip(exact.parameter_names, exact.parameter_values)),
        )
        parameters = problem.initial.copy()
        middle_index = problem.parameter_names.index(middle_name)
        _, analytic = likelihood._objective_and_gradient(problem, parameters)
        step = problem.scales[middle_index] * 2.0e-6
        plus = parameters.copy()
        minus = parameters.copy()
        plus[middle_index] += step
        minus[middle_index] -= step
        numerical = (
            likelihood._objective_and_gradient(problem, plus)[0]
            - likelihood._objective_and_gradient(problem, minus)[0]
        ) / (2.0 * step)
        self.assertAlmostEqual(
            analytic[middle_index],
            numerical,
            delta=max(2.0e-3, 2.0e-3 * abs(numerical)),
        )

    def test_violated_relaxed_quadratic_cone_invokes_stationary_fallback(self):
        spectra, spec, calibration, resolution, _ = (
            manufactured_quadratic_valley_case()
        )
        problem = likelihood._prepare_problem(
            spectra, spec, calibration, resolution, None
        )
        relaxed = likelihood._optimize_problem(problem)
        violated = relaxed.parameters.copy()
        low_index, middle_index, high_index = (
            problem.background_parameter_indices["valley"]
        )
        violated[middle_index] = -1.1 * np.sqrt(
            violated[low_index] * violated[high_index]
        )
        self.assertLess(
            likelihood.quadratic_bernstein_cone_margin(
                violated[low_index],
                violated[middle_index],
                violated[high_index],
            ),
            0.0,
        )
        constrained = likelihood._exact_cone_fallback(
            problem, replace(relaxed, parameters=violated)
        )
        diagnostics = likelihood._quadratic_background_cone_diagnostics(
            problem, constrained.parameters
        )
        self.assertTrue(constrained.solver_converged)
        self.assertTrue(constrained.stationarity_valid)
        self.assertLessEqual(
            constrained.scaled_projected_gradient_inf_norm,
            likelihood._STATIONARITY_TOLERANCE,
        )
        self.assertGreaterEqual(
            diagnostics["valley"]["normalized_cone_margin"],
            -likelihood._EXACT_CONE_FEASIBILITY_RELATIVE_TOLERANCE,
        )

    def test_active_exact_quadratic_cone_is_nonregular_and_invalidates_fisher(self):
        spectra, spec, calibration, resolution, truth = (
            manufactured_quadratic_valley_case()
        )
        spectrum = spectra[0]
        fraction = (spectrum.energy_keV - 20.0) / 100.0
        boundary_middle = -np.sqrt(truth["low"] * truth["high"])
        background_density = (
            truth["low"] * (1.0 - fraction) ** 2
            + 2.0 * boundary_middle * fraction * (1.0 - fraction)
            + truth["high"] * fraction**2
        )
        line_probability = peak_shape_bin_probabilities(
            spectrum.energy_keV - 0.5,
            spectrum.energy_keV + 0.5,
            70.0,
            2.0,
            0.0,
            2.0,
        )
        anchor_probability = peak_shape_bin_probabilities(
            spectrum.energy_keV - 0.5,
            spectrum.energy_keV + 0.5,
            40.0,
            2.0,
            0.0,
            2.0,
        )
        boundary_spectrum = replace(
            spectrum,
            counts=np.rint(
                spectrum.live_time
                * (
                    background_density
                    + truth["line"] * line_probability
                    + truth["anchor"] * anchor_probability
                )
            ),
        )
        result = fit_joint_peak_model(
            (boundary_spectrum,),
            spec,
            calibration,
            resolution,
            initial={
                "line.line.rate_counts_per_s": truth["line"],
                "line.anchor.rate_counts_per_s": truth["anchor"],
                "background.valley.low_counts_per_s_per_keV": truth["low"],
                "background.valley.middle_counts_per_s_per_keV": (
                    boundary_middle
                ),
                "background.valley.high_counts_per_s_per_keV": truth["high"],
            },
        )
        self.assertTrue(result.success, result.message)
        self.assertTrue(
            result.quadratic_background_constrained_fallback_used
        )
        self.assertEqual(
            result.quadratic_background_cone_active_windows, ("valley",)
        )
        self.assertFalse(result.fisher_covariance_valid)
        self.assertEqual(
            result.quadratic_background_cone_diagnostics["valley"][
                "normalized_cone_margin"
            ],
            0.0,
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

    def test_independent_yield_model_returns_one_rate_per_run_and_component(self):
        spectra, spec, calibration, resolution, truth = manufactured_case(
            seed=314,
            live_times=(1800.0, 1700.0),
            signal_scales=(1.0, 1.65),
            poisson=False,
        )
        result = fit_joint_peak_model(
            spectra,
            spec,
            calibration,
            resolution,
            yield_model="independent_runs",
        )
        self.assertTrue(result.success, result.message)
        self.assertEqual(len(result.line_names), 2 * len(spec.components))
        self.assertNotIn("spectrum.1.signal_scale_relative_to_spectrum_0", result.parameter_names)
        first = result.line_rate("spectrum.0.reference")
        second = result.line_rate("spectrum.1.reference")
        self.assertAlmostEqual(first, truth["reference"], delta=0.08)
        self.assertAlmostEqual(second, 1.65 * truth["reference"], delta=0.12)

    def test_sum_then_fit_equivalence_requires_identical_run_response(self):
        def summed_and_aggregate(**case_options):
            spectra, spec, calibration, resolution, _ = manufactured_case(
                poisson=False, **case_options
            )
            independent = fit_joint_peak_model(
                spectra,
                spec,
                calibration,
                resolution,
                yield_model="independent_runs",
            )
            aggregate = aggregate_independent_run_rates(
                [component.name for component in spec.components],
                [spectrum.live_time for spectrum in spectra],
                independent.line_rates_counts_per_s.reshape((2, -1)),
                independent.line_rate_covariance,
            )
            combined = replace(
                spectra[0],
                live_time=sum(spectrum.live_time for spectrum in spectra),
                counts=np.sum(
                    [spectrum.counts for spectrum in spectra], axis=0
                ),
            )
            return (
                spec,
                aggregate,
                fit_joint_peak_model(
                    (combined,), spec, calibration, resolution
                ),
            )

        _, aggregate, summed_fit = summed_and_aggregate(
            live_times=(700.0, 1300.0)
        )
        np.testing.assert_allclose(
            summed_fit.line_rates_counts_per_s,
            aggregate.aggregate_rates_counts_per_s,
            rtol=2e-3,
            atol=2e-4,
        )

        spec, aggregate, summed_fit = summed_and_aggregate(
            live_times=(900.0, 1100.0),
            calibration_offsets_keV=(0.0, 0.4),
            resolution_scales=(1.0, 1.3),
        )
        reference_index = [
            component.name for component in spec.components
        ].index("reference")
        relative_difference = abs(
            summed_fit.line_rates_counts_per_s[reference_index]
            / aggregate.aggregate_rates_counts_per_s[reference_index]
            - 1.0
        )
        self.assertGreater(relative_difference, 0.004)


class BoundaryAndCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        boundary = manufactured_case(
            seed=71, weak_rate=0.0, poisson=False
        )
        cls.boundary_case = boundary[:4]
        cls.boundary_result = fit_joint_peak_model(*cls.boundary_case)

        linear = manufactured_case(
            seed=82, live_times=(800.0,), poisson=False
        )
        cls.linear_case = linear[:4]
        cls.linear_result = fit_joint_peak_model(
            *cls.linear_case, yield_model="independent_runs"
        )

    def assert_profile_diagnostics(self, interval, *, linear=False):
        self.assertEqual(interval.inner_solver_failures, 0)
        self.assertLessEqual(
            abs(interval.base_nll_difference),
            interval.base_nll_consistency_tolerance,
        )
        if linear:
            self.assertLessEqual(
                interval.maximum_scaled_kkt_inf_norm,
                interval.inner_stationarity_tolerance,
            )
            self.assertLessEqual(
                interval.maximum_linear_constraint_relative_residual,
                interval.linear_constraint_relative_tolerance,
            )
        self.assertLessEqual(
            interval.maximum_stable_nll_difference_identity_error,
            interval.stable_nll_difference_identity_tolerance,
        )
        self.assertEqual(interval.exact_cone_invalid_inner_solves, 0)

    def test_linear_combination_ratio_profile_for_aggregate_estimand(self):
        spectra, spec, calibration, resolution = self.linear_case
        result = self.linear_result
        with self.assertRaisesRegex(
            ValueError,
            "shared_origin_scales.*profile_linear_ratio_interval",
        ):
            profile_ratio_interval(
                spectra,
                spec,
                calibration,
                resolution,
                result,
                RatioDefinition(
                    "middle/reference", "middle_a", "reference"
                ),
            )
        numerator = np.zeros(len(result.line_names))
        denominator = np.zeros(len(result.line_names))
        numerator[result.line_names.index("spectrum.0.middle_a")] = 1.0
        denominator[result.line_names.index("spectrum.0.reference")] = 1.0
        interval = profile_linear_ratio_interval(
            spectra,
            spec,
            calibration,
            resolution,
            result,
            numerator,
            denominator,
            ratio_name="aggregate_middle/reference",
            yield_model="independent_runs",
            max_evaluations=24,
        )
        self.assertEqual(interval.kind, "two_sided")
        self.assertLess(interval.lower, interval.estimate)
        self.assertGreater(interval.upper, interval.estimate)
        self.assert_profile_diagnostics(interval, linear=True)

    def test_absent_line_gets_profile_upper_limit(self):
        spectra, spec, calibration, resolution = self.boundary_case
        result = self.boundary_result
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
        self.assert_profile_diagnostics(interval)

    def test_profile_inner_failures_propagate_for_both_public_apis(self):
        spectra, spec, calibration, resolution = self.boundary_case
        result = self.boundary_result
        original = likelihood._profile_nll_at_ratio
        call_count = 0

        def fail_after_base(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            outcome = original(*args, **kwargs)
            if call_count == 2:
                return replace(
                    outcome,
                    success=False,
                    solver_converged=False,
                    message="deliberate inner failure",
                )
            return outcome

        with patch.object(
            likelihood,
            "_profile_nll_at_ratio",
            side_effect=fail_after_base,
        ):
            interval = profile_ratio_interval(
                spectra,
                spec,
                calibration,
                resolution,
                result,
                RatioDefinition(
                    "weak/reference", "weak", "reference"
                ),
                max_evaluations=30,
            )
        self.assertEqual(interval.kind, "failed")
        self.assertGreaterEqual(interval.inner_solver_failures, 1)
        self.assertIn("deliberate inner failure", interval.message)

        spectra, spec, calibration, resolution = self.linear_case
        result = self.linear_result
        numerator = np.zeros(len(result.line_names))
        denominator = np.zeros(len(result.line_names))
        numerator[result.line_names.index("spectrum.0.middle_a")] = 1.0
        denominator[result.line_names.index("spectrum.0.reference")] = 1.0
        original = likelihood._profile_nll_at_linear_ratio
        call_count = 0

        def fail_linear_after_base(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            outcome = original(*args, **kwargs)
            if call_count == 2:
                return replace(
                    outcome,
                    success=False,
                    stationarity_valid=False,
                    message="deliberate linear inner failure",
                )
            return outcome

        with patch.object(
            likelihood,
            "_profile_nll_at_linear_ratio",
            side_effect=fail_linear_after_base,
        ):
            interval = profile_linear_ratio_interval(
                spectra,
                spec,
                calibration,
                resolution,
                result,
                numerator,
                denominator,
                ratio_name="aggregate_middle/reference",
                yield_model="independent_runs",
                max_evaluations=24,
            )
        self.assertEqual(interval.kind, "failed")
        self.assertGreaterEqual(interval.inner_solver_failures, 1)
        self.assertIn("deliberate linear inner failure", interval.message)

    def test_profile_base_nll_mismatch_fails_closed(self):
        spectra, spec, calibration, resolution = self.boundary_case
        result = self.boundary_result
        original = likelihood._profile_nll_at_ratio

        def inconsistent_base(*args, **kwargs):
            outcome = original(*args, **kwargs)
            return replace(outcome, nll=outcome.nll + 0.006)

        with patch.object(
            likelihood,
            "_profile_nll_at_ratio",
            side_effect=inconsistent_base,
        ):
            interval = profile_ratio_interval(
                spectra,
                spec,
                calibration,
                resolution,
                result,
                RatioDefinition(
                    "weak/reference", "weak", "reference"
                ),
                base_nll_consistency_tolerance=0.005,
            )
        self.assertEqual(interval.kind, "failed")
        self.assertGreater(abs(interval.base_nll_difference), 0.005)
        self.assertIn("inconsistent", interval.message)


    def test_profile_inner_solve_inherits_exact_negative_middle_cone(self):
        spectra, spec, calibration, resolution, _ = (
            manufactured_quadratic_valley_case()
        )
        result = fit_joint_peak_model(spectra, spec, calibration, resolution)
        problem = likelihood._prepare_problem(
            spectra,
            spec,
            calibration,
            resolution,
            dict(zip(result.parameter_names, result.parameter_values)),
        )
        numerator_index = problem.component_parameter_indices["line"]
        denominator_index = problem.component_parameter_indices["anchor"]
        ratio = (
            result.parameter_values[numerator_index]
            / result.parameter_values[denominator_index]
        )
        outcome = likelihood._profile_nll_at_ratio(
            problem,
            result.parameter_values,
            numerator_index,
            denominator_index,
            ratio,
        )
        self.assertTrue(outcome.success, outcome.message)
        self.assertTrue(outcome.exact_cone_valid)
        self.assertLessEqual(
            outcome.scaled_kkt_inf_norm,
            likelihood._PROFILE_STATIONARITY_TOLERANCE,
        )

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
