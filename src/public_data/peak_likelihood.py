"""Declarative Poisson likelihoods for paper-facing HPGe peak yields.

The model consumes raw, unrebinned integer detector-channel counts.  Every
line shape is normalized on the real line and integrated over calibrated bin
edges, so a fitted line-rate parameter has units of detector counts/s.  The
implementation is ROOT-free and does not read or write SQLite itself.

Several spectra may be fit simultaneously.  The base declared
calibration/resolution curve is shared.  Line rates may either factor through
declared origin-class run scales or be independently estimated for every
run/component.  Optional Gaussian-constrained
per-run calibration deviations and resolution scales represent declared
drift.  Free signal scales for every declared physical-origin class and a
background scale for every run after the first absorb run-to-run intensity
changes.  A single origin class preserves the former common-scale model.  Each declared window has
its own background rate-density shape, shared across consecutive
same-configuration runs.  Thus the fit does not add spectra and retains
signal/background/reference correlations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from math import isfinite
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr
from scipy.stats import chi2

from src.public_data.browser import PublicSpectrum


ComponentRole = Literal["line", "fep", "sep", "dep", "contaminant"]
BackgroundModel = Literal["affine", "quadratic"]
ResolutionForm = Literal["linear", "sqrt"]
TailModel = Literal["none", "constant"]
YieldModel = Literal["shared_origin_scales", "independent_runs"]
_SQRT_2PI = np.sqrt(2.0 * np.pi)
_STATIONARITY_TOLERANCE = 1.0e-3
_ACTIVE_BOUND_TOLERANCE = 1.0e-6
_LBFGSB_OPTIONS = {
    "maxiter": 8000,
    "maxfun": 200000,
    "ftol": 1.0e-13,
    "gtol": 1.0e-5,
    "maxls": 100,
    "maxcor": 30,
}
_SLSQP_POLISH_OPTIONS = {"maxiter": 5000, "ftol": 1.0e-12}
_SCORING_MAX_ITERATIONS = 8
_SCORING_DAMPING_LADDER = (0.0, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4)
_SCORING_ARMIJO_COEFFICIENT = 1.0e-4
_SCORING_MAX_BACKTRACKS = 30
_SCORING_ZERO_STEP_TOLERANCE = 1.0e-14
_SCORING_NUMERICALLY_FLAT_NLL_TOLERANCE = 1.0e-9
_SCORING_NUMERICALLY_FLAT_GRADIENT_REDUCTION_FACTOR = 0.5
_POLISH_MAX_SCALED_DISPLACEMENT_INF = 10.0
_POLISH_MAX_STABLE_NLL_DECREASE = 1.0
_NLL_DIFFERENCE_IDENTITY_TOLERANCE = 1.0e-9
_QUADRATIC_MIDDLE_LOWER_BOUND = -np.inf
_EXACT_CONE_FEASIBILITY_RELATIVE_TOLERANCE = 1.0e-10
_EXACT_CONE_ACTIVITY_RELATIVE_TOLERANCE = 1.0e-6
_EXACT_CONE_SLSQP_OPTIONS = {"maxiter": 8000, "ftol": 1.0e-13}
_PROFILE_SLSQP_OPTIONS = {"maxiter": 3000, "ftol": 1.0e-12}
_PROFILE_SLSQP_MAX_CHAINS = 3
_PROFILE_STATIONARITY_TOLERANCE = 1.0e-3
_PROFILE_SCORING_MAX_ITERATIONS = 8
_PROFILE_LINEAR_CONSTRAINT_RELATIVE_TOLERANCE = 1.0e-8
_PROFILE_BASE_NLL_CONSISTENCY_TOLERANCE = 5.0e-3
_BOOTSTRAP_GAUSSIAN_PSEUDO_OBSERVATION_SEED_DOMAIN = (
    "gaussian-pseudo-observations-v1"
)
_BOOTSTRAP_GAUSSIAN_PSEUDO_OBSERVATION_CONVENTION = (
    "for every replica, independently draw the complete Gaussian "
    "constraint pseudo-observation from N(fitted generating nuisances, "
    "declared covariance) and refit with that draw"
)


@dataclass(frozen=True)
class FitWindow:
    """A predeclared half-open energy interval in keV."""

    name: str
    low_keV: float
    high_keV: float
    background_model: BackgroundModel = "affine"


@dataclass(frozen=True)
class LineComponent:
    """One normalized Gaussian component in a named fit window."""

    name: str
    energy_keV: float
    window: str
    role: ComponentRole = "line"
    parent: str = ""
    origin_class: str = "shared"


@dataclass(frozen=True)
class JointPeakSpec:
    """Stable fit identity, disjoint windows, and fixed component membership."""

    name: str
    windows: tuple[FitWindow, ...]
    components: tuple[LineComponent, ...]


@dataclass(frozen=True)
class CalibrationConstraint:
    """Gaussian constraint on a common affine calibration and optional curvature.

    Before optional per-run deviations, calibrated edges are

    ``A0_r + offset_keV + A1_r * (1 + stretch) * channel_edge``.

    When ``curvature_mean_keV`` is declared, a single zero-centered quadratic
    correction is added.  Its fixed basis is
    ``((E_nominal - curvature_pivot_keV) / curvature_scale_keV)**2``;
    therefore the coefficient is the correction in keV one declared scale
    from the pivot.  The affine terms remain free, so this adds only curvature,
    not a separately selected local centroid shift.

    When ``per_run_deviation_covariance`` is declared, spectra after index zero
    receive independent zero-centered offset/stretch deviations.  Spectrum
    zero fixes the relative convention.
    """

    offset_mean_keV: float
    stretch_mean: float
    covariance: np.ndarray
    offset_bounds_keV: tuple[float, float] = (-2.0, 2.0)
    stretch_bounds: tuple[float, float] = (-5e-4, 5e-4)
    per_run_deviation_covariance: np.ndarray | None = None
    per_run_offset_bounds_keV: tuple[float, float] = (-1.0, 1.0)
    per_run_stretch_bounds: tuple[float, float] = (-2e-4, 2e-4)
    curvature_mean_keV: float | None = None
    curvature_sigma_keV: float | None = None
    curvature_bounds_keV: tuple[float, float] = (-2.0, 2.0)
    curvature_pivot_keV: float = 0.0
    curvature_scale_keV: float = 1.0


@dataclass(frozen=True)
class DetectorResolution:
    """Shared resolution and optional normalized low-energy-tail model.

    ``form="linear"`` uses ``sigma(E) = intercept_keV + slope * E_keV``.
    ``form="sqrt"`` uses the HPGe-motivated
    ``sigma(E) = sqrt(intercept_keV**2 + slope * E_keV)``.  When
    ``tail_model="constant"``, the complete unit-area line shape is a mixture
    of that Gaussian and a left-exGaussian charge-loss tail whose exponential
    scale is ``tail_scale_in_sigma * sigma(E)``.  ``tail_model="none"`` omits
    both tail nuisance parameters rather than leaving an unidentified scale at
    zero mixture weight.
    """

    intercept_keV: float
    slope: float
    tail_fraction: float = 0.1
    tail_scale_in_sigma: float = 2.0
    form: ResolutionForm = "linear"
    tail_model: TailModel = "constant"
    per_run_scale_sigma: float = 0.0
    per_run_scale_bounds: tuple[float, float] = (0.7, 1.3)
    intercept_bounds_keV: tuple[float, float] = (0.05, 5.0)
    slope_bounds: tuple[float, float] = (0.0, 1e-3)
    tail_fraction_bounds: tuple[float, float] = (0.0, 0.8)
    tail_scale_bounds_in_sigma: tuple[float, float] = (0.1, 20.0)


# Backward-compatible public name used by existing callers.
LinearResolution = DetectorResolution


@dataclass(frozen=True)
class RatioDefinition:
    """A stable numerator/denominator line-rate ratio."""

    name: str
    numerator: str
    denominator: str


@dataclass(frozen=True)
class RatioResult:
    """Ratio vector and its full delta-method covariance."""

    labels: tuple[str, ...]
    values: np.ndarray
    covariance: np.ndarray
    jacobian: np.ndarray


@dataclass(frozen=True)
class ProfileInterval:
    """Likelihood-ratio interval or upper limit for one ratio."""

    ratio: str
    estimate: float
    confidence_level: float
    kind: Literal["two_sided", "upper_limit", "self_ratio", "failed"]
    lower: float
    upper: float
    threshold_delta_nll: float
    evaluations: int
    message: str
    profile_base_penalized_nll: float = float("nan")
    fit_penalized_nll: float = float("nan")
    base_nll_difference: float = float("nan")
    base_nll_consistency_tolerance: float = float("nan")
    inner_solver_failures: int = 0
    maximum_scaled_kkt_inf_norm: float = float("nan")
    inner_stationarity_tolerance: float = float("nan")
    maximum_linear_constraint_relative_residual: float = float("nan")
    linear_constraint_relative_tolerance: float = float("nan")
    maximum_stable_nll_difference_identity_error: float = float("nan")
    stable_nll_difference_identity_tolerance: float = float("nan")
    exact_cone_invalid_inner_solves: int = 0
    exact_cone_feasibility_relative_tolerance: float = float("nan")


@dataclass(frozen=True)
class BootstrapSummary:
    """Deterministic parametric-bootstrap comparison with Fisher errors."""

    seed: int
    requested_replicates: int
    successful_replicates: int
    line_names: tuple[str, ...]
    empirical_standard_deviation: np.ndarray
    fisher_standard_deviation: np.ndarray
    fisher_68_percent_coverage: np.ndarray
    fisher_95_percent_coverage: np.ndarray
    boundary_fraction: np.ndarray
    quadratic_background_constrained_fallback_replicates: int
    quadratic_background_exact_cone_invalid_replicates: int
    quadratic_background_cone_active_replicates: int
    minimum_quadratic_background_normalized_cone_margin: float
    gaussian_pseudo_observation_seed: int
    gaussian_pseudo_observation_convention: str
    gaussian_pseudo_observation_parameter_names: tuple[str, ...]
    gaussian_pseudo_observation_generating_values: np.ndarray
    gaussian_pseudo_observation_covariance: np.ndarray


@dataclass(frozen=True)
class JointPeakFitResult:
    """Joint maximum-likelihood fit and derived covariance products."""

    success: bool
    message: str
    parameter_names: tuple[str, ...]
    parameter_values: np.ndarray
    covariance: np.ndarray
    fisher_rank: int
    fisher_condition: float
    fisher_covariance_valid: bool
    fisher_covariance_inversion_coordinates: str
    active_bounds: tuple[str, ...]
    quadratic_background_cone_diagnostics: Mapping[
        str, Mapping[str, float]
    ]
    quadratic_background_exact_cone_valid: bool
    quadratic_background_cone_active_windows: tuple[str, ...]
    quadratic_background_constrained_fallback_used: bool
    quadratic_background_cone_feasibility_relative_tolerance: float
    quadratic_background_cone_activity_relative_tolerance: float
    line_names: tuple[str, ...]
    line_rates_counts_per_s: np.ndarray
    line_rate_covariance: np.ndarray
    observed_counts: np.ndarray
    expected_counts: np.ndarray
    observation_spectrum_indices: np.ndarray
    observation_window_names: tuple[str, ...]
    observation_channel_indices: np.ndarray
    data_poisson_nll: float
    gaussian_nuisance_prior_deviance: float
    penalized_nll: float
    poisson_deviance: float
    degrees_of_freedom: int
    penalized_degrees_of_freedom: int
    optimizer_converged: bool
    optimizer_method: str
    optimizer_stages: tuple[Mapping[str, object], ...]
    optimizer_configuration: Mapping[str, object]
    optimizer_scoring_iterations: int
    optimizer_polish_basin_valid: bool
    optimizer_polish_basin_restart_used: bool
    optimizer_polish_scaled_displacement_inf: float
    optimizer_polish_stable_nll_decrease: float
    optimizer_stationarity_valid: bool
    scaled_projected_gradient_inf_norm: float
    stationarity_tolerance: float
    optimizer_iterations: int
    optimizer_evaluations: int

    def parameter(self, name: str) -> float:
        """Return one physical parameter by its stable name."""

        try:
            index = self.parameter_names.index(name)
        except ValueError as error:
            raise KeyError(name) from error
        return float(self.parameter_values[index])

    def line_rate(self, name: str) -> float:
        """Return one full Gaussian-plus-declared-tail mixture rate in counts/s."""

        try:
            index = self.line_names.index(name)
        except ValueError as error:
            raise KeyError(name) from error
        return float(self.line_rates_counts_per_s[index])

    @property
    def poisson_nll(self) -> float:
        """Backward-compatible alias for the penalized fit objective."""

        return self.penalized_nll

    @property
    def calibration_prior_deviance(self) -> float:
        """Backward-compatible name for all declared Gaussian-prior deviance."""

        return self.gaussian_nuisance_prior_deviance


@dataclass
class _PreparedProblem:
    spectra: tuple[PublicSpectrum, ...]
    spec: JointPeakSpec
    calibration: CalibrationConstraint
    resolution: LinearResolution
    parameter_names: tuple[str, ...]
    initial: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    scales: np.ndarray
    prior_mean: np.ndarray
    prior_precision: np.ndarray
    prior_parameter_indices: np.ndarray
    component_parameter_indices: dict[str, int]
    run_component_parameter_indices: dict[tuple[int, str], int]
    run_scale_parameter_indices: dict[tuple[int, str], int]
    run_background_scale_parameter_indices: dict[int, int]
    run_calibration_parameter_indices: dict[int, tuple[int, int]]
    run_resolution_scale_parameter_indices: dict[int, int]
    calibration_curvature_parameter_index: int | None
    resolution_parameter_indices: tuple[int, int]
    tail_parameter_indices: tuple[int, int] | None
    background_parameter_indices: dict[str, tuple[int, ...]]
    observed: np.ndarray
    spectrum_indices: np.ndarray
    window_names: tuple[str, ...]
    channel_indices: np.ndarray
    slices: dict[tuple[int, str], slice]
    channel_edges_low: dict[tuple[int, str], np.ndarray]
    channel_edges_high: dict[tuple[int, str], np.ndarray]


@dataclass(frozen=True)
class _OptimizationOutcome:
    result: object
    parameters: np.ndarray
    solver_converged: bool
    method: str
    stages: tuple[Mapping[str, object], ...]
    stationarity_valid: bool
    scaled_projected_gradient_inf_norm: float
    scoring_iterations: int
    polish_basin_valid: bool
    basin_restart_used: bool
    polish_scaled_displacement_inf: float
    polish_stable_nll_decrease: float
    iterations: int
    evaluations: int


@dataclass(frozen=True)
class _ScoringStepOutcome:
    parameters: np.ndarray
    objective: float
    projected_gradient: float
    accepted: bool
    evaluations: int
    damping: float | None
    backtracks: int
    message: str


@dataclass(frozen=True)
class _ProfileOptimizationOutcome:
    nll: float
    success: bool
    solver_converged: bool
    stationarity_valid: bool
    scaled_kkt_inf_norm: float
    linear_constraint_relative_residual: float
    exact_cone_valid: bool
    stable_difference_identity_error: float
    message: str


def stable_seed(case_identity: str) -> int:
    """Derive a reproducible NumPy seed from a stable case identity."""

    if not case_identity:
        raise ValueError("case identity must not be empty")
    return int.from_bytes(sha256(case_identity.encode("utf-8")).digest()[:8], "big")


def resolution_sigma_and_derivatives(
    resolution: DetectorResolution,
    energy_keV: float,
    intercept_keV: float | None = None,
    slope: float | None = None,
) -> tuple[float, float, float]:
    """Return ``sigma`` and derivatives with respect to its two parameters."""

    intercept = resolution.intercept_keV if intercept_keV is None else intercept_keV
    gradient = resolution.slope if slope is None else slope
    if resolution.form == "linear":
        sigma = intercept + gradient * energy_keV
        return sigma, 1.0, energy_keV
    if resolution.form == "sqrt":
        variance = intercept**2 + gradient * energy_keV
        if variance <= 0:
            return float("nan"), float("nan"), float("nan")
        sigma = float(np.sqrt(variance))
        return sigma, intercept / sigma, energy_keV / (2.0 * sigma)
    raise ValueError(f"unsupported resolution form: {resolution.form!r}")


def gaussian_bin_probabilities(
    low_edges_keV: np.ndarray,
    high_edges_keV: np.ndarray,
    centroid_keV: float,
    sigma_keV: float,
) -> np.ndarray:
    """Integrate a unit-normalized Gaussian over calibrated bin edges."""

    low = np.asarray(low_edges_keV, dtype=np.float64)
    high = np.asarray(high_edges_keV, dtype=np.float64)
    if low.ndim != 1 or low.shape != high.shape:
        raise ValueError("Gaussian bin edges must be equal one-dimensional vectors")
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high <= low):
        raise ValueError("Gaussian bin edges must be finite with high > low")
    if not isfinite(centroid_keV):
        raise ValueError("Gaussian centroid must be finite")
    if not isfinite(sigma_keV) or sigma_keV <= 0:
        raise ValueError("Gaussian sigma must be finite and positive")
    return ndtr((high - centroid_keV) / sigma_keV) - ndtr(
        (low - centroid_keV) / sigma_keV
    )


def _left_exgaussian_cdf(
    energy_keV: np.ndarray,
    centroid_keV: float,
    sigma_keV: float,
    tail_scale_keV: float,
) -> np.ndarray:
    """CDF of ``Normal(centroid, sigma) - Exponential(tail_scale)``."""

    z = (energy_keV - centroid_keV) / sigma_keV
    log_tail_term = (
        (energy_keV - centroid_keV) / tail_scale_keV
        + 0.5 * (sigma_keV / tail_scale_keV) ** 2
        + log_ndtr(-z - sigma_keV / tail_scale_keV)
    )
    result = ndtr(z) + np.exp(np.minimum(log_tail_term, 0.0))
    return np.clip(result, 0.0, 1.0)


def _left_exgaussian_pdf(
    energy_keV: np.ndarray,
    centroid_keV: float,
    sigma_keV: float,
    tail_scale_keV: float,
) -> np.ndarray:
    z = (energy_keV - centroid_keV) / sigma_keV
    log_density = (
        -np.log(tail_scale_keV)
        + (energy_keV - centroid_keV) / tail_scale_keV
        + 0.5 * (sigma_keV / tail_scale_keV) ** 2
        + log_ndtr(-z - sigma_keV / tail_scale_keV)
    )
    return np.exp(np.clip(log_density, -745.0, 700.0))


def peak_shape_bin_probabilities(
    low_edges_keV: np.ndarray,
    high_edges_keV: np.ndarray,
    centroid_keV: float,
    sigma_keV: float,
    tail_fraction: float,
    tail_scale_in_sigma: float,
) -> np.ndarray:
    """Integrate the unit-normalized Gaussian/charge-loss mixture."""

    if not 0.0 <= tail_fraction <= 1.0:
        raise ValueError("tail fraction must lie in [0, 1]")
    if not isfinite(tail_scale_in_sigma) or tail_scale_in_sigma <= 0:
        raise ValueError("tail scale in sigma units must be finite and positive")
    gaussian = gaussian_bin_probabilities(
        low_edges_keV, high_edges_keV, centroid_keV, sigma_keV
    )
    tail_scale = tail_scale_in_sigma * sigma_keV
    tail = _left_exgaussian_cdf(
        high_edges_keV, centroid_keV, sigma_keV, tail_scale
    ) - _left_exgaussian_cdf(
        low_edges_keV, centroid_keV, sigma_keV, tail_scale
    )
    return np.maximum(
        (1.0 - tail_fraction) * gaussian + tail_fraction * tail,
        0.0,
    )


def poisson_nll(observed: np.ndarray, expected: np.ndarray) -> float:
    """Poisson negative log likelihood, excluding the data-only constant."""

    counts = np.asarray(observed, dtype=np.float64)
    means = np.asarray(expected, dtype=np.float64)
    if counts.ndim != 1 or counts.shape != means.shape:
        raise ValueError("observed and expected counts must be equal one-dimensional vectors")
    if (
        not np.isfinite(counts).all()
        or np.any(counts < 0)
        or not np.allclose(counts, np.rint(counts), rtol=0.0, atol=1e-9)
    ):
        raise ValueError("observed values must be finite nonnegative integer counts")
    if not np.isfinite(means).all() or np.any(means < 0):
        raise ValueError("expected values must be finite and nonnegative")
    if np.any((means == 0) & (counts > 0)):
        return float("inf")
    positive = counts > 0
    return float(np.sum(means) - np.sum(counts[positive] * np.log(means[positive])))


def _poisson_nll_relative_to_saturated(
    observed: np.ndarray, expected: np.ndarray
) -> float:
    """Evaluate the Poisson NLL relative to saturation without cancellation.

    Computing ``poisson_nll(y, mu) - poisson_nll(y, y)`` subtracts two sums
    of order ``sum(y log(y))``.  The fitted paper spectra make those sums many
    orders of magnitude larger than their difference, which can make an
    optimizer's function-reduction test fire at a non-stationary point.  This
    per-bin form is algebraically identical but keeps every accumulated term
    on the deviance scale.
    """

    counts = np.asarray(observed, dtype=np.float64)
    means = np.asarray(expected, dtype=np.float64)
    if counts.ndim != 1 or counts.shape != means.shape:
        raise ValueError("observed and expected counts must be equal vectors")
    if np.any((means <= 0) & (counts > 0)):
        return float("inf")
    terms = means - counts
    positive = counts > 0
    if np.any(positive):
        relative_difference = (
            counts[positive] - means[positive]
        ) / means[positive]
        terms = terms.astype(np.float64, copy=True)
        terms[positive] += counts[positive] * np.log1p(relative_difference)
    return float(np.sum(terms))


def _poisson_deviance(observed: np.ndarray, expected: np.ndarray) -> float:
    return 2.0 * _poisson_nll_relative_to_saturated(observed, expected)


def _validate_spectrum(spectrum: PublicSpectrum) -> None:
    counts = np.asarray(spectrum.counts, dtype=np.float64)
    if counts.ndim != 1 or counts.size < 5:
        raise ValueError("peak likelihood requires a one-dimensional spectrum")
    if (
        not np.isfinite(counts).all()
        or np.any(counts < 0)
        or not np.allclose(counts, np.rint(counts), rtol=0.0, atol=1e-9)
    ):
        raise ValueError("peak likelihood requires raw nonnegative integer counts")
    if not isfinite(spectrum.live_time) or spectrum.live_time <= 0:
        raise ValueError("spectrum live time must be finite and positive seconds")
    channels = np.arange(1, counts.size + 1, dtype=np.float64)
    expected_energy = spectrum.calibration_A0 + spectrum.calibration_A1 * channels
    expected_width = np.full(counts.size, spectrum.calibration_A1)
    if (
        not np.allclose(spectrum.energy_keV, expected_energy, rtol=1e-12, atol=1e-9)
        or not np.allclose(spectrum.bin_width_keV, expected_width, rtol=1e-12, atol=1e-12)
    ):
        raise ValueError("peak likelihood requires raw, unrebinned channel counts")


def _validate_spec(spec: JointPeakSpec) -> dict[str, FitWindow]:
    if not spec.name.strip():
        raise ValueError("fit identity must not be empty")
    if not spec.windows or not spec.components:
        raise ValueError("fit must declare windows and components")
    windows: dict[str, FitWindow] = {}
    for window in spec.windows:
        if not window.name.strip() or window.name in windows:
            raise ValueError("window names must be nonempty and unique")
        if (
            not isfinite(window.low_keV)
            or not isfinite(window.high_keV)
            or window.high_keV <= window.low_keV
        ):
            raise ValueError(f"window {window.name!r} has invalid bounds")
        if window.background_model not in ("affine", "quadratic"):
            raise ValueError(
                f"window {window.name!r} has unsupported background model"
            )
        windows[window.name] = window
    ordered = sorted(windows.values(), key=lambda item: item.low_keV)
    for left, right in zip(ordered, ordered[1:]):
        if right.low_keV < left.high_keV:
            raise ValueError("fit windows must not overlap")
    names: set[str] = set()
    for component in spec.components:
        if not component.name.strip() or component.name in names:
            raise ValueError("component names must be nonempty and unique")
        names.add(component.name)
        if component.window not in windows:
            raise ValueError(f"component {component.name!r} references an unknown window")
        if component.role not in ("line", "fep", "sep", "dep", "contaminant"):
            raise ValueError(f"component {component.name!r} has unsupported role")
        if not component.origin_class.strip():
            raise ValueError(f"component {component.name!r} has empty origin class")
        window = windows[component.window]
        if not isfinite(component.energy_keV) or not (
            window.low_keV < component.energy_keV < window.high_keV
        ):
            raise ValueError(f"component {component.name!r} lies outside its window")
    return windows


def _validate_constraints(
    calibration: CalibrationConstraint, resolution: LinearResolution
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(
        [calibration.offset_mean_keV, calibration.stretch_mean], dtype=np.float64
    )
    covariance = np.asarray(calibration.covariance, dtype=np.float64)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        raise ValueError("calibration covariance must be a finite 2x2 matrix")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-14):
        raise ValueError("calibration covariance must be symmetric")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("calibration covariance must be positive definite") from error
    if not (
        calibration.offset_bounds_keV[0]
        <= calibration.offset_mean_keV
        <= calibration.offset_bounds_keV[1]
        and calibration.stretch_bounds[0]
        <= calibration.stretch_mean
        <= calibration.stretch_bounds[1]
    ):
        raise ValueError("calibration mean lies outside its bounds")
    if calibration.per_run_deviation_covariance is not None:
        run_covariance = np.asarray(
            calibration.per_run_deviation_covariance, dtype=np.float64
        )
        if (
            run_covariance.shape != (2, 2)
            or not np.isfinite(run_covariance).all()
            or not np.allclose(
                run_covariance, run_covariance.T, rtol=1e-10, atol=1e-14
            )
        ):
            raise ValueError(
                "per-run calibration deviation covariance must be a finite symmetric 2x2 matrix"
            )
        try:
            np.linalg.cholesky(run_covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "per-run calibration deviation covariance must be positive definite"
            ) from error
        if not (
            calibration.per_run_offset_bounds_keV[0] < 0
            < calibration.per_run_offset_bounds_keV[1]
            and calibration.per_run_stretch_bounds[0] < 0
            < calibration.per_run_stretch_bounds[1]
        ):
            raise ValueError("per-run calibration deviation bounds must contain zero")
    if calibration.curvature_mean_keV is None:
        if calibration.curvature_sigma_keV is not None:
            raise ValueError(
                "calibration curvature sigma requires a declared curvature mean"
            )
    elif not (
        calibration.curvature_sigma_keV is not None
        and isfinite(calibration.curvature_mean_keV)
        and isfinite(calibration.curvature_sigma_keV)
        and calibration.curvature_sigma_keV > 0.0
        and calibration.curvature_bounds_keV[0]
        <= calibration.curvature_mean_keV
        <= calibration.curvature_bounds_keV[1]
        and isfinite(calibration.curvature_pivot_keV)
        and isfinite(calibration.curvature_scale_keV)
        and calibration.curvature_scale_keV > 0.0
    ):
        raise ValueError("calibration curvature constraint is invalid")
    if resolution.form not in ("linear", "sqrt"):
        raise ValueError("resolution form must be 'linear' or 'sqrt'")
    if resolution.tail_model not in ("none", "constant"):
        raise ValueError("tail model must be 'none' or 'constant'")
    if not (
        resolution.intercept_bounds_keV[0]
        <= resolution.intercept_keV
        <= resolution.intercept_bounds_keV[1]
        and resolution.slope_bounds[0]
        <= resolution.slope
        <= resolution.slope_bounds[1]
    ):
        raise ValueError("resolution initial value lies outside its bounds")
    sigma_checks = [
        resolution_sigma_and_derivatives(resolution, energy)[0]
        for energy in (0.0, 12000.0)
    ]
    if not np.isfinite(sigma_checks).all() or min(sigma_checks) <= 0:
        raise ValueError("resolution is not positive over the fitted energy range")
    if resolution.tail_model == "constant" and not (
        resolution.tail_fraction_bounds[0]
        <= resolution.tail_fraction
        <= resolution.tail_fraction_bounds[1]
        and resolution.tail_scale_bounds_in_sigma[0]
        <= resolution.tail_scale_in_sigma
        <= resolution.tail_scale_bounds_in_sigma[1]
    ):
        raise ValueError("tail initial value lies outside its bounds")
    if not isfinite(resolution.per_run_scale_sigma) or resolution.per_run_scale_sigma < 0:
        raise ValueError("per-run resolution-scale sigma must be finite and nonnegative")
    if resolution.per_run_scale_sigma > 0 and not (
        resolution.per_run_scale_bounds[0] < 1.0 < resolution.per_run_scale_bounds[1]
    ):
        raise ValueError("per-run resolution-scale bounds must contain one")
    return mean, np.linalg.inv(covariance)


def _prepare_problem(
    spectra: Sequence[PublicSpectrum],
    spec: JointPeakSpec,
    calibration: CalibrationConstraint,
    resolution: LinearResolution,
    initial: Mapping[str, float] | None,
    yield_model: YieldModel = "shared_origin_scales",
    gaussian_constraint_observations: Mapping[str, float] | None = None,
) -> _PreparedProblem:
    spectra_tuple = tuple(spectra)
    if not spectra_tuple:
        raise ValueError("at least one spectrum is required")
    for spectrum in spectra_tuple:
        _validate_spectrum(spectrum)
    windows = _validate_spec(spec)
    shared_prior_mean, shared_prior_precision = _validate_constraints(
        calibration, resolution
    )
    supplied = {} if initial is None else {key: float(value) for key, value in initial.items()}
    if not all(isfinite(value) for value in supplied.values()):
        raise ValueError("initial parameters must be finite")
    if yield_model not in ("shared_origin_scales", "independent_runs"):
        raise ValueError(f"unsupported yield model: {yield_model!r}")

    names = [
        "calibration.offset_keV",
        "calibration.fractional_gain_stretch",
        "resolution.intercept_keV",
        (
            "resolution.linear_sigma_slope_keV_per_keV"
            if resolution.form == "linear"
            else "resolution.variance_slope_keV"
        ),
    ]
    values = [
        calibration.offset_mean_keV,
        calibration.stretch_mean,
        resolution.intercept_keV,
        resolution.slope,
    ]
    lower = [
        calibration.offset_bounds_keV[0],
        calibration.stretch_bounds[0],
        resolution.intercept_bounds_keV[0],
        resolution.slope_bounds[0],
    ]
    upper = [
        calibration.offset_bounds_keV[1],
        calibration.stretch_bounds[1],
        resolution.intercept_bounds_keV[1],
        resolution.slope_bounds[1],
    ]
    scales = [
        max(float(np.sqrt(calibration.covariance[0, 0])), 1e-3),
        max(float(np.sqrt(calibration.covariance[1, 1])), 1e-7),
        max(resolution.intercept_keV, 0.2),
        max(resolution.slope, 1e-5),
    ]
    resolution_parameter_indices = (2, 3)
    tail_parameter_indices: tuple[int, int] | None = None
    if resolution.tail_model == "constant":
        tail_parameter_indices = (len(values), len(values) + 1)
        names.extend(
            (
                "shape.low_energy_tail_fraction",
                "shape.low_energy_tail_scale_in_sigma",
            )
        )
        values.extend((resolution.tail_fraction, resolution.tail_scale_in_sigma))
        lower.extend(
            (
                resolution.tail_fraction_bounds[0],
                resolution.tail_scale_bounds_in_sigma[0],
            )
        )
        upper.extend(
            (
                resolution.tail_fraction_bounds[1],
                resolution.tail_scale_bounds_in_sigma[1],
            )
        )
        scales.extend((0.1, max(resolution.tail_scale_in_sigma, 0.5)))

    calibration_curvature_parameter_index: int | None = None
    if calibration.curvature_mean_keV is not None:
        assert calibration.curvature_sigma_keV is not None
        calibration_curvature_parameter_index = len(values)
        names.append("calibration.quadratic_curvature_keV_at_domain_edges")
        values.append(calibration.curvature_mean_keV)
        lower.append(calibration.curvature_bounds_keV[0])
        upper.append(calibration.curvature_bounds_keV[1])
        scales.append(calibration.curvature_sigma_keV)

    run_calibration_parameter_indices: dict[int, tuple[int, int]] = {}
    if calibration.per_run_deviation_covariance is not None:
        run_covariance = np.asarray(
            calibration.per_run_deviation_covariance, dtype=np.float64
        )
        for run_index in range(1, len(spectra_tuple)):
            offset_index = len(values)
            stretch_index = offset_index + 1
            run_calibration_parameter_indices[run_index] = (
                offset_index,
                stretch_index,
            )
            names.extend(
                (
                    f"spectrum.{run_index}.calibration_offset_deviation_keV",
                    f"spectrum.{run_index}.fractional_gain_stretch_deviation",
                )
            )
            values.extend((0.0, 0.0))
            lower.extend(
                (
                    calibration.per_run_offset_bounds_keV[0],
                    calibration.per_run_stretch_bounds[0],
                )
            )
            upper.extend(
                (
                    calibration.per_run_offset_bounds_keV[1],
                    calibration.per_run_stretch_bounds[1],
                )
            )
            scales.extend(
                (
                    max(float(np.sqrt(run_covariance[0, 0])), 1e-4),
                    max(float(np.sqrt(run_covariance[1, 1])), 1e-8),
                )
            )

    run_resolution_scale_parameter_indices: dict[int, int] = {}
    if resolution.per_run_scale_sigma > 0:
        for run_index in range(1, len(spectra_tuple)):
            index = len(values)
            run_resolution_scale_parameter_indices[run_index] = index
            names.append(
                f"spectrum.{run_index}.resolution_scale_relative_to_spectrum_0"
            )
            values.append(1.0)
            lower.append(resolution.per_run_scale_bounds[0])
            upper.append(resolution.per_run_scale_bounds[1])
            scales.append(resolution.per_run_scale_sigma)

    indices_by_run_window: dict[tuple[int, str], np.ndarray] = {}
    slices: dict[tuple[int, str], slice] = {}
    low_channel_edges: dict[tuple[int, str], np.ndarray] = {}
    high_channel_edges: dict[tuple[int, str], np.ndarray] = {}
    observed_parts: list[np.ndarray] = []
    spectrum_index_parts: list[np.ndarray] = []
    window_name_parts: list[str] = []
    channel_index_parts: list[np.ndarray] = []
    start = 0
    for run_index, spectrum in enumerate(spectra_tuple):
        nominal_energy = np.asarray(spectrum.energy_keV, dtype=np.float64)
        for window in spec.windows:
            indices = np.flatnonzero(
                (nominal_energy >= window.low_keV)
                & (nominal_energy < window.high_keV)
            )
            if indices.size < 5:
                raise ValueError(
                    f"window {window.name!r} has fewer than five raw bins in spectrum "
                    f"{spectrum.file_id}"
                )
            key = (run_index, window.name)
            indices_by_run_window[key] = indices
            stop = start + indices.size
            slices[key] = slice(start, stop)
            start = stop
            low_channel_edges[key] = indices.astype(np.float64) + 0.5
            high_channel_edges[key] = indices.astype(np.float64) + 1.5
            observed_parts.append(np.asarray(spectrum.counts[indices], dtype=np.float64))
            spectrum_index_parts.append(np.full(indices.size, run_index, dtype=np.int64))
            window_name_parts.extend([window.name] * indices.size)
            channel_index_parts.append(indices.astype(np.int64))

    observed = np.concatenate(observed_parts)
    spectrum_indices = np.concatenate(spectrum_index_parts)
    channel_indices = np.concatenate(channel_index_parts)

    component_parameter_indices: dict[str, int] = {}
    run_component_parameter_indices: dict[tuple[int, str], int] = {}
    total_live = sum(spectrum.live_time for spectrum in spectra_tuple)
    for component in spec.components:
        sigma_guess = resolution_sigma_and_derivatives(
            resolution, component.energy_keV
        )[0]
        excess_counts = 0.0
        for run_index, spectrum in enumerate(spectra_tuple):
            indices = indices_by_run_window[(run_index, component.window)]
            energies = spectrum.energy_keV[indices]
            counts = np.asarray(spectrum.counts[indices], dtype=np.float64)
            side_count = max(2, counts.size // 5)
            background = float(
                np.median(np.concatenate((counts[:side_count], counts[-side_count:])))
            )
            near = np.abs(energies - component.energy_keV) <= max(3.0 * sigma_guess, spectrum.calibration_A1)
            if np.any(near):
                excess_counts += float(np.maximum(counts[near] - background, 0.0).sum())
        rate_guess = max(excess_counts / total_live, 1.0 / total_live)
        if yield_model == "shared_origin_scales":
            name = f"line.{component.name}.rate_counts_per_s"
            component_parameter_indices[component.name] = len(values)
            names.append(name)
            values.append(float(supplied.get(name, rate_guess)))
            lower.append(0.0)
            upper.append(np.inf)
            scales.append(max(rate_guess, 1.0 / total_live))
        else:
            for run_index, spectrum in enumerate(spectra_tuple):
                name = (
                    f"spectrum.{run_index}.line.{component.name}.rate_counts_per_s"
                )
                run_component_parameter_indices[(run_index, component.name)] = len(values)
                if run_index == 0:
                    component_parameter_indices[component.name] = len(values)
                names.append(name)
                values.append(float(supplied.get(name, rate_guess)))
                lower.append(0.0)
                upper.append(np.inf)
                scales.append(max(rate_guess, 1.0 / spectrum.live_time))

    run_scale_parameter_indices: dict[tuple[int, str], int] = {}
    origin_classes = tuple(
        dict.fromkeys(component.origin_class for component in spec.components)
    )
    for run_index in range(1, len(spectra_tuple)):
        for origin_class in (
            origin_classes if yield_model == "shared_origin_scales" else ()
        ):
            name = (
                f"spectrum.{run_index}.signal_scale_relative_to_spectrum_0"
                if origin_classes == ("shared",)
                else (
                    f"spectrum.{run_index}.signal_scale.{origin_class}"
                    "_relative_to_spectrum_0"
                )
            )
            run_scale_parameter_indices[(run_index, origin_class)] = len(values)
            names.append(name)
            values.append(float(supplied.get(name, 1.0)))
            lower.append(0.1)
            upper.append(10.0)
            scales.append(0.2)

    run_background_scale_parameter_indices: dict[int, int] = {}
    for run_index in range(1, len(spectra_tuple)):
        name = f"spectrum.{run_index}.background_scale_relative_to_spectrum_0"
        run_background_scale_parameter_indices[run_index] = len(values)
        names.append(name)
        values.append(float(supplied.get(name, 1.0)))
        lower.append(0.1)
        upper.append(10.0)
        scales.append(0.2)

    background_parameter_indices: dict[str, tuple[int, ...]] = {}
    for window in spec.windows:
        left_rates: list[float] = []
        right_rates: list[float] = []
        for run_index, spectrum in enumerate(spectra_tuple):
            counts = np.asarray(
                spectrum.counts[indices_by_run_window[(run_index, window.name)]],
                dtype=np.float64,
            )
            side_count = max(2, counts.size // 5)
            left_rates.append(
                float(np.median(counts[:side_count]))
                / spectrum.live_time
                / spectrum.calibration_A1
            )
            right_rates.append(
                float(np.median(counts[-side_count:]))
                / spectrum.live_time
                / spectrum.calibration_A1
            )
        left_guess = max(float(np.median(left_rates)), 1e-12)
        right_guess = max(float(np.median(right_rates)), 1e-12)
        coefficient_labels = (
            ("low", "high")
            if window.background_model == "affine"
            else ("low", "middle", "high")
        )
        coefficient_guesses = (
            (left_guess, right_guess)
            if window.background_model == "affine"
            else (left_guess, 0.5 * (left_guess + right_guess), right_guess)
        )
        parameter_indices = tuple(
            range(len(values), len(values) + len(coefficient_labels))
        )
        background_parameter_indices[window.name] = parameter_indices
        for label, guess in zip(coefficient_labels, coefficient_guesses):
            name = f"background.{window.name}.{label}_counts_per_s_per_keV"
            names.append(name)
            values.append(float(supplied.get(name, guess)))
            lower.append(
                _QUADRATIC_MIDDLE_LOWER_BOUND
                if window.background_model == "quadratic" and label == "middle"
                else 1.0e-15
            )
            upper.append(np.inf)
            scales.append(max(guess, 1e-8))

    values_array = np.asarray(values, dtype=np.float64)
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    scales_array = np.asarray(scales, dtype=np.float64)
    unknown = sorted(set(supplied).difference(names))
    if unknown:
        raise ValueError("unknown initial parameters: " + ", ".join(unknown))
    for index, name in enumerate(names):
        if name in supplied:
            values_array[index] = supplied[name]
    if np.any(values_array < lower_array) or np.any(values_array > upper_array):
        raise ValueError("an initial parameter lies outside its bounds")

    prior_parameter_indices_list = [0, 1]
    prior_mean_parts = [shared_prior_mean]
    prior_precision_blocks = [shared_prior_precision]
    if calibration_curvature_parameter_index is not None:
        assert calibration.curvature_mean_keV is not None
        assert calibration.curvature_sigma_keV is not None
        prior_parameter_indices_list.append(
            calibration_curvature_parameter_index
        )
        prior_mean_parts.append(
            np.asarray([calibration.curvature_mean_keV], dtype=np.float64)
        )
        prior_precision_blocks.append(
            np.asarray(
                [[1.0 / calibration.curvature_sigma_keV**2]],
                dtype=np.float64,
            )
        )
    if calibration.per_run_deviation_covariance is not None:
        run_precision = np.linalg.inv(
            np.asarray(calibration.per_run_deviation_covariance, dtype=np.float64)
        )
        for run_index in sorted(run_calibration_parameter_indices):
            prior_parameter_indices_list.extend(
                run_calibration_parameter_indices[run_index]
            )
            prior_mean_parts.append(np.zeros(2, dtype=np.float64))
            prior_precision_blocks.append(run_precision)
    if resolution.per_run_scale_sigma > 0:
        resolution_scale_precision = np.asarray(
            [[1.0 / resolution.per_run_scale_sigma**2]], dtype=np.float64
        )
        for run_index in sorted(run_resolution_scale_parameter_indices):
            prior_parameter_indices_list.append(
                run_resolution_scale_parameter_indices[run_index]
            )
            prior_mean_parts.append(np.ones(1, dtype=np.float64))
            prior_precision_blocks.append(resolution_scale_precision)
    prior_parameter_indices = np.asarray(
        prior_parameter_indices_list, dtype=np.int64
    )
    prior_mean = np.concatenate(prior_mean_parts)
    prior_precision = np.zeros((prior_mean.size, prior_mean.size), dtype=np.float64)
    prior_start = 0
    for block in prior_precision_blocks:
        prior_stop = prior_start + block.shape[0]
        prior_precision[prior_start:prior_stop, prior_start:prior_stop] = block
        prior_start = prior_stop
    if gaussian_constraint_observations is not None:
        observations = {
            str(name): float(value)
            for name, value in gaussian_constraint_observations.items()
        }
        prior_parameter_names = tuple(
            names[index] for index in prior_parameter_indices
        )
        missing = sorted(set(prior_parameter_names).difference(observations))
        unknown = sorted(set(observations).difference(prior_parameter_names))
        if missing or unknown:
            parts = []
            if missing:
                parts.append("missing: " + ", ".join(missing))
            if unknown:
                parts.append("unknown: " + ", ".join(unknown))
            raise ValueError(
                "Gaussian constraint observations must exactly match the "
                "constrained nuisance parameters (" + "; ".join(parts) + ")"
            )
        if not all(isfinite(value) for value in observations.values()):
            raise ValueError("Gaussian constraint observations must be finite")
        prior_mean = np.asarray(
            [observations[name] for name in prior_parameter_names],
            dtype=np.float64,
        )

    return _PreparedProblem(
        spectra_tuple,
        spec,
        calibration,
        resolution,
        tuple(names),
        values_array,
        lower_array,
        upper_array,
        scales_array,
        prior_mean,
        prior_precision,
        prior_parameter_indices,
        component_parameter_indices,
        run_component_parameter_indices,
        run_scale_parameter_indices,
        run_background_scale_parameter_indices,
        run_calibration_parameter_indices,
        run_resolution_scale_parameter_indices,
        calibration_curvature_parameter_index,
        resolution_parameter_indices,
        tail_parameter_indices,
        background_parameter_indices,
        observed,
        spectrum_indices,
        tuple(window_name_parts),
        channel_indices,
        slices,
        low_channel_edges,
        high_channel_edges,
    )


def quadratic_bernstein_cone_margin(
    low: float, middle: float, high: float
) -> float:
    """Return the exact interval-nonnegativity margin for a quadratic.

    For ``q(x) = low*(1-x)^2 + 2*middle*x*(1-x) + high*x^2`` on
    ``0 <= x <= 1``, nonnegativity is equivalent to nonnegative endpoints
    and ``middle + sqrt(low*high) >= 0``.
    """

    if low < 0.0 or high < 0.0:
        return float("-inf")
    return float(middle + np.sqrt(low * high))


def quadratic_bernstein_minimum(
    low: float, middle: float, high: float
) -> float:
    """Return the analytic minimum of a quadratic Bernstein polynomial."""

    curvature = low - 2.0 * middle + high
    if curvature <= 0.0:
        return float(min(low, high))
    location = np.clip((low - middle) / curvature, 0.0, 1.0)
    return float(
        low * (1.0 - location) ** 2
        + 2.0 * middle * location * (1.0 - location)
        + high * location**2
    )


def _quadratic_background_cone_diagnostics(
    problem: _PreparedProblem, parameters: np.ndarray
) -> dict[str, dict[str, float]]:
    diagnostics: dict[str, dict[str, float]] = {}
    windows = {window.name: window for window in problem.spec.windows}
    for window_name, indices in problem.background_parameter_indices.items():
        if windows[window_name].background_model != "quadratic":
            continue
        low, middle, high = (float(parameters[index]) for index in indices)
        margin = quadratic_bernstein_cone_margin(low, middle, high)
        scale = max(low, abs(middle), high, 1.0e-15)
        diagnostics[window_name] = {
            "low_counts_per_s_per_keV": low,
            "middle_counts_per_s_per_keV": middle,
            "high_counts_per_s_per_keV": high,
            "cone_margin_counts_per_s_per_keV": margin,
            "normalized_cone_margin": margin / scale,
            "minimum_density_counts_per_s_per_keV": (
                quadratic_bernstein_minimum(low, middle, high)
            ),
        }
    return diagnostics


def calibration_curvature_basis(
    spectrum: PublicSpectrum,
    calibration: CalibrationConstraint,
    channels: np.ndarray | float,
) -> np.ndarray:
    """Return the fixed dimensionless quadratic calibration basis."""

    channel_array = np.asarray(channels, dtype=np.float64)
    nominal = spectrum.calibration_A0 + spectrum.calibration_A1 * channel_array
    return (
        (nominal - calibration.curvature_pivot_keV)
        / calibration.curvature_scale_keV
    ) ** 2


def calibrated_channel_energy(
    spectrum: PublicSpectrum,
    calibration: CalibrationConstraint,
    channels: np.ndarray | float,
    offset_keV: float,
    fractional_gain_stretch: float,
    curvature_keV: float = 0.0,
) -> np.ndarray:
    """Map detector channels through the declared fitted calibration."""

    channel_array = np.asarray(channels, dtype=np.float64)
    energy = (
        spectrum.calibration_A0
        + offset_keV
        + spectrum.calibration_A1
        * (1.0 + fractional_gain_stretch)
        * channel_array
    )
    if curvature_keV:
        energy = energy + curvature_keV * calibration_curvature_basis(
            spectrum, calibration, channel_array
        )
    return energy


def _model_and_jacobian(
    problem: _PreparedProblem, parameters: np.ndarray, *, with_jacobian: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    offset, stretch = parameters[:2]
    resolution_intercept_index, resolution_slope_index = (
        problem.resolution_parameter_indices
    )
    resolution_intercept = parameters[resolution_intercept_index]
    resolution_slope = parameters[resolution_slope_index]
    if problem.tail_parameter_indices is None:
        tail_fraction = 0.0
        tail_scale_in_sigma = 1.0
    else:
        tail_fraction_index, tail_scale_index = problem.tail_parameter_indices
        tail_fraction = parameters[tail_fraction_index]
        tail_scale_in_sigma = parameters[tail_scale_index]
    curvature_index = problem.calibration_curvature_parameter_index
    curvature = 0.0 if curvature_index is None else parameters[curvature_index]
    expected = np.zeros(problem.observed.size, dtype=np.float64)
    jacobian = (
        np.zeros((problem.observed.size, parameters.size), dtype=np.float64)
        if with_jacobian
        else None
    )

    for run_index, spectrum in enumerate(problem.spectra):
        run_offset = offset
        run_stretch = stretch
        run_calibration_indices = problem.run_calibration_parameter_indices.get(
            run_index
        )
        if run_calibration_indices is not None:
            run_offset += parameters[run_calibration_indices[0]]
            run_stretch += parameters[run_calibration_indices[1]]
        run_resolution_scale_index = (
            problem.run_resolution_scale_parameter_indices.get(run_index)
        )
        run_resolution_scale = (
            1.0
            if run_resolution_scale_index is None
            else parameters[run_resolution_scale_index]
        )
        background_scale = (
            1.0
            if run_index == 0
            else parameters[
                problem.run_background_scale_parameter_indices[run_index]
            ]
        )
        A1 = spectrum.calibration_A1 * (1.0 + run_stretch)
        for window in problem.spec.windows:
            key = (run_index, window.name)
            segment = problem.slices[key]
            low_channels = problem.channel_edges_low[key]
            high_channels = problem.channel_edges_high[key]
            low_edges = calibrated_channel_energy(
                spectrum,
                problem.calibration,
                low_channels,
                run_offset,
                run_stretch,
                curvature,
            )
            high_edges = calibrated_channel_energy(
                spectrum,
                problem.calibration,
                high_channels,
                run_offset,
                run_stretch,
                curvature,
            )
            center_channels = 0.5 * (low_channels + high_channels)
            center_energy = calibrated_channel_energy(
                spectrum,
                problem.calibration,
                center_channels,
                run_offset,
                run_stretch,
                curvature,
            )
            bin_width = (
                np.full_like(low_edges, A1)
                if curvature_index is None
                else high_edges - low_edges
            )
            if np.any(bin_width <= 0.0):
                return np.full_like(expected, np.nan), jacobian
            raw_fraction = (center_energy - window.low_keV) / (
                window.high_keV - window.low_keV
            )
            fraction = np.clip(raw_fraction, 0.0, 1.0)
            differentiable = ((raw_fraction > 0.0) & (raw_fraction < 1.0)).astype(
                np.float64
            )
            width = window.high_keV - window.low_keV
            if window.background_model == "affine":
                basis = np.column_stack((1.0 - fraction, fraction))
                basis_derivative = np.column_stack(
                    (-np.ones_like(fraction), np.ones_like(fraction))
                )
            else:
                basis = np.column_stack(
                    (
                        (1.0 - fraction) ** 2,
                        2.0 * fraction * (1.0 - fraction),
                        fraction**2,
                    )
                )
                basis_derivative = np.column_stack(
                    (
                        -2.0 * (1.0 - fraction),
                        2.0 - 4.0 * fraction,
                        2.0 * fraction,
                    )
                )
            background_indices = problem.background_parameter_indices[window.name]
            background_coefficients = parameters[np.asarray(background_indices)]
            density = basis @ background_coefficients
            density_derivative_fraction = basis_derivative @ background_coefficients
            fraction_derivative_offset = differentiable / width
            fraction_derivative_stretch = (
                differentiable
                * spectrum.calibration_A1
                * center_channels
                / width
            )
            if curvature_index is None:
                curvature_basis_low = curvature_basis_high = None
                fraction_derivative_curvature = None
                bin_width_derivative_curvature = None
            else:
                curvature_basis_low = calibration_curvature_basis(
                    spectrum, problem.calibration, low_channels
                )
                curvature_basis_high = calibration_curvature_basis(
                    spectrum, problem.calibration, high_channels
                )
                curvature_basis_center = calibration_curvature_basis(
                    spectrum, problem.calibration, center_channels
                )
                fraction_derivative_curvature = (
                    differentiable * curvature_basis_center / width
                )
                bin_width_derivative_curvature = (
                    curvature_basis_high - curvature_basis_low
                )
            expected[segment] = (
                spectrum.live_time * background_scale * bin_width * density
            )
            if jacobian is not None:
                background_derivative_offset = (
                    spectrum.live_time
                    * background_scale
                    * bin_width
                    * density_derivative_fraction
                    * fraction_derivative_offset
                )
                background_derivative_stretch = (
                    spectrum.live_time
                    * background_scale
                    * (
                        spectrum.calibration_A1 * density
                        + bin_width
                        * density_derivative_fraction
                        * fraction_derivative_stretch
                    )
                )
                jacobian[segment, 0] += background_derivative_offset
                jacobian[segment, 1] += background_derivative_stretch
                if run_calibration_indices is not None:
                    jacobian[
                        segment, run_calibration_indices[0]
                    ] += background_derivative_offset
                    jacobian[
                        segment, run_calibration_indices[1]
                    ] += background_derivative_stretch
                if curvature_index is not None:
                    assert fraction_derivative_curvature is not None
                    assert bin_width_derivative_curvature is not None
                    jacobian[segment, curvature_index] += (
                        spectrum.live_time
                        * background_scale
                        * (
                            bin_width_derivative_curvature * density
                            + bin_width
                            * density_derivative_fraction
                            * fraction_derivative_curvature
                        )
                    )
                for basis_index, parameter_index in enumerate(background_indices):
                    jacobian[segment, parameter_index] = (
                        spectrum.live_time
                        * background_scale
                        * bin_width
                        * basis[:, basis_index]
                    )
                if run_index > 0:
                    background_scale_index = (
                        problem.run_background_scale_parameter_indices[run_index]
                    )
                    jacobian[segment, background_scale_index] += (
                        spectrum.live_time * bin_width * density
                    )

            for component in problem.spec.components:
                if component.window != window.name:
                    continue
                rate_index = problem.run_component_parameter_indices.get(
                    (run_index, component.name),
                    problem.component_parameter_indices[component.name],
                )
                rate = parameters[rate_index]
                signal_scale = (
                    1.0
                    if run_index == 0 or problem.run_component_parameter_indices
                    else parameters[
                        problem.run_scale_parameter_indices[
                            (run_index, component.origin_class)
                        ]
                    ]
                )
                base_sigma, sigma_derivative_intercept, sigma_derivative_slope = (
                    resolution_sigma_and_derivatives(
                        problem.resolution,
                        component.energy_keV,
                        resolution_intercept,
                        resolution_slope,
                    )
                )
                sigma = run_resolution_scale * base_sigma
                sigma_derivative_intercept *= run_resolution_scale
                sigma_derivative_slope *= run_resolution_scale
                if not isfinite(sigma) or sigma <= 0:
                    return np.full_like(expected, np.nan), jacobian
                probability = peak_shape_bin_probabilities(
                    low_edges,
                    high_edges,
                    component.energy_keV,
                    sigma,
                    tail_fraction,
                    tail_scale_in_sigma,
                )
                normalization = spectrum.live_time * signal_scale
                expected[segment] += normalization * rate * probability
                if jacobian is None:
                    continue
                z_low = (low_edges - component.energy_keV) / sigma
                z_high = (high_edges - component.energy_keV) / sigma
                gaussian_pdf_low = np.exp(-0.5 * z_low**2) / (_SQRT_2PI * sigma)
                gaussian_pdf_high = np.exp(-0.5 * z_high**2) / (_SQRT_2PI * sigma)
                tail_scale_keV = tail_scale_in_sigma * sigma
                tail_pdf_low = _left_exgaussian_pdf(
                    low_edges, component.energy_keV, sigma, tail_scale_keV
                )
                tail_pdf_high = _left_exgaussian_pdf(
                    high_edges, component.energy_keV, sigma, tail_scale_keV
                )
                pdf_low = (1.0 - tail_fraction) * gaussian_pdf_low + tail_fraction * tail_pdf_low
                pdf_high = (1.0 - tail_fraction) * gaussian_pdf_high + tail_fraction * tail_pdf_high
                d_probability_d_offset = pdf_high - pdf_low
                d_probability_d_stretch = (
                    spectrum.calibration_A1
                    * (high_channels * pdf_high - low_channels * pdf_low)
                )
                if curvature_index is not None:
                    assert curvature_basis_low is not None
                    assert curvature_basis_high is not None
                    d_probability_d_curvature = (
                        curvature_basis_high * pdf_high
                        - curvature_basis_low * pdf_low
                    )
                sigma_step = max(1e-5 * sigma, 1e-7)
                probability_sigma_high = peak_shape_bin_probabilities(
                    low_edges,
                    high_edges,
                    component.energy_keV,
                    sigma + sigma_step,
                    tail_fraction,
                    tail_scale_in_sigma,
                )
                probability_sigma_low = peak_shape_bin_probabilities(
                    low_edges,
                    high_edges,
                    component.energy_keV,
                    sigma - sigma_step,
                    tail_fraction,
                    tail_scale_in_sigma,
                )
                d_probability_d_sigma = (
                    probability_sigma_high - probability_sigma_low
                ) / (2.0 * sigma_step)
                gaussian_probability = ndtr(z_high) - ndtr(z_low)
                tail_probability = (
                    _left_exgaussian_cdf(
                        high_edges, component.energy_keV, sigma, tail_scale_keV
                    )
                    - _left_exgaussian_cdf(
                        low_edges, component.energy_keV, sigma, tail_scale_keV
                    )
                )
                d_probability_d_tail_fraction = tail_probability - gaussian_probability
                tail_scale_step = max(1e-5 * tail_scale_in_sigma, 1e-6)
                probability_tail_high = peak_shape_bin_probabilities(
                    low_edges,
                    high_edges,
                    component.energy_keV,
                    sigma,
                    tail_fraction,
                    tail_scale_in_sigma + tail_scale_step,
                )
                probability_tail_low = peak_shape_bin_probabilities(
                    low_edges,
                    high_edges,
                    component.energy_keV,
                    sigma,
                    tail_fraction,
                    tail_scale_in_sigma - tail_scale_step,
                )
                d_probability_d_tail_scale = (
                    probability_tail_high - probability_tail_low
                ) / (2.0 * tail_scale_step)
                common = normalization * rate
                jacobian[segment, 0] += common * d_probability_d_offset
                jacobian[segment, 1] += common * d_probability_d_stretch
                if curvature_index is not None:
                    jacobian[segment, curvature_index] += (
                        common * d_probability_d_curvature
                    )
                if run_calibration_indices is not None:
                    jacobian[segment, run_calibration_indices[0]] += (
                        common * d_probability_d_offset
                    )
                    jacobian[segment, run_calibration_indices[1]] += (
                        common * d_probability_d_stretch
                    )
                jacobian[segment, resolution_intercept_index] += (
                    common * d_probability_d_sigma * sigma_derivative_intercept
                )
                jacobian[segment, resolution_slope_index] += (
                    common * d_probability_d_sigma * sigma_derivative_slope
                )
                if run_resolution_scale_index is not None:
                    jacobian[segment, run_resolution_scale_index] += (
                        common * d_probability_d_sigma * base_sigma
                    )
                if problem.tail_parameter_indices is not None:
                    jacobian[segment, tail_fraction_index] += (
                        common * d_probability_d_tail_fraction
                    )
                    jacobian[segment, tail_scale_index] += (
                        common * d_probability_d_tail_scale
                    )
                jacobian[segment, rate_index] += normalization * probability
                if run_index > 0 and not problem.run_component_parameter_indices:
                    scale_index = problem.run_scale_parameter_indices[
                        (run_index, component.origin_class)
                    ]
                    jacobian[segment, scale_index] += (
                        spectrum.live_time * rate * probability
                    )
    return expected, jacobian


def _objective_and_gradient(
    problem: _PreparedProblem, parameters: np.ndarray
) -> tuple[float, np.ndarray]:
    nll, gradient, _ = _objective_gradient_and_expected(problem, parameters)
    return nll, gradient


def _objective_gradient_and_expected(
    problem: _PreparedProblem, parameters: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    expected, jacobian = _model_and_jacobian(problem, parameters, with_jacobian=True)
    assert jacobian is not None
    if not np.isfinite(expected).all() or np.any(expected <= 0):
        return (
            float("inf"),
            np.zeros(parameters.size, dtype=np.float64),
            expected,
        )
    residual_score = 1.0 - problem.observed / expected
    gradient = jacobian.T @ residual_score
    prior_delta = (
        parameters[problem.prior_parameter_indices] - problem.prior_mean
    )
    nll = _poisson_nll_relative_to_saturated(
        problem.observed, expected
    ) + 0.5 * float(
        prior_delta @ problem.prior_precision @ prior_delta
    )
    gradient[problem.prior_parameter_indices] += (
        problem.prior_precision @ prior_delta
    )
    return nll, gradient, expected


def _penalized_nll_difference(
    problem: _PreparedProblem,
    old_parameters: np.ndarray,
    new_parameters: np.ndarray,
    old_expected: np.ndarray,
    new_expected: np.ndarray,
) -> float:
    """Evaluate ``NLL(new) - NLL(old)`` without subtracting total NLLs."""

    if (
        np.any(old_expected <= 0.0)
        or np.any(new_expected <= 0.0)
        or not np.isfinite(old_expected).all()
        or not np.isfinite(new_expected).all()
    ):
        return float("inf")
    terms = new_expected - old_expected
    positive = problem.observed > 0.0
    if np.any(positive):
        relative_difference = (
            old_expected[positive] - new_expected[positive]
        ) / new_expected[positive]
        terms = terms.astype(np.float64, copy=True)
        terms[positive] += problem.observed[positive] * np.log1p(
            relative_difference
        )
    data_difference = np.sum(terms, dtype=np.longdouble)

    prior_indices = problem.prior_parameter_indices
    old_delta = old_parameters[prior_indices] - problem.prior_mean
    new_delta = new_parameters[prior_indices] - problem.prior_mean
    prior_sum = np.asarray(old_delta + new_delta, dtype=np.longdouble)
    prior_step = np.asarray(new_delta - old_delta, dtype=np.longdouble)
    prior_precision = np.asarray(problem.prior_precision, dtype=np.longdouble)
    prior_difference = np.longdouble(0.5) * np.sum(
        prior_sum * (prior_precision @ prior_step), dtype=np.longdouble
    )
    return float(data_difference + prior_difference)


def _scaled_bounds(problem: _PreparedProblem) -> list[tuple[float | None, float | None]]:
    return _scaled_bounds_from_reference(
        problem.initial, problem.lower, problem.upper, problem.scales
    )


def _scaled_bounds_from_reference(
    reference: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    scales: np.ndarray,
) -> list[tuple[float | None, float | None]]:
    bounds: list[tuple[float | None, float | None]] = []
    for value, low, high, scale in zip(
        reference, lower, upper, scales
    ):
        bounds.append(
            (
                None if not np.isfinite(low) else (low - value) / scale,
                None if not np.isfinite(high) else (high - value) / scale,
            )
        )
    return bounds


def _scaled_projected_gradient(
    problem: _PreparedProblem,
    parameters: np.ndarray,
    physical_gradient: np.ndarray,
) -> np.ndarray:
    """Return the bound-aware gradient in the optimizer's scaled coordinates."""

    return _scaled_projected_gradient_for_bounds(
        parameters,
        physical_gradient,
        problem.lower,
        problem.upper,
        problem.scales,
    )


def _scaled_projected_gradient_for_bounds(
    parameters: np.ndarray,
    physical_gradient: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Return a box-aware gradient in declared scaled coordinates."""

    projected = np.asarray(physical_gradient, dtype=np.float64) * scales
    projected = projected.copy()
    for index, (value, low, high, scale) in enumerate(
        zip(parameters, lower, upper, scales)
    ):
        tolerance = _ACTIVE_BOUND_TOLERANCE * max(scale, 1.0e-12)
        if np.isfinite(low) and value - low <= tolerance and projected[index] > 0:
            projected[index] = 0.0
        elif (
            np.isfinite(high)
            and high - value <= tolerance
            and projected[index] < 0
        ):
            projected[index] = 0.0
    return projected


def _stationarity_at(
    problem: _PreparedProblem, parameters: np.ndarray
) -> tuple[float, float]:
    value, gradient = _objective_and_gradient(problem, parameters)
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        return value, float("inf")
    projected = _scaled_projected_gradient(problem, parameters, gradient)
    return value, float(np.max(np.abs(projected), initial=0.0))


def _optimization_stage_record(
    method: str,
    optimization: object,
    objective: float,
    projected_gradient: float,
    *,
    accepted: bool,
) -> dict[str, object]:
    return {
        "method": method,
        "solver_success": bool(getattr(optimization, "success", False)),
        "status": int(getattr(optimization, "status", -1)),
        "nll": float(objective),
        "scaled_projected_gradient_inf_norm": float(projected_gradient),
        "accepted": bool(accepted),
        "message": str(getattr(optimization, "message", "")),
    }


def _polish_basin_metrics(
    problem: _PreparedProblem,
    warm_parameters: np.ndarray,
    warm_objective: float,
    candidate_parameters: np.ndarray,
    candidate_objective: float,
) -> tuple[float, float, bool]:
    scaled_displacement = float(
        np.max(
            np.abs((candidate_parameters - warm_parameters) / problem.scales),
            initial=0.0,
        )
    )
    stable_nll_decrease = float(warm_objective - candidate_objective)
    valid = bool(
        np.isfinite(scaled_displacement)
        and np.isfinite(stable_nll_decrease)
        and scaled_displacement <= _POLISH_MAX_SCALED_DISPLACEMENT_INF
        and stable_nll_decrease <= _POLISH_MAX_STABLE_NLL_DECREASE
    )
    return scaled_displacement, stable_nll_decrease, valid


def _scaled_expected_fisher(
    problem: _PreparedProblem, parameters: np.ndarray
) -> np.ndarray:
    expected, jacobian = _model_and_jacobian(
        problem, parameters, with_jacobian=True
    )
    if (
        jacobian is None
        or not np.isfinite(expected).all()
        or np.any(expected <= 0.0)
        or not np.isfinite(jacobian).all()
    ):
        return np.full(
            (parameters.size, parameters.size), float("nan"), dtype=np.float64
        )
    fisher = jacobian.T @ (jacobian / expected[:, np.newaxis])
    fisher[
        np.ix_(problem.prior_parameter_indices, problem.prior_parameter_indices)
    ] += problem.prior_precision
    scaled = (
        problem.scales[:, np.newaxis]
        * fisher
        * problem.scales[np.newaxis, :]
    )
    return (scaled + scaled.T) / 2.0


def _fisher_scoring_step(
    problem: _PreparedProblem,
    parameters: np.ndarray,
    objective: float,
) -> _ScoringStepOutcome:
    """Take one deterministic, bound-aware expected-Fisher scoring step."""

    current_objective, physical_gradient, current_expected = (
        _objective_gradient_and_expected(problem, parameters)
    )
    if not np.isfinite(current_objective) or not np.isfinite(
        physical_gradient
    ).all():
        return _ScoringStepOutcome(
            parameters,
            objective,
            float("inf"),
            False,
            1,
            None,
            0,
            "non-finite objective or gradient",
        )
    if current_objective != objective:
        objective = current_objective

    scaled_gradient = physical_gradient * problem.scales
    projected_gradient = _scaled_projected_gradient(
        problem, parameters, physical_gradient
    )
    fixed = np.zeros(parameters.size, dtype=bool)
    for index, (value, low, high, scale, gradient) in enumerate(
        zip(
            parameters,
            problem.lower,
            problem.upper,
            problem.scales,
            scaled_gradient,
        )
    ):
        tolerance = _ACTIVE_BOUND_TOLERANCE * max(scale, 1.0e-12)
        fixed[index] = bool(
            (np.isfinite(low) and value - low <= tolerance and gradient > 0.0)
            or (
                np.isfinite(high)
                and high - value <= tolerance
                and gradient < 0.0
            )
        )
    free = np.flatnonzero(~fixed)
    if free.size == 0:
        return _ScoringStepOutcome(
            parameters,
            objective,
            float(np.max(np.abs(projected_gradient), initial=0.0)),
            False,
            1,
            None,
            0,
            "no free parameters in scoring system",
        )

    scaled_fisher = _scaled_expected_fisher(problem, parameters)
    free_fisher = scaled_fisher[np.ix_(free, free)]
    if not np.isfinite(free_fisher).all():
        return _ScoringStepOutcome(
            parameters,
            objective,
            float(np.max(np.abs(projected_gradient), initial=0.0)),
            False,
            1,
            None,
            0,
            "non-finite expected Fisher matrix",
        )
    diagonal_scale = np.maximum(np.abs(np.diag(free_fisher)), 1.0)
    damping_used: float | None = None
    free_direction: np.ndarray | None = None
    for damping in _SCORING_DAMPING_LADDER:
        trial_fisher = free_fisher.copy()
        if damping > 0.0:
            trial_fisher.flat[:: trial_fisher.shape[0] + 1] += (
                damping * diagonal_scale
            )
        try:
            factor = cho_factor(
                trial_fisher,
                lower=True,
                overwrite_a=False,
                check_finite=True,
            )
            free_direction = cho_solve(
                factor,
                -scaled_gradient[free],
                overwrite_b=False,
                check_finite=True,
            )
        except (np.linalg.LinAlgError, ValueError):
            continue
        damping_used = damping
        break
    if free_direction is None or not np.isfinite(free_direction).all():
        return _ScoringStepOutcome(
            parameters,
            objective,
            float(np.max(np.abs(projected_gradient), initial=0.0)),
            False,
            1,
            None,
            0,
            "expected Fisher Cholesky failed for fixed damping ladder",
        )

    direction = np.zeros(parameters.size, dtype=np.float64)
    direction[free] = free_direction
    directional_derivative = float(scaled_gradient @ direction)
    if not np.isfinite(directional_derivative) or directional_derivative >= 0.0:
        return _ScoringStepOutcome(
            parameters,
            objective,
            float(np.max(np.abs(projected_gradient), initial=0.0)),
            False,
            1,
            damping_used,
            0,
            "expected Fisher direction is not a descent direction",
        )

    current_scaled = (parameters - problem.initial) / problem.scales
    scale_for_zero_step = max(
        1.0, float(np.max(np.abs(current_scaled), initial=0.0))
    )
    evaluations = 1
    for backtracks in range(_SCORING_MAX_BACKTRACKS + 1):
        step_scale = 0.5**backtracks
        candidate = parameters + problem.scales * (step_scale * direction)
        candidate = np.maximum(candidate, problem.lower)
        candidate = np.minimum(candidate, problem.upper)
        actual_step = (candidate - parameters) / problem.scales
        step_norm = float(np.max(np.abs(actual_step), initial=0.0))
        if step_norm <= _SCORING_ZERO_STEP_TOLERANCE * scale_for_zero_step:
            return _ScoringStepOutcome(
                parameters,
                objective,
                float(np.max(np.abs(projected_gradient), initial=0.0)),
                False,
                evaluations,
                damping_used,
                backtracks,
                "bound-clipped scoring step is numerically zero",
            )
        actual_directional_derivative = float(scaled_gradient @ actual_step)
        if actual_directional_derivative >= 0.0:
            continue
        (
            candidate_objective,
            candidate_gradient,
            candidate_expected,
        ) = _objective_gradient_and_expected(problem, candidate)
        objective_difference = _penalized_nll_difference(
            problem,
            parameters,
            candidate,
            current_expected,
            candidate_expected,
        )
        evaluations += 1
        armijo_limit = (
            _SCORING_ARMIJO_COEFFICIENT * actual_directional_derivative
        )
        candidate_projected = _scaled_projected_gradient(
            problem, candidate, candidate_gradient
        )
        candidate_projected_norm = float(
            np.max(np.abs(candidate_projected), initial=0.0)
        )
        strict_armijo = bool(
            objective_difference <= 0.0
            and objective_difference <= armijo_limit
        )
        numerically_flat_stationarity_repair = bool(
            abs(objective_difference)
            <= _SCORING_NUMERICALLY_FLAT_NLL_TOLERANCE
            and candidate_projected_norm
            <= max(
                _STATIONARITY_TOLERANCE,
                _SCORING_NUMERICALLY_FLAT_GRADIENT_REDUCTION_FACTOR
                * float(np.max(np.abs(projected_gradient), initial=0.0)),
            )
        )
        if (
            np.isfinite(candidate_objective)
            and np.isfinite(candidate_gradient).all()
            and np.isfinite(objective_difference)
            and (strict_armijo or numerically_flat_stationarity_repair)
        ):
            return _ScoringStepOutcome(
                candidate,
                candidate_objective,
                candidate_projected_norm,
                True,
                evaluations,
                damping_used,
                backtracks,
                (
                    "Armijo step accepted"
                    if strict_armijo
                    else (
                        "numerically flat step accepted after the declared "
                        "projected-gradient reduction"
                    )
                ),
            )
    return _ScoringStepOutcome(
        parameters,
        objective,
        float(np.max(np.abs(projected_gradient), initial=0.0)),
        False,
        evaluations,
        damping_used,
        _SCORING_MAX_BACKTRACKS,
        "Armijo backtracking exhausted",
    )


def _quadratic_cone_coordinate_system(
    problem: _PreparedProblem, parameters: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map physical quadratic coefficients to exact-cone slack coordinates."""

    coordinates = np.asarray(parameters, dtype=np.float64).copy()
    lower = problem.lower.copy()
    upper = problem.upper.copy()
    scales = problem.scales.copy()
    windows = {window.name: window for window in problem.spec.windows}
    for window_name, indices in problem.background_parameter_indices.items():
        if windows[window_name].background_model != "quadratic":
            continue
        low_index, middle_index, high_index = indices
        low = coordinates[low_index]
        high = coordinates[high_index]
        root = np.sqrt(low * high)
        coordinates[middle_index] = max(
            parameters[middle_index] + root, 0.0
        )
        lower[middle_index] = 0.0
        upper[middle_index] = np.inf
        scales[middle_index] = max(scales[middle_index], root, 1.0e-8)
    return coordinates, lower, upper, scales


def _physical_from_quadratic_cone_coordinates(
    problem: _PreparedProblem, coordinates: np.ndarray
) -> np.ndarray:
    physical = np.asarray(coordinates, dtype=np.float64).copy()
    windows = {window.name: window for window in problem.spec.windows}
    for window_name, indices in problem.background_parameter_indices.items():
        if windows[window_name].background_model != "quadratic":
            continue
        low_index, middle_index, high_index = indices
        physical[middle_index] = coordinates[middle_index] - np.sqrt(
            coordinates[low_index] * coordinates[high_index]
        )
    return physical


def _gradient_in_quadratic_cone_coordinates(
    problem: _PreparedProblem,
    coordinates: np.ndarray,
    physical_gradient: np.ndarray,
) -> np.ndarray:
    gradient = np.asarray(physical_gradient, dtype=np.float64).copy()
    windows = {window.name: window for window in problem.spec.windows}
    for window_name, indices in problem.background_parameter_indices.items():
        if windows[window_name].background_model != "quadratic":
            continue
        low_index, middle_index, high_index = indices
        low = coordinates[low_index]
        high = coordinates[high_index]
        middle_gradient = physical_gradient[middle_index]
        gradient[low_index] += middle_gradient * (
            -0.5 * np.sqrt(high / low)
        )
        gradient[high_index] += middle_gradient * (
            -0.5 * np.sqrt(low / high)
        )
        gradient[middle_index] = middle_gradient
    return gradient


def _quadratic_cone_physical_coordinate_jacobian(
    problem: _PreparedProblem, coordinates: np.ndarray
) -> np.ndarray:
    """Return d(physical parameters)/d(exact-cone coordinates)."""

    transformation = np.eye(coordinates.size, dtype=np.float64)
    windows = {window.name: window for window in problem.spec.windows}
    for window_name, indices in problem.background_parameter_indices.items():
        if windows[window_name].background_model != "quadratic":
            continue
        low_index, middle_index, high_index = indices
        low = coordinates[low_index]
        high = coordinates[high_index]
        transformation[middle_index, low_index] = (
            -0.5 * np.sqrt(high / low)
        )
        transformation[middle_index, high_index] = (
            -0.5 * np.sqrt(low / high)
        )
    return transformation


def _scaled_expected_fisher_in_quadratic_cone_coordinates(
    problem: _PreparedProblem,
    coordinates: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    physical = _physical_from_quadratic_cone_coordinates(
        problem, coordinates
    )
    expected, jacobian = _model_and_jacobian(
        problem, physical, with_jacobian=True
    )
    if (
        jacobian is None
        or not np.isfinite(expected).all()
        or np.any(expected <= 0.0)
        or not np.isfinite(jacobian).all()
    ):
        return np.full(
            (coordinates.size, coordinates.size),
            float("nan"),
            dtype=np.float64,
        )
    physical_fisher = jacobian.T @ (
        jacobian / expected[:, np.newaxis]
    )
    physical_fisher[
        np.ix_(problem.prior_parameter_indices, problem.prior_parameter_indices)
    ] += problem.prior_precision
    transformation = _quadratic_cone_physical_coordinate_jacobian(
        problem, coordinates
    )
    coordinate_fisher = (
        transformation.T @ physical_fisher @ transformation
    )
    scaled = (
        scales[:, np.newaxis]
        * coordinate_fisher
        * scales[np.newaxis, :]
    )
    return (scaled + scaled.T) / 2.0


def _exact_cone_fallback(
    problem: _PreparedProblem, relaxed: _OptimizationOutcome
) -> _OptimizationOutcome:
    """Solve the exact quadratic-background cone in smooth slack coordinates."""

    coordinate_initial, lower, upper, scales = (
        _quadratic_cone_coordinate_system(problem, relaxed.parameters)
    )
    reference_physical = _physical_from_quadratic_cone_coordinates(
        problem, coordinate_initial
    )
    reference_nll, _, reference_expected = _objective_gradient_and_expected(
        problem, reference_physical
    )

    def objective_scaled(scaled: np.ndarray) -> tuple[float, np.ndarray]:
        coordinates = coordinate_initial + scales * scaled
        physical = _physical_from_quadratic_cone_coordinates(
            problem, coordinates
        )
        _, physical_gradient, expected = _objective_gradient_and_expected(
            problem, physical
        )
        value = _penalized_nll_difference(
            problem,
            reference_physical,
            physical,
            reference_expected,
            expected,
        )
        coordinate_gradient = _gradient_in_quadratic_cone_coordinates(
            problem, coordinates, physical_gradient
        )
        return value, coordinate_gradient * scales

    scaled_bounds = [
        (
            None if not np.isfinite(low) else (low - value) / scale,
            None if not np.isfinite(high) else (high - value) / scale,
        )
        for value, low, high, scale in zip(
            coordinate_initial, lower, upper, scales
        )
    ]
    transformed_warm = minimize(
        objective_scaled,
        np.zeros(coordinate_initial.size, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=scaled_bounds,
        options=dict(_LBFGSB_OPTIONS),
    )
    transformed_polish = minimize(
        objective_scaled,
        transformed_warm.x,
        method="SLSQP",
        jac=True,
        bounds=scaled_bounds,
        options=dict(_EXACT_CONE_SLSQP_OPTIONS),
    )

    def assess(result: object, method: str) -> dict[str, object]:
        coordinates = coordinate_initial + scales * np.asarray(result.x)
        physical = _physical_from_quadratic_cone_coordinates(
            problem, coordinates
        )
        nll, physical_gradient, expected = _objective_gradient_and_expected(
            problem, physical
        )
        difference = _penalized_nll_difference(
            problem,
            reference_physical,
            physical,
            reference_expected,
            expected,
        )
        identity_error = abs(difference - (nll - reference_nll))
        coordinate_gradient = _gradient_in_quadratic_cone_coordinates(
            problem, coordinates, physical_gradient
        )
        projected = _scaled_projected_gradient_for_bounds(
            coordinates, coordinate_gradient, lower, upper, scales
        )
        projected_norm = float(np.max(np.abs(projected), initial=0.0))
        cone_diagnostics = _quadratic_background_cone_diagnostics(
            problem, physical
        )
        cone_valid = all(
            item["normalized_cone_margin"]
            >= -_EXACT_CONE_FEASIBILITY_RELATIVE_TOLERANCE
            for item in cone_diagnostics.values()
        )
        monotone = bool(np.isfinite(difference) and difference <= 0.0)
        stationary = bool(
            monotone
            and cone_valid
            and np.isfinite(nll)
            and projected_norm <= _STATIONARITY_TOLERANCE
            and identity_error <= _NLL_DIFFERENCE_IDENTITY_TOLERANCE
        )
        return {
            "result": result,
            "method": method,
            "coordinates": coordinates,
            "physical": physical,
            "nll": nll,
            "difference": difference,
            "identity_error": identity_error,
            "projected_norm": projected_norm,
            "cone_valid": cone_valid,
            "monotone": monotone,
            "stationary": stationary,
            "solver_success": bool(getattr(result, "success", False)),
        }

    candidates = (
        assess(transformed_warm, "exact quadratic cone transformed L-BFGS-B"),
        assess(transformed_polish, "exact quadratic cone transformed SLSQP"),
    )
    accepted = [
        candidate
        for candidate in candidates
        if candidate["solver_success"] and candidate["stationary"]
    ]
    if accepted:
        chosen = min(accepted, key=lambda candidate: float(candidate["nll"]))
    else:
        monotone_candidates = [
            candidate for candidate in candidates if candidate["monotone"]
        ]
        chosen = (
            min(
                monotone_candidates,
                key=lambda candidate: float(candidate["nll"]),
            )
            if monotone_candidates
            else candidates[0]
        )
    chosen_accepted = any(candidate is chosen for candidate in accepted)
    stages: list[Mapping[str, object]] = []
    for candidate in candidates:
        result = candidate["result"]
        stages.append(
            {
                "method": candidate["method"],
                "solver_success": candidate["solver_success"],
                "status": int(getattr(result, "status", -1)),
                "nll": float(candidate["nll"]),
                "scaled_projected_gradient_inf_norm": float(
                    candidate["projected_norm"]
                ),
                "accepted": bool(candidate is chosen and chosen_accepted),
                "exact_cone_parameterization": (
                    "middle = slack - sqrt(low*high); slack >= 0"
                ),
                "stable_difference_from_projected_reference": float(
                    candidate["difference"]
                ),
                "stable_difference_identity_error": float(
                    candidate["identity_error"]
                ),
                "stable_difference_identity_tolerance": (
                    _NLL_DIFFERENCE_IDENTITY_TOLERANCE
                ),
                "cone_valid": candidate["cone_valid"],
                "message": str(getattr(result, "message", "")),
            }
        )
    chosen_result = chosen["result"]
    stationarity_valid = bool(chosen["stationary"])
    return _OptimizationOutcome(
        chosen_result,
        np.asarray(chosen["physical"]),
        bool(chosen["solver_success"]),
        f"{relaxed.method} -> {chosen['method']}",
        relaxed.stages + tuple(stages),
        stationarity_valid,
        float(chosen["projected_norm"]),
        relaxed.scoring_iterations,
        relaxed.polish_basin_valid,
        relaxed.basin_restart_used,
        relaxed.polish_scaled_displacement_inf,
        relaxed.polish_stable_nll_decrease,
        relaxed.iterations
        + int(getattr(transformed_warm, "nit", 0))
        + int(getattr(transformed_polish, "nit", 0)),
        relaxed.evaluations
        + int(getattr(transformed_warm, "nfev", 0))
        + int(getattr(transformed_polish, "nfev", 0)),
    )


def _optimize_problem(
    problem: _PreparedProblem,
    *,
    allow_basin_restart: bool = False,
    basin_restart_source: str = "",
) -> _OptimizationOutcome:
    def objective_scaled(scaled: np.ndarray) -> tuple[float, np.ndarray]:
        physical = problem.initial + problem.scales * scaled
        value, gradient = _objective_and_gradient(problem, physical)
        return value, gradient * problem.scales

    warm = minimize(
        objective_scaled,
        np.zeros(problem.initial.size, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=_scaled_bounds(problem),
        options=dict(_LBFGSB_OPTIONS),
    )
    warm_parameters = problem.initial + problem.scales * warm.x
    warm_objective, warm_projected_gradient = _stationarity_at(
        problem, warm_parameters
    )
    stages = [
        _optimization_stage_record(
            "L-BFGS-B",
            warm,
            warm_objective,
            warm_projected_gradient,
            accepted=True,
        )
    ]
    stages[0].update(
        {
            "scaled_displacement_inf_from_warm": 0.0,
            "stable_nll_decrease_from_warm": 0.0,
            "polish_basin_guard_valid": True,
        }
    )
    total_iterations = int(getattr(warm, "nit", 0))
    total_evaluations = int(getattr(warm, "nfev", 0))
    chosen = warm
    chosen_parameters = warm_parameters
    chosen_objective = warm_objective
    chosen_projected_gradient = warm_projected_gradient
    chosen_solver_converged = bool(getattr(warm, "success", False))
    polish_basin_valid = True
    polish_scaled_displacement = 0.0
    polish_stable_nll_decrease = 0.0
    method = "L-BFGS-B"

    if (
        not bool(getattr(warm, "success", False))
        or warm_projected_gradient > _STATIONARITY_TOLERANCE
    ):
        _, _, warm_expected = _objective_gradient_and_expected(
            problem, warm_parameters
        )

        def polish_objective_scaled(
            scaled: np.ndarray,
        ) -> tuple[float, np.ndarray]:
            physical = problem.initial + problem.scales * scaled
            _, gradient, expected = _objective_gradient_and_expected(
                problem, physical
            )
            value = _penalized_nll_difference(
                problem,
                warm_parameters,
                physical,
                warm_expected,
                expected,
            )
            return value, gradient * problem.scales

        polish = minimize(
            polish_objective_scaled,
            warm.x,
            method="SLSQP",
            jac=True,
            bounds=_scaled_bounds(problem),
            options=dict(_SLSQP_POLISH_OPTIONS),
        )
        polish_parameters = problem.initial + problem.scales * polish.x
        polish_objective, polish_projected_gradient = _stationarity_at(
            problem, polish_parameters
        )
        _, _, polish_expected = _objective_gradient_and_expected(
            problem, polish_parameters
        )
        polish_objective_difference = _penalized_nll_difference(
            problem,
            warm_parameters,
            polish_parameters,
            warm_expected,
            polish_expected,
        )
        polish_identity_error = abs(
            polish_objective_difference
            - (polish_objective - warm_objective)
        )
        polish_identity_valid = bool(
            np.isfinite(polish_identity_error)
            and polish_identity_error <= _NLL_DIFFERENCE_IDENTITY_TOLERANCE
        )
        total_iterations += int(getattr(polish, "nit", 0))
        total_evaluations += int(getattr(polish, "nfev", 0))
        polish_is_monotone = bool(
            np.isfinite(polish_objective)
            and (
                not np.isfinite(warm_objective)
                or polish_objective <= warm_objective
            )
        )
        (
            candidate_displacement,
            candidate_nll_decrease,
            candidate_basin_valid,
        ) = _polish_basin_metrics(
            problem,
            warm_parameters,
            warm_objective,
            polish_parameters,
            polish_objective,
        )
        candidate_basin_valid = (
            candidate_basin_valid and polish_identity_valid
        )
        polish_basin_valid = candidate_basin_valid
        polish_stage = _optimization_stage_record(
            "SLSQP box polish",
            polish,
            polish_objective,
            polish_projected_gradient,
            accepted=polish_is_monotone and candidate_basin_valid,
        )
        polish_stage.update(
            {
                "scaled_displacement_inf_from_warm": candidate_displacement,
                "stable_nll_decrease_from_warm": candidate_nll_decrease,
                "polish_basin_guard_valid": candidate_basin_valid,
                "stable_difference_identity_error": polish_identity_error,
                "stable_difference_identity_tolerance": (
                    _NLL_DIFFERENCE_IDENTITY_TOLERANCE
                ),
            }
        )
        stages.append(polish_stage)
        if polish_is_monotone and candidate_basin_valid:
            chosen = polish
            chosen_parameters = polish_parameters
            chosen_objective = polish_objective
            chosen_projected_gradient = polish_projected_gradient
            polish_scaled_displacement = candidate_displacement
            polish_stable_nll_decrease = candidate_nll_decrease
            chosen_solver_converged = bool(
                getattr(polish, "success", False)
            )
            method = "L-BFGS-B -> SLSQP box polish"
        elif (
            allow_basin_restart
            and polish_is_monotone
            and polish_identity_valid
            and bool(getattr(polish, "success", False))
        ):
            restart_problem = replace(
                problem, initial=polish_parameters.copy()
            )
            restarted = _optimize_problem(
                restart_problem,
                allow_basin_restart=False,
                basin_restart_source="",
            )
            chain_one_stages = tuple(
                {**stage, "optimizer_chain": 1} for stage in stages
            )
            restart_marker = {
                "method": "guarded full-chain restart",
                "solver_success": polish_stage["solver_success"],
                "status": polish_stage["status"],
                "nll": float(polish_objective),
                "scaled_projected_gradient_inf_norm": float(
                    polish_projected_gradient
                ),
                "accepted": True,
                "scaled_displacement_inf_from_warm": candidate_displacement,
                "stable_nll_decrease_from_warm": candidate_nll_decrease,
                "polish_basin_guard_valid": False,
                "restart_initial_source": basin_restart_source,
                "message": (
                    "guard-excursive monotone SLSQP candidate selected as "
                    "the one allowed full-chain restart initial"
                ),
                "optimizer_chain": "restart",
            }
            chain_two_stages = tuple(
                {**stage, "optimizer_chain": 2}
                for stage in restarted.stages
            )
            return _OptimizationOutcome(
                restarted.result,
                restarted.parameters,
                restarted.solver_converged,
                (
                    "L-BFGS-B -> guard-excursive SLSQP candidate -> "
                    f"guarded full-chain restart -> {restarted.method}"
                ),
                chain_one_stages + (restart_marker,) + chain_two_stages,
                restarted.stationarity_valid,
                restarted.scaled_projected_gradient_inf_norm,
                restarted.scoring_iterations,
                restarted.polish_basin_valid,
                True,
                restarted.polish_scaled_displacement_inf,
                restarted.polish_stable_nll_decrease,
                total_iterations + restarted.iterations,
                total_evaluations + restarted.evaluations,
            )

    scoring_iterations = 0
    if (
        polish_basin_valid
        and chosen_projected_gradient > _STATIONARITY_TOLERANCE
    ):
        for scoring_attempt in range(1, _SCORING_MAX_ITERATIONS + 1):
            scoring = _fisher_scoring_step(
                problem, chosen_parameters, chosen_objective
            )
            total_iterations += 1
            total_evaluations += scoring.evaluations
            (
                candidate_displacement,
                candidate_nll_decrease,
                candidate_basin_valid,
            ) = _polish_basin_metrics(
                problem,
                warm_parameters,
                warm_objective,
                scoring.parameters,
                scoring.objective,
            )
            scoring_accepted = scoring.accepted and candidate_basin_valid
            stages.append(
                {
                    "method": f"expected-Fisher scoring {scoring_attempt}",
                    "solver_success": scoring.accepted,
                    "status": 0 if scoring.accepted else 1,
                    "nll": float(scoring.objective),
                    "scaled_projected_gradient_inf_norm": float(
                        scoring.projected_gradient
                    ),
                    "accepted": scoring_accepted,
                    "damping": scoring.damping,
                    "backtracks": scoring.backtracks,
                    "scaled_displacement_inf_from_warm": (
                        candidate_displacement
                    ),
                    "stable_nll_decrease_from_warm": candidate_nll_decrease,
                    "polish_basin_guard_valid": candidate_basin_valid,
                    "message": (
                        scoring.message
                        if candidate_basin_valid
                        else f"{scoring.message}; rejected by polish basin guard"
                    ),
                }
            )
            if not candidate_basin_valid:
                polish_basin_valid = False
            if not scoring_accepted:
                break
            scoring_iterations += 1
            chosen_parameters = scoring.parameters
            chosen_objective = scoring.objective
            chosen_projected_gradient = scoring.projected_gradient
            polish_scaled_displacement = candidate_displacement
            polish_stable_nll_decrease = candidate_nll_decrease
            if not method.endswith("expected-Fisher scoring"):
                method = f"{method} -> expected-Fisher scoring"
            if chosen_projected_gradient <= _STATIONARITY_TOLERANCE:
                break

    stationarity_valid = bool(
        polish_basin_valid
        and np.isfinite(chosen_objective)
        and chosen_projected_gradient <= _STATIONARITY_TOLERANCE
    )
    return _OptimizationOutcome(
        chosen,
        chosen_parameters,
        chosen_solver_converged,
        method,
        tuple(stages),
        stationarity_valid,
        chosen_projected_gradient,
        scoring_iterations,
        polish_basin_valid,
        False,
        polish_scaled_displacement,
        polish_stable_nll_decrease,
        total_iterations,
        total_evaluations,
    )


def fit_joint_peak_model(
    spectra: Sequence[PublicSpectrum],
    spec: JointPeakSpec,
    calibration: CalibrationConstraint,
    resolution: LinearResolution,
    *,
    initial: Mapping[str, float] | None = None,
    yield_model: YieldModel = "shared_origin_scales",
    warm_start: JointPeakFitResult | None = None,
    warm_start_source: str = "",
    allow_guarded_basin_restart: bool = False,
    guarded_basin_restart_source: str = "",
    gaussian_constraint_observations: Mapping[str, float] | None = None,
) -> JointPeakFitResult:
    """Fit declared components to one or more raw spectra simultaneously.

    ``gaussian_constraint_observations`` may replace the complete observed
    auxiliary-measurement vector while retaining every declared covariance.
    The joint parametric bootstrap uses this to refit independently redrawn
    constraint pseudo-observations; ordinary fits use declared centers.
    """

    if initial is not None and warm_start is not None:
        raise ValueError("initial and warm_start are mutually exclusive")
    if allow_guarded_basin_restart and not guarded_basin_restart_source:
        raise ValueError("guarded basin restart requires an explicit source")

    problem = _prepare_problem(
        spectra,
        spec,
        calibration,
        resolution,
        initial,
        yield_model=yield_model,
        gaussian_constraint_observations=(
            gaussian_constraint_observations
        ),
    )
    warm_start_parameter_names: list[str] = []
    warm_start_skipped_names: list[str] = []
    if warm_start is not None:
        source_parameters = dict(
            zip(warm_start.parameter_names, warm_start.parameter_values)
        )
        for index, name in enumerate(problem.parameter_names):
            if name not in source_parameters:
                continue
            value = float(source_parameters[name])
            if (
                not np.isfinite(value)
                or value < problem.lower[index]
                or value > problem.upper[index]
            ):
                warm_start_skipped_names.append(name)
                continue
            problem.initial[index] = value
            warm_start_parameter_names.append(name)
    optimization = _optimize_problem(
        problem,
        allow_basin_restart=allow_guarded_basin_restart,
        basin_restart_source=guarded_basin_restart_source,
    )
    relaxed_cone_diagnostics = _quadratic_background_cone_diagnostics(
        problem, optimization.parameters
    )
    constrained_fallback_used = any(
        item["cone_margin_counts_per_s_per_keV"] < 0.0
        for item in relaxed_cone_diagnostics.values()
    )
    if constrained_fallback_used:
        optimization = _exact_cone_fallback(problem, optimization)
    fitted = optimization.parameters
    cone_diagnostics = _quadratic_background_cone_diagnostics(
        problem, fitted
    )
    exact_cone_valid = all(
        item["normalized_cone_margin"]
        >= -_EXACT_CONE_FEASIBILITY_RELATIVE_TOLERANCE
        for item in cone_diagnostics.values()
    )
    cone_active_windows = tuple(
        window_name
        for window_name, item in cone_diagnostics.items()
        if item["normalized_cone_margin"]
        <= _EXACT_CONE_ACTIVITY_RELATIVE_TOLERANCE
    )
    expected, jacobian = _model_and_jacobian(problem, fitted, with_jacobian=True)
    assert jacobian is not None

    fisher = jacobian.T @ (jacobian / expected[:, np.newaxis])
    fisher[
        np.ix_(problem.prior_parameter_indices, problem.prior_parameter_indices)
    ] += problem.prior_precision
    fisher = (fisher + fisher.T) / 2.0
    scaled_fisher = (
        problem.scales[:, np.newaxis]
        * fisher
        * problem.scales[np.newaxis, :]
    )
    fisher_rank = int(np.linalg.matrix_rank(scaled_fisher))
    try:
        fisher_condition = float(np.linalg.cond(scaled_fisher))
    except np.linalg.LinAlgError:
        fisher_condition = float("inf")
    fisher_covariance_valid = bool(
        fisher_rank == fitted.size
        and isfinite(fisher_condition)
        and fisher_condition <= 1e14
        and not cone_active_windows
    )
    try:
        scaled_covariance = np.linalg.inv(scaled_fisher)
    except np.linalg.LinAlgError:
        scaled_covariance = np.linalg.pinv(scaled_fisher, rcond=1e-12)
        fisher_covariance_valid = False
    covariance = (
        problem.scales[:, np.newaxis]
        * scaled_covariance
        * problem.scales[np.newaxis, :]
    )
    covariance = (covariance + covariance.T) / 2.0

    active_bounds: list[str] = []
    for name, value, low, high, scale in zip(
        problem.parameter_names,
        fitted,
        problem.lower,
        problem.upper,
        problem.scales,
    ):
        tolerance = _ACTIVE_BOUND_TOLERANCE * max(scale, 1e-12)
        if np.isfinite(low) and value - low <= tolerance:
            active_bounds.append(name)
        elif np.isfinite(high) and high - value <= tolerance:
            active_bounds.append(name)

    if yield_model == "shared_origin_scales":
        line_names = tuple(component.name for component in spec.components)
        line_indices = np.asarray(
            [problem.component_parameter_indices[name] for name in line_names],
            dtype=np.int64,
        )
    else:
        line_names = tuple(
            f"spectrum.{run_index}.{component.name}"
            for run_index in range(len(problem.spectra))
            for component in spec.components
        )
        line_indices = np.asarray(
            [
                problem.run_component_parameter_indices[(run_index, component.name)]
                for run_index in range(len(problem.spectra))
                for component in spec.components
            ],
            dtype=np.int64,
        )
    prior_delta = fitted[problem.prior_parameter_indices] - problem.prior_mean
    data_poisson_nll = _poisson_nll_relative_to_saturated(
        problem.observed, expected
    )
    calibration_prior_deviance = float(
        prior_delta @ problem.prior_precision @ prior_delta
    )
    fitted_nll = data_poisson_nll + 0.5 * calibration_prior_deviance
    # The pure count-data deviance is compared with n_bins - n_parameters.
    # A separate penalized convention adds every declared Gaussian nuisance
    # pseudo-observation and its deviance contribution.
    degrees_of_freedom = int(problem.observed.size - fitted.size)
    penalized_degrees_of_freedom = (
        degrees_of_freedom + problem.prior_parameter_indices.size
    )
    return JointPeakFitResult(
        bool(
            optimization.solver_converged
            and optimization.stationarity_valid
            and exact_cone_valid
            and np.isfinite(fitted_nll)
        ),
        " | ".join(
            f"{stage['method']}: {stage['message']}"
            for stage in optimization.stages
        ),
        problem.parameter_names,
        fitted,
        covariance,
        fisher_rank,
        fisher_condition,
        fisher_covariance_valid,
        "scaled_parameter_coordinates_then_transformed_to_physical",
        tuple(active_bounds),
        cone_diagnostics,
        exact_cone_valid,
        cone_active_windows,
        constrained_fallback_used,
        _EXACT_CONE_FEASIBILITY_RELATIVE_TOLERANCE,
        _EXACT_CONE_ACTIVITY_RELATIVE_TOLERANCE,
        line_names,
        fitted[line_indices],
        covariance[np.ix_(line_indices, line_indices)],
        problem.observed,
        expected,
        problem.spectrum_indices,
        problem.window_names,
        problem.channel_indices,
        data_poisson_nll,
        calibration_prior_deviance,
        fitted_nll,
        _poisson_deviance(problem.observed, expected),
        degrees_of_freedom,
        penalized_degrees_of_freedom,
        optimization.solver_converged,
        optimization.method,
        optimization.stages,
        {
            "warm_method": "L-BFGS-B",
            "warm_options": dict(_LBFGSB_OPTIONS),
            "initialization_source": (
                warm_start_source
                if warm_start is not None
                else "declared model initialization"
            ),
            "warm_start_shared_parameter_count": len(
                warm_start_parameter_names
            ),
            "warm_start_shared_parameter_names": warm_start_parameter_names,
            "warm_start_skipped_parameter_names": warm_start_skipped_names,
            "polish_method": "SLSQP",
            "polish_options": dict(_SLSQP_POLISH_OPTIONS),
            "polish_trigger": "warm solver failure or stationarity-gate failure",
            "polish_acceptance": "objective must not increase",
            "polish_objective": (
                "algebraically constant-shifted penalized NLL evaluated "
                "as a stable per-bin difference from the warm point"
            ),
            "polish_objective_identity_tolerance": (
                _NLL_DIFFERENCE_IDENTITY_TOLERANCE
            ),
            "polish_max_scaled_displacement_inf": (
                _POLISH_MAX_SCALED_DISPLACEMENT_INF
            ),
            "polish_max_stable_nll_decrease": (
                _POLISH_MAX_STABLE_NLL_DECREASE
            ),
            "polish_basin_guard_action": (
                "fail closed unless one explicitly authorized full-chain "
                "restart passes every solver, stationarity, and basin gate"
            ),
            "guarded_basin_restart_allowed": allow_guarded_basin_restart,
            "guarded_basin_restart_source": guarded_basin_restart_source,
            "guarded_basin_restart_maximum_count": 1,
            "guarded_basin_restart_used": optimization.basin_restart_used,
            "scoring_method": "scaled expected Fisher with Cholesky",
            "scoring_max_iterations": _SCORING_MAX_ITERATIONS,
            "scoring_iterations": optimization.scoring_iterations,
            "scoring_damping_ladder": list(_SCORING_DAMPING_LADDER),
            "scoring_damping_diagonal_scale": "max(abs(diag(H_scaled)), 1)",
            "scoring_armijo_coefficient": _SCORING_ARMIJO_COEFFICIENT,
            "scoring_max_backtracks": _SCORING_MAX_BACKTRACKS,
            "scoring_zero_step_tolerance": _SCORING_ZERO_STEP_TOLERANCE,
            "scoring_numerically_flat_nll_tolerance": (
                _SCORING_NUMERICALLY_FLAT_NLL_TOLERANCE
            ),
            "scoring_numerically_flat_gradient_reduction_factor": (
                _SCORING_NUMERICALLY_FLAT_GRADIENT_REDUCTION_FACTOR
            ),
            "quadratic_background_constraint": (
                "exact interval-nonnegative Bernstein cone: endpoints >= 0 "
                "and middle + sqrt(low*high) >= 0"
            ),
            "quadratic_background_relaxed_middle_lower_bound": "unbounded",
            "quadratic_background_constrained_fallback_used": (
                constrained_fallback_used
            ),
            "quadratic_background_exact_cone_parameterization": (
                "middle = slack - sqrt(low*high); slack >= 0"
            ),
            "quadratic_background_exact_cone_options": dict(
                _EXACT_CONE_SLSQP_OPTIONS
            ),
            "quadratic_background_cone_feasibility_relative_tolerance": (
                _EXACT_CONE_FEASIBILITY_RELATIVE_TOLERANCE
            ),
            "quadratic_background_cone_activity_relative_tolerance": (
                _EXACT_CONE_ACTIVITY_RELATIVE_TOLERANCE
            ),
        },
        optimization.scoring_iterations,
        optimization.polish_basin_valid,
        optimization.basin_restart_used,
        optimization.polish_scaled_displacement_inf,
        optimization.polish_stable_nll_decrease,
        optimization.stationarity_valid,
        optimization.scaled_projected_gradient_inf_norm,
        _STATIONARITY_TOLERANCE,
        optimization.iterations,
        optimization.evaluations,
    )


def ratio_values_and_covariance(
    line_names: Sequence[str],
    line_rates: np.ndarray,
    line_rate_covariance: np.ndarray,
    definitions: Sequence[RatioDefinition],
) -> RatioResult:
    """Compute general ratio covariance as ``J Sigma J.T``.

    A definition with identical numerator and denominator is handled as the
    algebraic constant one.  Its Jacobian row is exactly zero, hence its
    variance and covariance with every other ratio are exactly zero.
    """

    names = tuple(line_names)
    rates = np.asarray(line_rates, dtype=np.float64)
    covariance = np.asarray(line_rate_covariance, dtype=np.float64)
    if rates.shape != (len(names),) or covariance.shape != (len(names), len(names)):
        raise ValueError("line-rate vector and covariance do not match line names")
    if not np.isfinite(rates).all() or not np.isfinite(covariance).all():
        raise ValueError("line rates and covariance must be finite")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-14):
        raise ValueError("line-rate covariance must be symmetric")
    index = {name: position for position, name in enumerate(names)}
    if len(index) != len(names):
        raise ValueError("line names must be unique")
    values = np.empty(len(definitions), dtype=np.float64)
    jacobian = np.zeros((len(definitions), len(names)), dtype=np.float64)
    labels: list[str] = []
    for row, definition in enumerate(definitions):
        if definition.numerator not in index or definition.denominator not in index:
            raise KeyError(f"unknown ratio line in {definition.name!r}")
        numerator = index[definition.numerator]
        denominator = index[definition.denominator]
        labels.append(definition.name)
        if numerator == denominator:
            values[row] = 1.0
            continue
        if rates[denominator] <= 0:
            raise ValueError(f"ratio {definition.name!r} has a nonpositive denominator")
        values[row] = rates[numerator] / rates[denominator]
        jacobian[row, numerator] = 1.0 / rates[denominator]
        jacobian[row, denominator] = -rates[numerator] / rates[denominator] ** 2
    ratio_covariance = jacobian @ covariance @ jacobian.T
    ratio_covariance = (ratio_covariance + ratio_covariance.T) / 2.0
    return RatioResult(tuple(labels), values, ratio_covariance, jacobian)


def ratios_from_fit(
    result: JointPeakFitResult, definitions: Sequence[RatioDefinition]
) -> RatioResult:
    """Convenience wrapper for a fitted line-rate covariance."""

    return ratio_values_and_covariance(
        result.line_names,
        result.line_rates_counts_per_s,
        result.line_rate_covariance,
        definitions,
    )


def _profile_nll_at_ratio(
    problem: _PreparedProblem,
    mle: np.ndarray,
    numerator_index: int,
    denominator_index: int,
    ratio: float,
) -> _ProfileOptimizationOutcome:
    if ratio < 0 or not isfinite(ratio):
        return _ProfileOptimizationOutcome(
            float("inf"),
            False,
            False,
            False,
            float("inf"),
            0.0,
            False,
            float("inf"),
            "profile ratio is outside its physical domain",
        )
    (
        coordinate_initial,
        coordinate_lower,
        coordinate_upper,
        coordinate_scales,
    ) = _quadratic_cone_coordinate_system(problem, mle)
    keep = np.asarray(
        [index for index in range(mle.size) if index != numerator_index],
        dtype=np.int64,
    )
    denominator_reduced = int(np.flatnonzero(keep == denominator_index)[0])
    reduced_initial = coordinate_initial[keep].copy()
    reduced_scales = coordinate_scales[keep]
    reduced_lower = coordinate_lower[keep]
    reduced_upper = coordinate_upper[keep]
    reference_nll, _, reference_expected = _objective_gradient_and_expected(
        problem, mle
    )

    def physical_and_gradient(
        scaled: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        reduced = reduced_initial + reduced_scales * scaled
        coordinates = np.empty_like(mle)
        coordinates[keep] = reduced
        coordinates[numerator_index] = (
            ratio * coordinates[denominator_index]
        )
        physical = _physical_from_quadratic_cone_coordinates(
            problem, coordinates
        )
        nll, physical_gradient, expected = _objective_gradient_and_expected(
            problem, physical
        )
        coordinate_gradient = _gradient_in_quadratic_cone_coordinates(
            problem, coordinates, physical_gradient
        )
        reduced_gradient = coordinate_gradient[keep]
        reduced_gradient[denominator_reduced] += (
            ratio * coordinate_gradient[numerator_index]
        )
        return physical, nll, expected, reduced_gradient

    def objective_scaled(scaled: np.ndarray) -> tuple[float, np.ndarray]:
        physical, _, expected, reduced_gradient = physical_and_gradient(
            scaled
        )
        value = _penalized_nll_difference(
            problem,
            mle,
            physical,
            reference_expected,
            expected,
        )
        return value, reduced_gradient * reduced_scales

    bounds = _scaled_bounds_from_reference(
        reduced_initial, reduced_lower, reduced_upper, reduced_scales
    )
    optimization = minimize(
        objective_scaled,
        np.zeros(reduced_initial.size, dtype=np.float64),
        method="SLSQP",
        jac=True,
        bounds=bounds,
        options=dict(_PROFILE_SLSQP_OPTIONS),
    )
    physical, nll, expected, reduced_gradient = physical_and_gradient(
        np.asarray(optimization.x, dtype=np.float64)
    )
    reduced = reduced_initial + reduced_scales * np.asarray(
        optimization.x, dtype=np.float64
    )
    projected = _scaled_projected_gradient_for_bounds(
        reduced,
        reduced_gradient,
        reduced_lower,
        reduced_upper,
        reduced_scales,
    )
    scaled_kkt = float(np.max(np.abs(projected), initial=0.0))
    difference = _penalized_nll_difference(
        problem, mle, physical, reference_expected, expected
    )
    identity_error = abs(difference - (nll - reference_nll))
    cone_diagnostics = _quadratic_background_cone_diagnostics(
        problem, physical
    )
    exact_cone_valid = all(
        item["normalized_cone_margin"]
        >= -_EXACT_CONE_FEASIBILITY_RELATIVE_TOLERANCE
        for item in cone_diagnostics.values()
    )
    solver_converged = bool(getattr(optimization, "success", False))
    stationarity_valid = bool(
        np.isfinite(nll)
        and np.isfinite(scaled_kkt)
        and scaled_kkt <= _PROFILE_STATIONARITY_TOLERANCE
    )
    success = bool(
        solver_converged
        and stationarity_valid
        and exact_cone_valid
        and identity_error <= _NLL_DIFFERENCE_IDENTITY_TOLERANCE
    )
    return _ProfileOptimizationOutcome(
        float(nll),
        success,
        solver_converged,
        stationarity_valid,
        scaled_kkt,
        0.0,
        exact_cone_valid,
        identity_error,
        (
            f"SLSQP: {getattr(optimization, 'message', '')}; "
            f"scaled KKT inf-norm={scaled_kkt:.6g}; "
            f"exact cone valid={exact_cone_valid}"
        ),
    )


def _validate_profile_interval_options(
    confidence_level: float,
    max_evaluations: int,
    base_nll_consistency_tolerance: float,
) -> None:
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence level must lie between 0.5 and 1")
    if max_evaluations < 8:
        raise ValueError("max_evaluations must be at least eight")
    if (
        not isfinite(base_nll_consistency_tolerance)
        or base_nll_consistency_tolerance <= 0.0
    ):
        raise ValueError("base-NLL consistency tolerance must be positive")


def _profile_diagnostic_fields(
    outcomes: Mapping[float, _ProfileOptimizationOutcome],
    *,
    profile_base_nll: float,
    fit_base_nll: float,
    base_nll_consistency_tolerance: float,
    linear_constraint: bool,
) -> dict[str, float | int]:
    finite_kkt = [
        outcome.scaled_kkt_inf_norm
        for outcome in outcomes.values()
        if np.isfinite(outcome.scaled_kkt_inf_norm)
    ]
    finite_identity_errors = [
        outcome.stable_difference_identity_error
        for outcome in outcomes.values()
        if np.isfinite(outcome.stable_difference_identity_error)
    ]
    fields: dict[str, float | int] = {
        "profile_base_penalized_nll": profile_base_nll,
        "fit_penalized_nll": fit_base_nll,
        "base_nll_difference": profile_base_nll - fit_base_nll,
        "base_nll_consistency_tolerance": base_nll_consistency_tolerance,
        "inner_solver_failures": sum(
            not outcome.success for outcome in outcomes.values()
        ),
        "maximum_scaled_kkt_inf_norm": (
            max(finite_kkt) if finite_kkt else float("nan")
        ),
        "inner_stationarity_tolerance": _PROFILE_STATIONARITY_TOLERANCE,
        "maximum_stable_nll_difference_identity_error": (
            max(finite_identity_errors)
            if finite_identity_errors
            else float("nan")
        ),
        "stable_nll_difference_identity_tolerance": (
            _NLL_DIFFERENCE_IDENTITY_TOLERANCE
        ),
        "exact_cone_invalid_inner_solves": sum(
            not outcome.exact_cone_valid for outcome in outcomes.values()
        ),
        "exact_cone_feasibility_relative_tolerance": (
            _EXACT_CONE_FEASIBILITY_RELATIVE_TOLERANCE
        ),
    }
    if linear_constraint:
        finite_residuals = [
            outcome.linear_constraint_relative_residual
            for outcome in outcomes.values()
            if np.isfinite(outcome.linear_constraint_relative_residual)
        ]
        fields.update(
            {
                "maximum_linear_constraint_relative_residual": (
                    max(finite_residuals)
                    if finite_residuals
                    else float("nan")
                ),
                "linear_constraint_relative_tolerance": (
                    _PROFILE_LINEAR_CONSTRAINT_RELATIVE_TOLERANCE
                ),
            }
        )
    return fields


def _profile_interval_from_inner_solves(
    *,
    ratio_name: str,
    estimate: float,
    confidence_level: float,
    boundary: bool,
    max_evaluations: int,
    base_nll_consistency_tolerance: float,
    fit_base_nll: float,
    inner_solve: Callable[[float], _ProfileOptimizationOutcome],
    linear_constraint: bool,
    infer_upper_limit_from_zero: bool,
) -> ProfileInterval:
    """Apply common profile caching, bracketing, and fail-closed diagnostics."""

    threshold = 0.5 * float(
        chi2.ppf(
            2.0 * confidence_level - 1.0 if boundary else confidence_level,
            1,
        )
    )
    base_outcome = inner_solve(estimate)
    evaluations = 1
    cache = {float(estimate): base_outcome}
    profile_label = "linear-ratio profile" if linear_constraint else "profile"

    def diagnostic_fields(
        profile_base_nll: float,
    ) -> dict[str, float | int]:
        return _profile_diagnostic_fields(
            cache,
            profile_base_nll=profile_base_nll,
            fit_base_nll=fit_base_nll,
            base_nll_consistency_tolerance=base_nll_consistency_tolerance,
            linear_constraint=linear_constraint,
        )

    if not base_outcome.success:
        return ProfileInterval(
            ratio_name,
            estimate,
            confidence_level,
            "failed",
            float("nan"),
            float("nan"),
            threshold,
            evaluations,
            f"{profile_label} inner solve failed at the fitted ratio: "
            + base_outcome.message,
            **diagnostic_fields(base_outcome.nll),
        )
    profile_base_nll = base_outcome.nll
    base_nll_difference = profile_base_nll - fit_base_nll
    if abs(base_nll_difference) > base_nll_consistency_tolerance:
        return ProfileInterval(
            ratio_name,
            estimate,
            confidence_level,
            "failed",
            float("nan"),
            float("nan"),
            threshold,
            evaluations,
            (
                f"tight {profile_label} base is inconsistent with the fitted "
                f"NLL: delta={base_nll_difference:.6g}, tolerance="
                f"{base_nll_consistency_tolerance:.6g}"
            ),
            **diagnostic_fields(profile_base_nll),
        )
    failure_messages: list[str] = []

    def delta(value: float) -> float:
        nonlocal evaluations
        key = float(value)
        if key not in cache:
            if evaluations >= max_evaluations:
                return float("nan")
            cache[key] = inner_solve(key)
            evaluations += 1
        outcome = cache[key]
        if not outcome.success:
            failure_messages.append(f"ratio={key:.12g}: {outcome.message}")
            return float("nan")
        return outcome.nll - profile_base_nll - threshold

    def bisect(left: float, right: float) -> float:
        f_left = delta(left)
        f_right = delta(right)
        if (
            not np.isfinite(f_left)
            or not np.isfinite(f_right)
            or f_left * f_right > 0
        ):
            return float("nan")
        for _ in range(40):
            if evaluations >= max_evaluations:
                break
            middle = 0.5 * (left + right)
            f_middle = delta(middle)
            if not np.isfinite(f_middle):
                break
            if (
                abs(f_middle) < 2e-3
                or right - left <= 1e-5 * max(1.0, estimate)
            ):
                return middle
            if f_left * f_middle <= 0:
                right = middle
                f_right = f_middle
            else:
                left = middle
                f_left = f_middle
        return 0.5 * (left + right)

    zero_delta = delta(0.0)
    if not np.isfinite(zero_delta) and failure_messages:
        return ProfileInterval(
            ratio_name,
            estimate,
            confidence_level,
            "failed",
            float("nan"),
            float("nan"),
            threshold,
            evaluations,
            f"{profile_label} inner solve failed at zero ratio: "
            + failure_messages[0],
            **diagnostic_fields(profile_base_nll),
        )
    is_upper_limit = boundary or (
        infer_upper_limit_from_zero
        and np.isfinite(zero_delta)
        and zero_delta <= 0
    )
    if is_upper_limit and not boundary:
        threshold = 0.5 * float(
            chi2.ppf(2.0 * confidence_level - 1.0, 1)
        )
        zero_delta = delta(0.0)
    lower = 0.0
    if not is_upper_limit and estimate > 0:
        if np.isfinite(zero_delta) and zero_delta >= 0:
            lower = bisect(0.0, estimate)
        elif infer_upper_limit_from_zero:
            is_upper_limit = True

    upper_trial = max(estimate * 1.5, estimate + 0.05, 0.05)
    upper_delta = delta(upper_trial)
    while (
        np.isfinite(upper_delta)
        and upper_delta < 0
        and evaluations < max_evaluations - 1
    ):
        upper_trial *= 2.0
        upper_delta = delta(upper_trial)
    upper = (
        bisect(estimate, upper_trial)
        if np.isfinite(upper_delta) and upper_delta >= 0
        else float("nan")
    )
    if not np.isfinite(lower) or not np.isfinite(upper):
        budget_suffix = (
            " within the evaluation budget" if not linear_constraint else ""
        )
        return ProfileInterval(
            ratio_name,
            estimate,
            confidence_level,
            "failed",
            float("nan"),
            float("nan"),
            threshold,
            evaluations,
            (
                f"{profile_label} threshold was not bracketed{budget_suffix}"
                + (
                    "; first inner failure: " + failure_messages[0]
                    if failure_messages
                    else ""
                )
            ),
            **diagnostic_fields(profile_base_nll),
        )
    return ProfileInterval(
        ratio_name,
        estimate,
        confidence_level,
        "upper_limit" if is_upper_limit else "two_sided",
        0.0 if is_upper_limit else lower,
        upper,
        threshold,
        evaluations,
        (
            (
                "one-sided boundary profile of all run yields and shared "
                "nuisances"
            )
            if linear_constraint and is_upper_limit
            else (
                "two-sided profile of all run yields and shared nuisances"
                if linear_constraint
                else (
                    "one-sided boundary construction"
                    if is_upper_limit
                    else "two-sided profile-likelihood interval"
                )
            )
        ),
        **diagnostic_fields(profile_base_nll),
    )


def profile_ratio_interval(
    spectra: Sequence[PublicSpectrum],
    spec: JointPeakSpec,
    calibration: CalibrationConstraint,
    resolution: LinearResolution,
    result: JointPeakFitResult,
    definition: RatioDefinition,
    *,
    confidence_level: float = 0.95,
    max_evaluations: int = 36,
    base_nll_consistency_tolerance: float = (
        _PROFILE_BASE_NLL_CONSISTENCY_TOLERANCE
    ),
) -> ProfileInterval:
    """Profile all nuisance parameters for a ratio interval or upper limit."""

    expected_shared_line_names = tuple(
        component.name for component in spec.components
    )
    if result.line_names != expected_shared_line_names:
        raise ValueError(
            "profile_ratio_interval supports only shared_origin_scales fits "
            "whose line names match spec.components; use "
            "profile_linear_ratio_interval with yield_model='independent_runs' "
            "for independent-run fits"
        )
    _validate_profile_interval_options(
        confidence_level,
        max_evaluations,
        base_nll_consistency_tolerance,
    )
    if definition.numerator == definition.denominator:
        return ProfileInterval(
            definition.name,
            1.0,
            confidence_level,
            "self_ratio",
            1.0,
            1.0,
            0.0,
            0,
            "same fitted variable divided by itself",
        )
    if not result.success:
        return ProfileInterval(
            definition.name,
            float("nan"),
            confidence_level,
            "failed",
            float("nan"),
            float("nan"),
            float("nan"),
            0,
            "cannot profile an unsuccessful fit",
        )
    problem = _prepare_problem(
        spectra,
        spec,
        calibration,
        resolution,
        dict(zip(result.parameter_names, result.parameter_values)),
    )
    numerator_index = problem.component_parameter_indices[definition.numerator]
    denominator_index = problem.component_parameter_indices[definition.denominator]
    numerator = result.parameter_values[numerator_index]
    denominator = result.parameter_values[denominator_index]
    estimate = numerator / denominator
    if denominator <= 0:
        return ProfileInterval(
            definition.name,
            estimate,
            confidence_level,
            "failed",
            float("nan"),
            float("nan"),
            float("nan"),
            0,
            "profile denominator is nonpositive",
        )
    active_name = problem.parameter_names[numerator_index]
    boundary = active_name in result.active_bounds or numerator <= 1e-10
    return _profile_interval_from_inner_solves(
        ratio_name=definition.name,
        estimate=float(estimate),
        confidence_level=confidence_level,
        boundary=boundary,
        max_evaluations=max_evaluations,
        base_nll_consistency_tolerance=base_nll_consistency_tolerance,
        fit_base_nll=result.penalized_nll,
        inner_solve=lambda value: _profile_nll_at_ratio(
            problem,
            result.parameter_values,
            numerator_index,
            denominator_index,
            value,
        ),
        linear_constraint=False,
        infer_upper_limit_from_zero=True,
    )

def _profile_equality_multiplier(
    coordinates: np.ndarray,
    objective_gradient: np.ndarray,
    constraint_gradient: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    scales: np.ndarray,
) -> float:
    """Estimate the equality multiplier from interior, then active variables."""

    interior = np.ones(coordinates.size, dtype=bool)
    for index, (value, low, high, scale) in enumerate(
        zip(coordinates, lower, upper, scales)
    ):
        bound_tolerance = _ACTIVE_BOUND_TOLERANCE * max(scale, 1.0e-12)
        interior[index] = not (
            (np.isfinite(low) and value - low <= bound_tolerance)
            or (np.isfinite(high) and high - value <= bound_tolerance)
        )
    scaled_objective = objective_gradient * scales
    scaled_constraint = constraint_gradient * scales
    nonzero = np.abs(scaled_constraint) > np.finfo(np.float64).eps
    multiplier_mask = interior & nonzero
    if not np.any(multiplier_mask):
        multiplier_mask = nonzero
    denominator = float(
        scaled_constraint[multiplier_mask]
        @ scaled_constraint[multiplier_mask]
    )
    if denominator <= 0.0 or not np.isfinite(denominator):
        return float("inf")
    return -float(
        scaled_constraint[multiplier_mask]
        @ scaled_objective[multiplier_mask]
    ) / denominator


def _profile_equality_kkt_inf_norm(
    coordinates: np.ndarray,
    objective_gradient: np.ndarray,
    constraint_gradient: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    scales: np.ndarray,
) -> float:
    """Return a box/equality KKT residual after estimating one multiplier."""

    multiplier = _profile_equality_multiplier(
        coordinates,
        objective_gradient,
        constraint_gradient,
        lower,
        upper,
        scales,
    )
    if not np.isfinite(multiplier):
        return float("inf")
    lagrangian_gradient = (
        objective_gradient + multiplier * constraint_gradient
    )
    projected = _scaled_projected_gradient_for_bounds(
        coordinates,
        lagrangian_gradient,
        lower,
        upper,
        scales,
    )
    return float(np.max(np.abs(projected), initial=0.0))


def _repair_linear_profile_stationarity(
    problem: _PreparedProblem,
    coordinate_initial: np.ndarray,
    coordinate_lower: np.ndarray,
    coordinate_upper: np.ndarray,
    coordinate_scales: np.ndarray,
    constraint_coefficients: np.ndarray,
    constraint_normalization: float,
    scaled_initial: np.ndarray,
) -> tuple[np.ndarray, int, str]:
    """Apply equality-tangent expected-Fisher steps after profile SLSQP."""

    scaled = np.asarray(scaled_initial, dtype=np.float64).copy()

    def assess(
        trial_scaled: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        float,
        np.ndarray,
        np.ndarray,
        float,
        float,
    ]:
        coordinates = coordinate_initial + coordinate_scales * trial_scaled
        physical = _physical_from_quadratic_cone_coordinates(
            problem, coordinates
        )
        nll, physical_gradient, expected = _objective_gradient_and_expected(
            problem, physical
        )
        coordinate_gradient = _gradient_in_quadratic_cone_coordinates(
            problem, coordinates, physical_gradient
        )
        constraint_gradient = (
            _gradient_in_quadratic_cone_coordinates(
                problem, coordinates, constraint_coefficients
            )
            / constraint_normalization
        )
        constraint_value = float(
            constraint_coefficients @ physical / constraint_normalization
        )
        kkt = _profile_equality_kkt_inf_norm(
            coordinates,
            coordinate_gradient,
            constraint_gradient,
            coordinate_lower,
            coordinate_upper,
            coordinate_scales,
        )
        return (
            coordinates,
            physical,
            nll,
            expected,
            coordinate_gradient,
            constraint_value,
            kkt,
        )

    for iteration in range(_PROFILE_SCORING_MAX_ITERATIONS + 1):
        (
            coordinates,
            physical,
            nll,
            expected,
            coordinate_gradient,
            constraint_value,
            kkt,
        ) = assess(scaled)
        if (
            np.isfinite(kkt)
            and kkt <= _PROFILE_STATIONARITY_TOLERANCE
            and abs(constraint_value)
            <= _PROFILE_LINEAR_CONSTRAINT_RELATIVE_TOLERANCE
        ):
            return scaled, iteration, "profile stationarity gate passed"
        if iteration == _PROFILE_SCORING_MAX_ITERATIONS:
            break

        constraint_gradient = (
            _gradient_in_quadratic_cone_coordinates(
                problem, coordinates, constraint_coefficients
            )
            / constraint_normalization
        )
        multiplier = _profile_equality_multiplier(
            coordinates,
            coordinate_gradient,
            constraint_gradient,
            coordinate_lower,
            coordinate_upper,
            coordinate_scales,
        )
        if not np.isfinite(multiplier):
            return scaled, iteration, "profile equality multiplier failed"
        scaled_gradient = coordinate_gradient * coordinate_scales
        scaled_constraint = constraint_gradient * coordinate_scales
        scaled_lagrangian_gradient = (
            scaled_gradient + multiplier * scaled_constraint
        )
        fixed = np.zeros(coordinates.size, dtype=bool)
        for index, (value, low, high, scale, gradient) in enumerate(
            zip(
                coordinates,
                coordinate_lower,
                coordinate_upper,
                coordinate_scales,
                scaled_lagrangian_gradient,
            )
        ):
            tolerance = _ACTIVE_BOUND_TOLERANCE * max(scale, 1.0e-12)
            fixed[index] = bool(
                (
                    np.isfinite(low)
                    and value - low <= tolerance
                    and gradient > 0.0
                )
                or (
                    np.isfinite(high)
                    and high - value <= tolerance
                    and gradient < 0.0
                )
            )
        free = np.flatnonzero(~fixed)
        if free.size == 0:
            return scaled, iteration, "profile scoring has no free parameters"

        scaled_fisher = (
            _scaled_expected_fisher_in_quadratic_cone_coordinates(
                problem, coordinates, coordinate_scales
            )
        )
        free_fisher = scaled_fisher[np.ix_(free, free)]
        free_constraint = scaled_constraint[free]
        if not np.isfinite(free_fisher).all():
            return scaled, iteration, "profile expected Fisher is non-finite"
        direction = None
        diagonal_scale = np.maximum(
            np.abs(np.diag(free_fisher)), 1.0
        )
        for damping in _SCORING_DAMPING_LADDER:
            trial_fisher = free_fisher.copy()
            if damping > 0.0:
                trial_fisher.flat[:: trial_fisher.shape[0] + 1] += (
                    damping * diagonal_scale
                )
            if np.linalg.norm(free_constraint) > np.finfo(np.float64).eps:
                system = np.block(
                    [
                        [
                            trial_fisher,
                            free_constraint[:, np.newaxis],
                        ],
                        [
                            free_constraint[np.newaxis, :],
                            np.zeros((1, 1), dtype=np.float64),
                        ],
                    ]
                )
                right_hand_side = np.concatenate(
                    (-scaled_gradient[free], [-constraint_value])
                )
            else:
                system = trial_fisher
                right_hand_side = -scaled_gradient[free]
            try:
                solution = np.linalg.solve(system, right_hand_side)
            except np.linalg.LinAlgError:
                continue
            free_direction = solution[: free.size]
            if np.isfinite(free_direction).all():
                direction = np.zeros(coordinates.size, dtype=np.float64)
                direction[free] = free_direction
                break
        if direction is None:
            return scaled, iteration, "profile expected-Fisher solve failed"
        direction[
            np.abs(direction)
            <= 1.0e-12
            * max(1.0, float(np.max(np.abs(direction), initial=0.0)))
        ] = 0.0
        directional_derivative = float(scaled_gradient @ direction)
        if (
            not np.isfinite(directional_derivative)
            or directional_derivative >= 0.0
        ):
            return scaled, iteration, "profile scoring direction is not descent"

        maximum_step = 1.0
        for value, low, high, scale, component in zip(
            coordinates,
            coordinate_lower,
            coordinate_upper,
            coordinate_scales,
            direction,
        ):
            if component > 0.0 and np.isfinite(high):
                maximum_step = min(
                    maximum_step,
                    (high - value) / (scale * component),
                )
            elif component < 0.0 and np.isfinite(low):
                maximum_step = min(
                    maximum_step,
                    (low - value) / (scale * component),
                )
        if not np.isfinite(maximum_step) or maximum_step <= 0.0:
            return scaled, iteration, "profile scoring step is box-infeasible"

        accepted = False
        for backtracks in range(_SCORING_MAX_BACKTRACKS + 1):
            step = maximum_step * 0.5**backtracks
            candidate_scaled = scaled + step * direction
            (
                _,
                candidate_physical,
                _,
                candidate_expected,
                _,
                candidate_constraint,
                candidate_kkt,
            ) = assess(candidate_scaled)
            difference = _penalized_nll_difference(
                problem,
                physical,
                candidate_physical,
                expected,
                candidate_expected,
            )
            strict_armijo = bool(
                np.isfinite(difference)
                and difference <= 0.0
                and difference
                <= _SCORING_ARMIJO_COEFFICIENT
                * step
                * directional_derivative
            )
            numerically_flat_repair = bool(
                np.isfinite(difference)
                and abs(difference)
                <= _SCORING_NUMERICALLY_FLAT_NLL_TOLERANCE
                and candidate_kkt
                <= max(
                    _PROFILE_STATIONARITY_TOLERANCE,
                    _SCORING_NUMERICALLY_FLAT_GRADIENT_REDUCTION_FACTOR
                    * kkt,
                )
            )
            if (
                (strict_armijo or numerically_flat_repair)
                and abs(candidate_constraint)
                <= _PROFILE_LINEAR_CONSTRAINT_RELATIVE_TOLERANCE
            ):
                scaled = candidate_scaled
                accepted = True
                break
        if not accepted:
            return scaled, iteration, "profile scoring line search failed"
    return (
        scaled,
        _PROFILE_SCORING_MAX_ITERATIONS,
        "profile stationarity gate failed after scoring cap",
    )


def _profile_nll_at_linear_ratio(
    problem: _PreparedProblem,
    mle: np.ndarray,
    numerator_coefficients: np.ndarray,
    denominator_coefficients: np.ndarray,
    ratio: float,
) -> _ProfileOptimizationOutcome:
    if ratio < 0 or not isfinite(ratio):
        return _ProfileOptimizationOutcome(
            float("inf"),
            False,
            False,
            False,
            float("inf"),
            float("inf"),
            False,
            float("inf"),
            "linear profile ratio is outside its physical domain",
        )
    constraint_coefficients = numerator_coefficients - ratio * denominator_coefficients
    (
        coordinate_initial,
        coordinate_lower,
        coordinate_upper,
        coordinate_scales,
    ) = _quadratic_cone_coordinate_system(problem, mle)
    reference_nll, _, reference_expected = _objective_gradient_and_expected(
        problem, mle
    )
    constraint_normalization = max(
        float(
            np.abs(constraint_coefficients)
            @ np.maximum(np.abs(mle), problem.scales)
        ),
        1.0e-12,
    )

    def physical_and_gradient(
        scaled: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        coordinates = coordinate_initial + coordinate_scales * scaled
        physical = _physical_from_quadratic_cone_coordinates(
            problem, coordinates
        )
        nll, physical_gradient, expected = _objective_gradient_and_expected(
            problem, physical
        )
        coordinate_gradient = _gradient_in_quadratic_cone_coordinates(
            problem, coordinates, physical_gradient
        )
        return physical, nll, expected, coordinate_gradient

    def objective_scaled(scaled: np.ndarray) -> tuple[float, np.ndarray]:
        physical, _, expected, coordinate_gradient = physical_and_gradient(
            scaled
        )
        value = _penalized_nll_difference(
            problem,
            mle,
            physical,
            reference_expected,
            expected,
        )
        return value, coordinate_gradient * coordinate_scales

    def constraint_scaled(scaled: np.ndarray) -> float:
        physical = _physical_from_quadratic_cone_coordinates(
            problem, coordinate_initial + coordinate_scales * scaled
        )
        return float(
            constraint_coefficients @ physical / constraint_normalization
        )

    def constraint_jacobian(scaled: np.ndarray) -> np.ndarray:
        coordinates = coordinate_initial + coordinate_scales * scaled
        coordinate_gradient = _gradient_in_quadratic_cone_coordinates(
            problem, coordinates, constraint_coefficients
        )
        return (
            coordinate_gradient
            * coordinate_scales
            / constraint_normalization
        )

    bounds = _scaled_bounds_from_reference(
        coordinate_initial,
        coordinate_lower,
        coordinate_upper,
        coordinate_scales,
    )
    scaled = np.zeros(mle.size, dtype=np.float64)
    solver_converged = True
    total_scoring_iterations = 0
    chain_messages: list[str] = []
    optimization: object | None = None
    for chain in range(1, _PROFILE_SLSQP_MAX_CHAINS + 1):
        optimization = minimize(
            objective_scaled,
            scaled,
            method="SLSQP",
            jac=True,
            bounds=bounds,
            constraints={
                "type": "eq",
                "fun": constraint_scaled,
                "jac": constraint_jacobian,
            },
            options=dict(_PROFILE_SLSQP_OPTIONS),
        )
        scaled = np.asarray(optimization.x, dtype=np.float64)
        chain_solver_converged = bool(
            getattr(optimization, "success", False)
        )
        solver_converged = solver_converged and chain_solver_converged
        if not chain_solver_converged:
            chain_messages.append(
                f"chain {chain} SLSQP failed: "
                f"{getattr(optimization, 'message', '')}"
            )
            break
        scaled, scoring_iterations, scoring_message = (
            _repair_linear_profile_stationarity(
                problem,
                coordinate_initial,
                coordinate_lower,
                coordinate_upper,
                coordinate_scales,
                constraint_coefficients,
                constraint_normalization,
                scaled,
            )
        )
        total_scoring_iterations += scoring_iterations
        chain_messages.append(
            f"chain {chain}: {scoring_message}"
        )
        coordinates_for_gate = (
            coordinate_initial + coordinate_scales * scaled
        )
        physical_for_gate, _, _, gradient_for_gate = (
            physical_and_gradient(scaled)
        )
        constraint_gradient_for_gate = (
            _gradient_in_quadratic_cone_coordinates(
                problem,
                coordinates_for_gate,
                constraint_coefficients,
            )
            / constraint_normalization
        )
        gate_kkt = _profile_equality_kkt_inf_norm(
            coordinates_for_gate,
            gradient_for_gate,
            constraint_gradient_for_gate,
            coordinate_lower,
            coordinate_upper,
            coordinate_scales,
        )
        gate_constraint = abs(
            float(constraint_coefficients @ physical_for_gate)
        ) / constraint_normalization
        if (
            gate_kkt <= _PROFILE_STATIONARITY_TOLERANCE
            and gate_constraint
            <= _PROFILE_LINEAR_CONSTRAINT_RELATIVE_TOLERANCE
        ):
            break
    assert optimization is not None
    coordinates = coordinate_initial + coordinate_scales * scaled
    physical, nll, expected, coordinate_gradient = physical_and_gradient(
        scaled
    )
    normalized_constraint_gradient = (
        _gradient_in_quadratic_cone_coordinates(
            problem, coordinates, constraint_coefficients
        )
        / constraint_normalization
    )
    constraint_residual = abs(
        float(constraint_coefficients @ physical)
    ) / constraint_normalization
    scaled_kkt = _profile_equality_kkt_inf_norm(
        coordinates,
        coordinate_gradient,
        normalized_constraint_gradient,
        coordinate_lower,
        coordinate_upper,
        coordinate_scales,
    )
    difference = _penalized_nll_difference(
        problem, mle, physical, reference_expected, expected
    )
    identity_error = abs(difference - (nll - reference_nll))
    cone_diagnostics = _quadratic_background_cone_diagnostics(
        problem, physical
    )
    exact_cone_valid = all(
        item["normalized_cone_margin"]
        >= -_EXACT_CONE_FEASIBILITY_RELATIVE_TOLERANCE
        for item in cone_diagnostics.values()
    )
    stationarity_valid = bool(
        np.isfinite(nll)
        and np.isfinite(scaled_kkt)
        and scaled_kkt <= _PROFILE_STATIONARITY_TOLERANCE
        and constraint_residual
        <= _PROFILE_LINEAR_CONSTRAINT_RELATIVE_TOLERANCE
    )
    success = bool(
        solver_converged
        and stationarity_valid
        and exact_cone_valid
        and identity_error <= _NLL_DIFFERENCE_IDENTITY_TOLERANCE
    )
    return _ProfileOptimizationOutcome(
        float(nll),
        success,
        solver_converged,
        stationarity_valid,
        scaled_kkt,
        constraint_residual,
        exact_cone_valid,
        identity_error,
        (
            f"final SLSQP: {getattr(optimization, 'message', '')}; "
            f"scaled KKT inf-norm={scaled_kkt:.6g}; normalized equality "
            f"residual={constraint_residual:.6g}; exact cone valid="
            f"{exact_cone_valid}; total scoring iterations="
            f"{total_scoring_iterations}; "
            + " | ".join(chain_messages)
        ),
    )


def profile_linear_ratio_interval(
    spectra: Sequence[PublicSpectrum],
    spec: JointPeakSpec,
    calibration: CalibrationConstraint,
    resolution: LinearResolution,
    result: JointPeakFitResult,
    numerator_line_weights: Sequence[float],
    denominator_line_weights: Sequence[float],
    *,
    ratio_name: str,
    yield_model: YieldModel = "shared_origin_scales",
    confidence_level: float = 0.95,
    max_evaluations: int = 28,
    base_nll_consistency_tolerance: float = (
        _PROFILE_BASE_NLL_CONSISTENCY_TOLERANCE
    ),
) -> ProfileInterval:
    """Profile a ratio of two linear combinations of fitted line rates."""

    _validate_profile_interval_options(
        confidence_level,
        max_evaluations,
        base_nll_consistency_tolerance,
    )
    numerator_weights = np.asarray(numerator_line_weights, dtype=np.float64)
    denominator_weights = np.asarray(denominator_line_weights, dtype=np.float64)
    if (
        numerator_weights.shape != (len(result.line_names),)
        or denominator_weights.shape != numerator_weights.shape
        or not np.isfinite(numerator_weights).all()
        or not np.isfinite(denominator_weights).all()
    ):
        raise ValueError("linear-ratio weights must match fitted line names")
    if not result.success:
        return ProfileInterval(
            ratio_name,
            float("nan"),
            confidence_level,
            "failed",
            float("nan"),
            float("nan"),
            float("nan"),
            0,
            "cannot profile an unsuccessful fit",
            fit_penalized_nll=result.penalized_nll,
            base_nll_consistency_tolerance=(
                base_nll_consistency_tolerance
            ),
            inner_stationarity_tolerance=_PROFILE_STATIONARITY_TOLERANCE,
        )
    problem = _prepare_problem(
        spectra,
        spec,
        calibration,
        resolution,
        dict(zip(result.parameter_names, result.parameter_values)),
        yield_model,
    )
    numerator_coefficients = np.zeros(len(result.parameter_names), dtype=np.float64)
    denominator_coefficients = np.zeros_like(numerator_coefficients)
    for line_index, line_name in enumerate(result.line_names):
        parameter_name = (
            f"spectrum.{line_name.split('.', 2)[1]}.line."
            f"{line_name.split('.', 2)[2]}.rate_counts_per_s"
            if line_name.startswith("spectrum.")
            else f"line.{line_name}.rate_counts_per_s"
        )
        parameter_index = problem.parameter_names.index(parameter_name)
        numerator_coefficients[parameter_index] = numerator_weights[line_index]
        denominator_coefficients[parameter_index] = denominator_weights[line_index]
    numerator = float(numerator_coefficients @ result.parameter_values)
    denominator = float(denominator_coefficients @ result.parameter_values)
    if numerator < 0 or denominator <= 0:
        return ProfileInterval(
            ratio_name,
            float("nan"),
            confidence_level,
            "failed",
            float("nan"),
            float("nan"),
            float("nan"),
            0,
            "linear-ratio numerator or denominator is outside its physical domain",
        )
    estimate = numerator / denominator
    boundary = numerator <= 1e-10
    return _profile_interval_from_inner_solves(
        ratio_name=ratio_name,
        estimate=estimate,
        confidence_level=confidence_level,
        boundary=boundary,
        max_evaluations=max_evaluations,
        base_nll_consistency_tolerance=base_nll_consistency_tolerance,
        fit_base_nll=result.penalized_nll,
        inner_solve=lambda value: _profile_nll_at_linear_ratio(
            problem,
            result.parameter_values,
            numerator_coefficients,
            denominator_coefficients,
            value,
        ),
        linear_constraint=True,
        infer_upper_limit_from_zero=False,
    )

def parametric_bootstrap(
    spectra: Sequence[PublicSpectrum],
    spec: JointPeakSpec,
    calibration: CalibrationConstraint,
    resolution: LinearResolution,
    result: JointPeakFitResult,
    *,
    case_identity: str,
    replicates: int,
    yield_model: YieldModel = "shared_origin_scales",
) -> BootstrapSummary:
    """Refit joint Poisson/Gaussian replicas and compare with Fisher errors."""

    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    if not result.success:
        raise ValueError("cannot bootstrap an unsuccessful fit")
    seed = stable_seed(case_identity)
    poisson_rng = np.random.default_rng(seed)
    gaussian_seed = stable_seed(
        case_identity
        + "\0"
        + _BOOTSTRAP_GAUSSIAN_PSEUDO_OBSERVATION_SEED_DOMAIN
    )
    gaussian_rng = np.random.default_rng(gaussian_seed)
    spectra_tuple = tuple(spectra)
    initial = dict(zip(result.parameter_names, result.parameter_values))
    generating_problem = _prepare_problem(
        spectra_tuple,
        spec,
        calibration,
        resolution,
        initial,
        yield_model,
    )
    if generating_problem.parameter_names != result.parameter_names:
        raise ValueError(
            "bootstrap reconstruction parameter names differ from the fitted "
            "result"
        )
    gaussian_parameter_names = tuple(
        generating_problem.parameter_names[index]
        for index in generating_problem.prior_parameter_indices
    )
    gaussian_generating_values = np.asarray(
        result.parameter_values[generating_problem.prior_parameter_indices],
        dtype=np.float64,
    )
    try:
        gaussian_covariance = np.linalg.inv(
            generating_problem.prior_precision
        )
        gaussian_covariance = (
            gaussian_covariance + gaussian_covariance.T
        ) / 2.0
        gaussian_cholesky = np.linalg.cholesky(gaussian_covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "Gaussian constraint covariance is not positive definite"
        ) from error
    fitted_rates: list[np.ndarray] = []
    fitted_errors: list[np.ndarray] = []
    boundary_rows: list[np.ndarray] = []
    constrained_fallback_replicates = 0
    exact_cone_invalid_replicates = 0
    cone_active_replicates = 0
    normalized_cone_margins: list[float] = []
    for _ in range(replicates):
        replica_counts = [np.asarray(spectrum.counts, dtype=np.float64).copy() for spectrum in spectra_tuple]
        generated = poisson_rng.poisson(result.expected_counts).astype(
            np.float64
        )
        gaussian_observations = gaussian_generating_values + (
            gaussian_cholesky
            @ gaussian_rng.standard_normal(gaussian_generating_values.size)
        )
        for observation_index, value in enumerate(generated):
            run_index = int(result.observation_spectrum_indices[observation_index])
            channel_index = int(result.observation_channel_indices[observation_index])
            replica_counts[run_index][channel_index] = value
        replicas = tuple(
            replace(spectrum, counts=counts) for spectrum, counts in zip(spectra_tuple, replica_counts)
        )
        replica_result = fit_joint_peak_model(
            replicas,
            spec,
            calibration,
            resolution,
            initial=initial,
            yield_model=yield_model,
            gaussian_constraint_observations=dict(
                zip(gaussian_parameter_names, gaussian_observations)
            ),
        )
        constrained_fallback_replicates += int(
            replica_result.quadratic_background_constrained_fallback_used
        )
        exact_cone_invalid_replicates += int(
            not replica_result.quadratic_background_exact_cone_valid
        )
        cone_active_replicates += int(
            bool(replica_result.quadratic_background_cone_active_windows)
        )
        normalized_cone_margins.extend(
            float(item["normalized_cone_margin"])
            for item in replica_result.quadratic_background_cone_diagnostics.values()
        )
        if not replica_result.success:
            continue
        fitted_rates.append(replica_result.line_rates_counts_per_s)
        diagonal = np.diag(replica_result.line_rate_covariance)
        fitted_errors.append(np.sqrt(np.maximum(diagonal, 0.0)))
        active = set(replica_result.active_bounds)
        boundary_rows.append(
            np.asarray(
                [
                    (
                        f"spectrum.{name.split('.', 2)[1]}.line."
                        f"{name.split('.', 2)[2]}.rate_counts_per_s"
                        if name.startswith("spectrum.")
                        else f"line.{name}.rate_counts_per_s"
                    )
                    in active
                    for name in replica_result.line_names
                ],
                dtype=np.float64,
            )
        )
    fisher_sd = np.sqrt(np.maximum(np.diag(result.line_rate_covariance), 0.0))
    if not fitted_rates:
        nan = np.full(len(result.line_names), np.nan)
        return BootstrapSummary(
            seed=seed,
            requested_replicates=replicates,
            successful_replicates=0,
            line_names=result.line_names,
            empirical_standard_deviation=nan,
            fisher_standard_deviation=fisher_sd,
            fisher_68_percent_coverage=nan.copy(),
            fisher_95_percent_coverage=nan.copy(),
            boundary_fraction=nan.copy(),
            quadratic_background_constrained_fallback_replicates=(
                constrained_fallback_replicates
            ),
            quadratic_background_exact_cone_invalid_replicates=(
                exact_cone_invalid_replicates
            ),
            quadratic_background_cone_active_replicates=(
                cone_active_replicates
            ),
            minimum_quadratic_background_normalized_cone_margin=(
                min(normalized_cone_margins)
                if normalized_cone_margins
                else float("nan")
            ),
            gaussian_pseudo_observation_seed=gaussian_seed,
            gaussian_pseudo_observation_convention=(
                _BOOTSTRAP_GAUSSIAN_PSEUDO_OBSERVATION_CONVENTION
            ),
            gaussian_pseudo_observation_parameter_names=(
                gaussian_parameter_names
            ),
            gaussian_pseudo_observation_generating_values=(
                gaussian_generating_values
            ),
            gaussian_pseudo_observation_covariance=gaussian_covariance,
        )
    rate_array = np.asarray(fitted_rates)
    error_array = np.asarray(fitted_errors)
    truth = result.line_rates_counts_per_s[np.newaxis, :]
    absolute_difference = np.abs(rate_array - truth)
    empirical_sd = np.std(rate_array, axis=0, ddof=1) if len(rate_array) > 1 else np.zeros(rate_array.shape[1])
    coverage_68 = np.mean(absolute_difference <= error_array, axis=0)
    coverage_95 = np.mean(absolute_difference <= 1.959963984540054 * error_array, axis=0)
    return BootstrapSummary(
        seed=seed,
        requested_replicates=replicates,
        successful_replicates=len(rate_array),
        line_names=result.line_names,
        empirical_standard_deviation=empirical_sd,
        fisher_standard_deviation=fisher_sd,
        fisher_68_percent_coverage=coverage_68,
        fisher_95_percent_coverage=coverage_95,
        boundary_fraction=np.mean(np.asarray(boundary_rows), axis=0),
        quadratic_background_constrained_fallback_replicates=(
            constrained_fallback_replicates
        ),
        quadratic_background_exact_cone_invalid_replicates=(
            exact_cone_invalid_replicates
        ),
        quadratic_background_cone_active_replicates=(
            cone_active_replicates
        ),
        minimum_quadratic_background_normalized_cone_margin=(
            min(normalized_cone_margins)
            if normalized_cone_margins
            else float("nan")
        ),
        gaussian_pseudo_observation_seed=gaussian_seed,
        gaussian_pseudo_observation_convention=(
            _BOOTSTRAP_GAUSSIAN_PSEUDO_OBSERVATION_CONVENTION
        ),
        gaussian_pseudo_observation_parameter_names=(
            gaussian_parameter_names
        ),
        gaussian_pseudo_observation_generating_values=(
            gaussian_generating_values
        ),
        gaussian_pseudo_observation_covariance=gaussian_covariance,
    )
