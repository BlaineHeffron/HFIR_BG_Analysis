"""Declarative Poisson likelihoods for paper-facing HPGe peak yields.

The model consumes raw, unrebinned integer detector-channel counts.  Every
line shape is normalized on the real line and integrated over calibrated bin
edges, so a fitted line-rate parameter has units of detector counts/s.  The
implementation is ROOT-free and does not read or write SQLite itself.

Several spectra may be fit simultaneously.  Line rates and the base declared
calibration/resolution curve are shared.  Optional Gaussian-constrained
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
from typing import Literal, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr
from scipy.stats import chi2

from src.public_data.browser import PublicSpectrum


ComponentRole = Literal["line", "fep", "sep", "dep", "contaminant"]
BackgroundModel = Literal["affine", "quadratic"]
ResolutionForm = Literal["linear", "sqrt"]
TailModel = Literal["none", "constant"]
_SQRT_2PI = np.sqrt(2.0 * np.pi)


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
    """Gaussian constraint on a common offset and fractional gain stretch.

    Before optional per-run deviations, calibrated edges are

    ``A0_r + offset_keV + A1_r * (1 + stretch) * channel_edge``.

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
    active_bounds: tuple[str, ...]
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
    run_scale_parameter_indices: dict[tuple[int, str], int]
    run_background_scale_parameter_indices: dict[int, int]
    run_calibration_parameter_indices: dict[int, tuple[int, int]]
    run_resolution_scale_parameter_indices: dict[int, int]
    resolution_parameter_indices: tuple[int, int]
    tail_parameter_indices: tuple[int, int] | None
    background_parameter_indices: dict[str, tuple[int, ...]]
    observed: np.ndarray
    saturated_nll: float
    spectrum_indices: np.ndarray
    window_names: tuple[str, ...]
    channel_indices: np.ndarray
    slices: dict[tuple[int, str], slice]
    channel_edges_low: dict[tuple[int, str], np.ndarray]
    channel_edges_high: dict[tuple[int, str], np.ndarray]


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


def _poisson_deviance(observed: np.ndarray, expected: np.ndarray) -> float:
    positive = observed > 0
    if np.any((expected <= 0) & positive):
        return float("inf")
    terms = expected - observed
    terms = terms.astype(np.float64, copy=True)
    terms[positive] += observed[positive] * np.log(
        observed[positive] / expected[positive]
    )
    return float(2.0 * np.sum(terms))


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
        name = f"line.{component.name}.rate_counts_per_s"
        component_parameter_indices[component.name] = len(values)
        names.append(name)
        values.append(float(supplied.get(name, rate_guess)))
        lower.append(0.0)
        upper.append(np.inf)
        scales.append(max(rate_guess, 1.0 / total_live))

    run_scale_parameter_indices: dict[tuple[int, str], int] = {}
    origin_classes = tuple(
        dict.fromkeys(component.origin_class for component in spec.components)
    )
    for run_index in range(1, len(spectra_tuple)):
        for origin_class in origin_classes:
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
            lower.append(1e-15)
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
        run_scale_parameter_indices,
        run_background_scale_parameter_indices,
        run_calibration_parameter_indices,
        run_resolution_scale_parameter_indices,
        resolution_parameter_indices,
        tail_parameter_indices,
        background_parameter_indices,
        observed,
        poisson_nll(observed, observed),
        spectrum_indices,
        tuple(window_name_parts),
        channel_indices,
        slices,
        low_channel_edges,
        high_channel_edges,
    )


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
            low_edges = spectrum.calibration_A0 + run_offset + A1 * low_channels
            high_edges = spectrum.calibration_A0 + run_offset + A1 * high_channels
            center_channels = 0.5 * (low_channels + high_channels)
            center_energy = (
                spectrum.calibration_A0 + run_offset + A1 * center_channels
            )
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
            expected[segment] = spectrum.live_time * background_scale * A1 * density
            if jacobian is not None:
                background_derivative_offset = (
                    spectrum.live_time
                    * background_scale
                    * A1
                    * density_derivative_fraction
                    * fraction_derivative_offset
                )
                background_derivative_stretch = (
                    spectrum.live_time
                    * background_scale
                    * (
                        spectrum.calibration_A1 * density
                        + A1
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
                for basis_index, parameter_index in enumerate(background_indices):
                    jacobian[segment, parameter_index] = (
                        spectrum.live_time
                        * background_scale
                        * A1
                        * basis[:, basis_index]
                    )
                if run_index > 0:
                    background_scale_index = (
                        problem.run_background_scale_parameter_indices[run_index]
                    )
                    jacobian[segment, background_scale_index] += (
                        spectrum.live_time * A1 * density
                    )

            for component in problem.spec.components:
                if component.window != window.name:
                    continue
                rate_index = problem.component_parameter_indices[component.name]
                rate = parameters[rate_index]
                signal_scale = (
                    1.0
                    if run_index == 0
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
                if run_index > 0:
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
    expected, jacobian = _model_and_jacobian(problem, parameters, with_jacobian=True)
    assert jacobian is not None
    if not np.isfinite(expected).all() or np.any(expected <= 0):
        return float("inf"), np.zeros(parameters.size, dtype=np.float64)
    residual_score = 1.0 - problem.observed / expected
    gradient = jacobian.T @ residual_score
    prior_delta = (
        parameters[problem.prior_parameter_indices] - problem.prior_mean
    )
    nll = (
        poisson_nll(problem.observed, expected)
        - problem.saturated_nll
        + 0.5
        * float(prior_delta @ problem.prior_precision @ prior_delta)
    )
    gradient[problem.prior_parameter_indices] += (
        problem.prior_precision @ prior_delta
    )
    return nll, gradient


def _scaled_bounds(problem: _PreparedProblem) -> list[tuple[float | None, float | None]]:
    bounds: list[tuple[float | None, float | None]] = []
    for value, low, high, scale in zip(
        problem.initial, problem.lower, problem.upper, problem.scales
    ):
        bounds.append(
            (
                None if not np.isfinite(low) else (low - value) / scale,
                None if not np.isfinite(high) else (high - value) / scale,
            )
        )
    return bounds


def _optimize_problem(problem: _PreparedProblem) -> tuple[object, np.ndarray]:
    def objective_scaled(scaled: np.ndarray) -> tuple[float, np.ndarray]:
        physical = problem.initial + problem.scales * scaled
        value, gradient = _objective_and_gradient(problem, physical)
        return value, gradient * problem.scales

    optimization = minimize(
        objective_scaled,
        np.zeros(problem.initial.size, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=_scaled_bounds(problem),
        options={"maxiter": 1200, "ftol": 1e-6, "gtol": 1e-4, "maxls": 50},
    )
    return optimization, problem.initial + problem.scales * optimization.x


def fit_joint_peak_model(
    spectra: Sequence[PublicSpectrum],
    spec: JointPeakSpec,
    calibration: CalibrationConstraint,
    resolution: LinearResolution,
    *,
    initial: Mapping[str, float] | None = None,
) -> JointPeakFitResult:
    """Fit declared components to one or more raw spectra simultaneously."""

    problem = _prepare_problem(spectra, spec, calibration, resolution, initial)
    optimization, fitted = _optimize_problem(problem)
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
    )
    try:
        covariance = np.linalg.inv(fisher)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(fisher, rcond=1e-12)
        fisher_covariance_valid = False
    covariance = (covariance + covariance.T) / 2.0

    active_bounds: list[str] = []
    for name, value, low, high, scale in zip(
        problem.parameter_names,
        fitted,
        problem.lower,
        problem.upper,
        problem.scales,
    ):
        tolerance = 1e-6 * max(scale, 1e-12)
        if np.isfinite(low) and value - low <= tolerance:
            active_bounds.append(name)
        elif np.isfinite(high) and high - value <= tolerance:
            active_bounds.append(name)

    line_names = tuple(component.name for component in spec.components)
    line_indices = np.asarray(
        [problem.component_parameter_indices[name] for name in line_names],
        dtype=np.int64,
    )
    prior_delta = fitted[problem.prior_parameter_indices] - problem.prior_mean
    data_poisson_nll = poisson_nll(problem.observed, expected) - problem.saturated_nll
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
        bool(optimization.success and np.isfinite(fitted_nll)),
        str(optimization.message),
        problem.parameter_names,
        fitted,
        covariance,
        fisher_rank,
        fisher_condition,
        fisher_covariance_valid,
        tuple(active_bounds),
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
        int(getattr(optimization, "nit", 0)),
        int(getattr(optimization, "nfev", 0)),
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
) -> tuple[float, bool]:
    if ratio < 0 or not isfinite(ratio):
        return float("inf"), False
    keep = np.asarray(
        [index for index in range(mle.size) if index != numerator_index],
        dtype=np.int64,
    )
    denominator_reduced = int(np.flatnonzero(keep == denominator_index)[0])
    reduced_initial = mle[keep].copy()
    reduced_scales = problem.scales[keep]
    reduced_lower = problem.lower[keep]
    reduced_upper = problem.upper[keep]

    def objective_scaled(scaled: np.ndarray) -> tuple[float, np.ndarray]:
        reduced = reduced_initial + reduced_scales * scaled
        physical = np.empty_like(mle)
        physical[keep] = reduced
        physical[numerator_index] = ratio * physical[denominator_index]
        value, full_gradient = _objective_and_gradient(problem, physical)
        reduced_gradient = full_gradient[keep]
        reduced_gradient[denominator_reduced] += ratio * full_gradient[numerator_index]
        return value, reduced_gradient * reduced_scales

    bounds = []
    for value, low, high, scale in zip(
        reduced_initial, reduced_lower, reduced_upper, reduced_scales
    ):
        bounds.append(
            (
                None if not np.isfinite(low) else (low - value) / scale,
                None if not np.isfinite(high) else (high - value) / scale,
            )
        )
    optimization = minimize(
        objective_scaled,
        np.zeros(reduced_initial.size, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 5e-10, "gtol": 2e-6, "maxls": 40},
    )
    return float(optimization.fun), bool(optimization.success)


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
) -> ProfileInterval:
    """Profile all nuisance parameters for a ratio interval or upper limit."""

    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence level must lie between 0.5 and 1")
    if max_evaluations < 8:
        raise ValueError("max_evaluations must be at least eight")
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
    threshold = 0.5 * float(
        chi2.ppf(2.0 * confidence_level - 1.0 if boundary else confidence_level, 1)
    )
    base_nll = result.poisson_nll
    evaluations = 0
    cache: dict[float, tuple[float, bool]] = {}

    def delta(value: float) -> float:
        nonlocal evaluations
        key = float(value)
        if key not in cache:
            if evaluations >= max_evaluations:
                return float("nan")
            cache[key] = _profile_nll_at_ratio(
                problem,
                result.parameter_values,
                numerator_index,
                denominator_index,
                key,
            )
            evaluations += 1
        return cache[key][0] - base_nll - threshold

    def bisect(left: float, right: float) -> float:
        f_left = delta(left)
        f_right = delta(right)
        if not np.isfinite(f_left) or not np.isfinite(f_right) or f_left * f_right > 0:
            return float("nan")
        for _ in range(40):
            if evaluations >= max_evaluations:
                break
            middle = 0.5 * (left + right)
            f_middle = delta(middle)
            if not np.isfinite(f_middle):
                break
            if abs(f_middle) < 2e-3 or right - left <= 1e-5 * max(1.0, estimate):
                return middle
            if f_left * f_middle <= 0:
                right = middle
                f_right = f_middle
            else:
                left = middle
                f_left = f_middle
        return 0.5 * (left + right)

    zero_delta = delta(0.0)
    is_upper_limit = boundary or (np.isfinite(zero_delta) and zero_delta <= 0)
    if is_upper_limit and not boundary:
        threshold = 0.5 * float(chi2.ppf(2.0 * confidence_level - 1.0, 1))
        zero_delta = delta(0.0)
    lower_value = 0.0
    if not is_upper_limit and estimate > 0:
        if np.isfinite(zero_delta) and zero_delta >= 0:
            lower_value = bisect(0.0, estimate)
        else:
            is_upper_limit = True

    right = max(estimate * 1.5, estimate + 0.05, 0.05)
    right_delta = delta(right)
    while (
        np.isfinite(right_delta)
        and right_delta < 0
        and evaluations < max_evaluations - 1
    ):
        right *= 2.0
        right_delta = delta(right)
    upper_value = bisect(estimate, right) if np.isfinite(right_delta) and right_delta >= 0 else float("nan")
    if not np.isfinite(upper_value) or (not is_upper_limit and not np.isfinite(lower_value)):
        return ProfileInterval(
            definition.name,
            estimate,
            confidence_level,
            "failed",
            float("nan"),
            float("nan"),
            threshold,
            evaluations,
            "profile threshold was not bracketed within the evaluation budget",
        )
    return ProfileInterval(
        definition.name,
        estimate,
        confidence_level,
        "upper_limit" if is_upper_limit else "two_sided",
        0.0 if is_upper_limit else lower_value,
        upper_value,
        threshold,
        evaluations,
        "one-sided boundary construction" if is_upper_limit else "two-sided profile-likelihood interval",
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
) -> BootstrapSummary:
    """Refit Poisson replicas and compare empirical spread with Fisher errors."""

    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    if not result.success:
        raise ValueError("cannot bootstrap an unsuccessful fit")
    seed = stable_seed(case_identity)
    rng = np.random.default_rng(seed)
    spectra_tuple = tuple(spectra)
    initial = dict(zip(result.parameter_names, result.parameter_values))
    fitted_rates: list[np.ndarray] = []
    fitted_errors: list[np.ndarray] = []
    boundary_rows: list[np.ndarray] = []
    for _ in range(replicates):
        replica_counts = [np.asarray(spectrum.counts, dtype=np.float64).copy() for spectrum in spectra_tuple]
        generated = rng.poisson(result.expected_counts).astype(np.float64)
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
                    f"line.{name}.rate_counts_per_s" in active
                    for name in replica_result.line_names
                ],
                dtype=np.float64,
            )
        )
    fisher_sd = np.sqrt(np.maximum(np.diag(result.line_rate_covariance), 0.0))
    if not fitted_rates:
        nan = np.full(len(result.line_names), np.nan)
        return BootstrapSummary(
            seed,
            replicates,
            0,
            result.line_names,
            nan,
            fisher_sd,
            nan.copy(),
            nan.copy(),
            nan.copy(),
        )
    rate_array = np.asarray(fitted_rates)
    error_array = np.asarray(fitted_errors)
    truth = result.line_rates_counts_per_s[np.newaxis, :]
    absolute_difference = np.abs(rate_array - truth)
    empirical_sd = np.std(rate_array, axis=0, ddof=1) if len(rate_array) > 1 else np.zeros(rate_array.shape[1])
    coverage_68 = np.mean(absolute_difference <= error_array, axis=0)
    coverage_95 = np.mean(absolute_difference <= 1.959963984540054 * error_array, axis=0)
    return BootstrapSummary(
        seed,
        replicates,
        len(rate_array),
        result.line_names,
        empirical_sd,
        fisher_sd,
        coverage_68,
        coverage_95,
        np.mean(np.asarray(boundary_rows), axis=0),
    )
