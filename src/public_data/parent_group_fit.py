"""ROOT-free Poisson fits for related HPGe full- and escape-energy peaks.

The fitter operates on raw, unrebinned :class:`PublicSpectrum` channel
counts.  Fit windows are converted to a fixed set of channel bins using the
mean affine calibration.  The affine coefficients then remain Gaussian-
constrained nuisance parameters during the fit, so their covariance is
propagated to peak locations, areas, and escape ratios without changing the
selected data.

This first implementation intentionally uses Gaussian peak components.  A
low-energy-tail model should be added only as a normalized, bin-integrated
component with an explicit mixture fraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, Mapping

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from src.public_data.browser import PublicSpectrum


PeakRole = Literal["fep", "sep", "dep", "contaminant"]
_TARGET_ROLES = ("fep", "sep", "dep")


@dataclass(frozen=True)
class AffineCalibrationConstraint:
    """Mean and covariance for ``energy_keV = A0 + A1 * channel``."""

    A0_keV: float
    A1_keV_per_channel: float
    covariance: np.ndarray


@dataclass(frozen=True)
class FitWindow:
    """A predeclared half-open energy interval used by one local fit."""

    name: str
    low_keV: float
    high_keV: float


@dataclass(frozen=True)
class PeakComponent:
    """One target or contaminant line in a named fit window."""

    name: str
    role: PeakRole
    energy_keV: float
    window: str
    sigma_guess_keV: float
    centroid_half_width_keV: float = 20.0
    sigma_min_keV: float = 0.2
    sigma_max_keV: float = 12.0


@dataclass(frozen=True)
class ParentGroupSpec:
    """Explicit components and disjoint windows for one parent group."""

    name: str
    windows: tuple[FitWindow, ...]
    components: tuple[PeakComponent, ...]


@dataclass(frozen=True)
class ParentGroupFitResult:
    """Fit values, covariance, derived target rates, and diagnostics."""

    success: bool
    message: str
    parameter_names: tuple[str, ...]
    parameter_values: np.ndarray
    covariance: np.ndarray
    covariance_valid: bool
    covariance_rank: int
    covariance_condition: float
    active_bounds: tuple[str, ...]
    bin_indices: np.ndarray
    bin_window_names: tuple[str, ...]
    observed_counts: np.ndarray
    expected_counts: np.ndarray
    poisson_nll: float
    poisson_deviance: float
    degrees_of_freedom: int
    target_labels: tuple[str, ...]
    target_areas_counts: np.ndarray
    target_area_covariance: np.ndarray
    target_rates_per_s: np.ndarray
    target_rate_covariance: np.ndarray
    ratio_labels: tuple[str, ...]
    ratios: np.ndarray
    ratio_covariance: np.ndarray

    def parameter(self, name: str) -> float:
        """Return a fitted physical parameter by its stable name."""

        try:
            index = self.parameter_names.index(name)
        except ValueError as error:
            raise KeyError(name) from error
        return float(self.parameter_values[index])


def gaussian_bin_probabilities(
    low_edges_keV: np.ndarray,
    high_edges_keV: np.ndarray,
    centroid_keV: float,
    sigma_keV: float,
) -> np.ndarray:
    """Integrate a unit-normalized Gaussian over each energy bin."""

    low = np.asarray(low_edges_keV, dtype=np.float64)
    high = np.asarray(high_edges_keV, dtype=np.float64)
    if low.ndim != 1 or high.ndim != 1 or low.shape != high.shape:
        raise ValueError("Gaussian bin edges must be equal-length one-dimensional arrays")
    if (
        not np.isfinite(low).all()
        or not np.isfinite(high).all()
        or np.any(high <= low)
    ):
        raise ValueError("Gaussian bin edges must be finite and increasing within each bin")
    if not isfinite(centroid_keV):
        raise ValueError("Gaussian centroid must be finite")
    if not isfinite(sigma_keV) or sigma_keV <= 0:
        raise ValueError("Gaussian sigma must be finite and positive")
    return ndtr((high - centroid_keV) / sigma_keV) - ndtr(
        (low - centroid_keV) / sigma_keV
    )


def poisson_nll(observed: np.ndarray, expected: np.ndarray) -> float:
    """Poisson negative log likelihood up to the data-only constant."""

    counts = np.asarray(observed, dtype=np.float64)
    means = np.asarray(expected, dtype=np.float64)
    if counts.ndim != 1 or means.ndim != 1 or counts.shape != means.shape:
        raise ValueError("observed and expected counts must be equal-length vectors")
    if (
        not np.isfinite(counts).all()
        or np.any(counts < 0)
        or not np.allclose(counts, np.rint(counts), rtol=0.0, atol=1e-9)
    ):
        raise ValueError("observed values must be finite nonnegative integer counts")
    if not np.isfinite(means).all() or np.any(means < 0):
        raise ValueError("expected values must be finite and nonnegative")
    impossible = (means == 0) & (counts > 0)
    if np.any(impossible):
        return float("inf")
    positive = counts > 0
    return float(np.sum(means) - np.sum(counts[positive] * np.log(means[positive])))


def _poisson_deviance(observed: np.ndarray, expected: np.ndarray) -> float:
    positive = observed > 0
    if np.any((expected == 0) & positive):
        return float("inf")
    terms = expected - observed
    terms = terms.astype(np.float64, copy=True)
    terms[positive] += observed[positive] * np.log(
        observed[positive] / expected[positive]
    )
    return float(2.0 * np.sum(terms))


def _validate_spec(spec: ParentGroupSpec) -> tuple[dict[str, FitWindow], dict[str, int]]:
    if not spec.name.strip():
        raise ValueError("parent-group name must not be empty")
    if not spec.windows:
        raise ValueError("parent group must declare at least one fit window")
    if not spec.components:
        raise ValueError("parent group must declare at least one peak component")

    windows: dict[str, FitWindow] = {}
    for window in spec.windows:
        if not window.name.strip() or window.name in windows:
            raise ValueError("fit-window names must be nonempty and unique")
        if (
            not isfinite(window.low_keV)
            or not isfinite(window.high_keV)
            or window.high_keV <= window.low_keV
        ):
            raise ValueError(f"fit window {window.name!r} has invalid bounds")
        windows[window.name] = window
    ordered_windows = sorted(windows.values(), key=lambda item: item.low_keV)
    for left, right in zip(ordered_windows, ordered_windows[1:]):
        if right.low_keV < left.high_keV:
            raise ValueError("fit windows must not overlap")

    names: set[str] = set()
    target_indices: dict[str, int] = {}
    for index, component in enumerate(spec.components):
        if not component.name.strip() or component.name in names:
            raise ValueError("peak-component names must be nonempty and unique")
        names.add(component.name)
        if component.role not in (*_TARGET_ROLES, "contaminant"):
            raise ValueError(f"unsupported peak role: {component.role}")
        if component.role in _TARGET_ROLES:
            if component.role in target_indices:
                raise ValueError(f"parent group has more than one {component.role} component")
            target_indices[component.role] = index
        if component.window not in windows:
            raise ValueError(
                f"component {component.name!r} references unknown window {component.window!r}"
            )
        window = windows[component.window]
        values = (
            component.energy_keV,
            component.sigma_guess_keV,
            component.centroid_half_width_keV,
            component.sigma_min_keV,
            component.sigma_max_keV,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError(f"component {component.name!r} contains a nonfinite value")
        if not window.low_keV < component.energy_keV < window.high_keV:
            raise ValueError(f"component {component.name!r} lies outside its fit window")
        if (
            component.sigma_guess_keV <= 0
            or component.centroid_half_width_keV <= 0
            or component.sigma_min_keV <= 0
            or component.sigma_max_keV <= component.sigma_min_keV
            or not component.sigma_min_keV
            <= component.sigma_guess_keV
            <= component.sigma_max_keV
        ):
            raise ValueError(f"component {component.name!r} has invalid width settings")
        if (
            component.energy_keV - component.centroid_half_width_keV
            <= window.low_keV
            or component.energy_keV + component.centroid_half_width_keV
            >= window.high_keV
        ):
            raise ValueError(
                f"component {component.name!r} centroid bounds leave its fit window"
            )
    if set(target_indices) != set(_TARGET_ROLES):
        raise ValueError("parent group must contain exactly one fep, sep, and dep target")
    return windows, target_indices


def _validate_calibration(
    calibration: AffineCalibrationConstraint,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(
        [calibration.A0_keV, calibration.A1_keV_per_channel], dtype=np.float64
    )
    covariance = np.asarray(calibration.covariance, dtype=np.float64)
    if not np.isfinite(mean).all() or mean[1] <= 0:
        raise ValueError("calibration mean must be finite with positive A1")
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        raise ValueError("calibration covariance must be a finite 2x2 matrix")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-14):
        raise ValueError("calibration covariance must be symmetric")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("calibration covariance must be positive definite") from error
    return mean, covariance


def _validate_raw_spectrum(spectrum: PublicSpectrum) -> None:
    counts = np.asarray(spectrum.counts, dtype=np.float64)
    if not np.allclose(counts, np.rint(counts), rtol=0.0, atol=1e-9):
        raise ValueError("parent-group fitting requires raw integer channel counts")
    channels = np.arange(1, counts.size + 1, dtype=np.float64)
    expected_energy = spectrum.calibration_A0 + spectrum.calibration_A1 * channels
    expected_width = np.full(counts.size, spectrum.calibration_A1)
    if not np.allclose(
        spectrum.energy_keV, expected_energy, rtol=1e-12, atol=1e-9
    ) or not np.allclose(
        spectrum.bin_width_keV, expected_width, rtol=1e-12, atol=1e-12
    ):
        raise ValueError("parent-group fitting requires an unrebinned PublicSpectrum")


def _numerical_model_jacobian(
    model,
    parameters: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    baseline = model(parameters)
    jacobian = np.empty((baseline.size, parameters.size), dtype=np.float64)
    for index in range(parameters.size):
        step = max(abs(parameters[index]) * 1e-5, scales[index] * 1e-5, 1e-7)
        can_lower = parameters[index] - step > lower[index]
        can_upper = parameters[index] + step < upper[index]
        if can_lower and can_upper:
            plus = parameters.copy()
            minus = parameters.copy()
            plus[index] += step
            minus[index] -= step
            jacobian[:, index] = (model(plus) - model(minus)) / (2.0 * step)
        elif can_upper:
            plus = parameters.copy()
            plus[index] += step
            jacobian[:, index] = (model(plus) - baseline) / step
        elif can_lower:
            minus = parameters.copy()
            minus[index] -= step
            jacobian[:, index] = (baseline - model(minus)) / step
        else:
            jacobian[:, index] = 0.0
    return jacobian


def fit_parent_group(
    spectrum: PublicSpectrum,
    spec: ParentGroupSpec,
    calibration: AffineCalibrationConstraint,
    *,
    initial: Mapping[str, float] | None = None,
) -> ParentGroupFitResult:
    """Fit a parent and its two escape peaks with a joint Poisson likelihood.

    Target and optional contaminant areas, centroid shifts, and Gaussian
    widths are free.  Each disjoint ROI has a nonnegative affine background,
    represented by its count-density values at the two ROI channel edges.
    """

    _validate_raw_spectrum(spectrum)
    windows, target_component_indices = _validate_spec(spec)
    calibration_mean, calibration_covariance = _validate_calibration(calibration)
    calibration_precision = np.linalg.inv(calibration_covariance)
    supplied_initial = {} if initial is None else dict(initial)
    if not all(isfinite(float(value)) for value in supplied_initial.values()):
        raise ValueError("initial parameter values must be finite")

    channel_centers = np.arange(1, spectrum.counts.size + 1, dtype=np.float64)
    nominal_energy = calibration_mean[0] + calibration_mean[1] * channel_centers
    indices_by_window: dict[str, np.ndarray] = {}
    for window in spec.windows:
        indices = np.flatnonzero(
            (nominal_energy >= window.low_keV)
            & (nominal_energy < window.high_keV)
        )
        if indices.size < 5:
            raise ValueError(f"fit window {window.name!r} contains fewer than five bins")
        indices_by_window[window.name] = indices
    all_indices = np.concatenate(
        [indices_by_window[window.name] for window in spec.windows]
    )
    if np.unique(all_indices).size != all_indices.size:
        raise ValueError("fit windows select overlapping channel bins")
    observed = np.asarray(spectrum.counts[all_indices], dtype=np.float64)
    local_slices: dict[str, slice] = {}
    start = 0
    bin_window_names: list[str] = []
    for window in spec.windows:
        stop = start + indices_by_window[window.name].size
        local_slices[window.name] = slice(start, stop)
        bin_window_names.extend([window.name] * (stop - start))
        start = stop

    names = ["calibration.A0_keV", "calibration.A1_keV_per_channel"]
    values = [calibration_mean[0], calibration_mean[1]]
    lower = [-np.inf, np.finfo(np.float64).tiny]
    upper = [np.inf, np.inf]
    scales = [
        max(float(np.sqrt(calibration_covariance[0, 0])), 1e-3),
        max(float(np.sqrt(calibration_covariance[1, 1])), 1e-6),
    ]
    component_parameter_indices: dict[str, tuple[int, int, int]] = {}

    for component in spec.components:
        window_indices = indices_by_window[component.window]
        window_counts = np.asarray(spectrum.counts[window_indices], dtype=np.float64)
        window_energy = nominal_energy[window_indices]
        side_count = max(2, window_counts.size // 5)
        background_per_bin = max(
            float(
                np.median(
                    np.concatenate(
                        (window_counts[:side_count], window_counts[-side_count:])
                    )
                )
            ),
            1e-6,
        )
        search = (
            np.abs(window_energy - component.energy_keV)
            <= component.centroid_half_width_keV
        )
        if np.any(search):
            local_peak = int(np.argmax(window_counts[search]))
            peak_energy = window_energy[search][local_peak]
        else:
            peak_energy = component.energy_keV
        near = np.abs(window_energy - peak_energy) <= max(
            3.0 * component.sigma_guess_keV, calibration_mean[1]
        )
        excess = np.maximum(window_counts[near] - background_per_bin, 0.0)
        area_guess = max(float(np.sum(excess)), 1.0)
        if np.any(search):
            shift_guess = float(
                np.clip(
                    peak_energy - component.energy_keV,
                    -0.8 * component.centroid_half_width_keV,
                    0.8 * component.centroid_half_width_keV,
                )
            )
        else:
            shift_guess = 0.0
        area_name = f"line.{component.name}.area_counts"
        shift_name = f"line.{component.name}.centroid_shift_keV"
        sigma_name = f"line.{component.name}.sigma_keV"
        component_parameter_indices[component.name] = (
            len(values),
            len(values) + 1,
            len(values) + 2,
        )
        names.extend((area_name, shift_name, sigma_name))
        values.extend(
            (
                float(supplied_initial.get(area_name, area_guess)),
                float(supplied_initial.get(shift_name, shift_guess)),
                float(
                    supplied_initial.get(sigma_name, component.sigma_guess_keV)
                ),
            )
        )
        lower.extend((0.0, -component.centroid_half_width_keV, component.sigma_min_keV))
        upper.extend((np.inf, component.centroid_half_width_keV, component.sigma_max_keV))
        scales.extend(
            (
                max(area_guess, 1.0),
                max(component.centroid_half_width_keV / 2.0, 1.0),
                max(component.sigma_guess_keV, 1.0),
            )
        )

    background_parameter_indices: dict[str, tuple[int, int]] = {}
    for window in spec.windows:
        window_counts = np.asarray(
            spectrum.counts[indices_by_window[window.name]], dtype=np.float64
        )
        side_count = max(2, window_counts.size // 5)
        left_guess = max(
            float(np.median(window_counts[:side_count])) / calibration_mean[1],
            1e-9,
        )
        right_guess = max(
            float(np.median(window_counts[-side_count:])) / calibration_mean[1],
            1e-9,
        )
        left_name = f"background.{window.name}.low_counts_per_keV"
        right_name = f"background.{window.name}.high_counts_per_keV"
        background_parameter_indices[window.name] = (len(values), len(values) + 1)
        names.extend((left_name, right_name))
        values.extend(
            (
                float(supplied_initial.get(left_name, left_guess)),
                float(supplied_initial.get(right_name, right_guess)),
            )
        )
        lower.extend((1e-12, 1e-12))
        upper.extend((np.inf, np.inf))
        scales.extend((max(left_guess, 1.0), max(right_guess, 1.0)))

    parameters0 = np.asarray(values, dtype=np.float64)
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    scale_array = np.asarray(scales, dtype=np.float64)
    if set(supplied_initial).difference(names):
        unknown = sorted(set(supplied_initial).difference(names))
        raise ValueError("unknown initial parameters: " + ", ".join(unknown))
    if np.any(parameters0 < lower_array) or np.any(parameters0 > upper_array):
        raise ValueError("an initial parameter value lies outside its bounds")

    selected_channel_edges_low = all_indices.astype(np.float64) + 0.5
    selected_channel_edges_high = all_indices.astype(np.float64) + 1.5

    def model(physical: np.ndarray) -> np.ndarray:
        A0, A1 = physical[:2]
        if not isfinite(A0) or not isfinite(A1) or A1 <= 0:
            return np.full(observed.size, np.nan)
        low_edges = A0 + A1 * selected_channel_edges_low
        high_edges = A0 + A1 * selected_channel_edges_high
        expected = np.zeros(observed.size, dtype=np.float64)
        for window in spec.windows:
            window_slice = local_slices[window.name]
            count = window_slice.stop - window_slice.start
            fractions = np.linspace(0.0, 1.0, count + 1)
            background_low_index, background_high_index = (
                background_parameter_indices[window.name]
            )
            density_low = physical[background_low_index] + (
                physical[background_high_index] - physical[background_low_index]
            ) * fractions[:-1]
            density_high = physical[background_low_index] + (
                physical[background_high_index] - physical[background_low_index]
            ) * fractions[1:]
            expected[window_slice] = A1 * (density_low + density_high) / 2.0
        for component in spec.components:
            area_index, shift_index, sigma_index = component_parameter_indices[
                component.name
            ]
            window_slice = local_slices[component.window]
            centroid = component.energy_keV + physical[shift_index]
            probabilities = gaussian_bin_probabilities(
                low_edges[window_slice],
                high_edges[window_slice],
                centroid,
                physical[sigma_index],
            )
            expected[window_slice] += physical[area_index] * probabilities
        return expected

    def objective_scaled(scaled: np.ndarray) -> float:
        physical = parameters0 + scale_array * scaled
        expected = model(physical)
        if not np.isfinite(expected).all() or np.any(expected <= 0):
            return float("inf")
        delta_calibration = physical[:2] - calibration_mean
        penalty = 0.5 * float(
            delta_calibration @ calibration_precision @ delta_calibration
        )
        return poisson_nll(observed, expected) + penalty

    scaled0 = np.zeros(parameters0.size, dtype=np.float64)
    scaled_bounds = []
    for value, low_bound, high_bound, scale in zip(
        parameters0, lower_array, upper_array, scale_array
    ):
        scaled_low = (
            None if not np.isfinite(low_bound) else (low_bound - value) / scale
        )
        scaled_high = (
            None if not np.isfinite(high_bound) else (high_bound - value) / scale
        )
        scaled_bounds.append((scaled_low, scaled_high))
    optimization = minimize(
        objective_scaled,
        scaled0,
        method="L-BFGS-B",
        bounds=scaled_bounds,
        options={"maxiter": 4000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
    )
    fitted = parameters0 + scale_array * optimization.x
    fitted_expected = model(fitted)

    tolerance = 1e-6
    active_bounds: list[str] = []
    for name, value, low_bound, high_bound, scale in zip(
        names, fitted, lower_array, upper_array, scale_array
    ):
        absolute_tolerance = tolerance * max(scale, 1.0)
        if np.isfinite(low_bound) and value - low_bound <= absolute_tolerance:
            active_bounds.append(name)
        elif np.isfinite(high_bound) and high_bound - value <= absolute_tolerance:
            active_bounds.append(name)

    jacobian = _numerical_model_jacobian(
        model, fitted, lower_array, upper_array, scale_array
    )
    fisher = jacobian.T @ (jacobian / fitted_expected[:, np.newaxis])
    fisher[:2, :2] += calibration_precision
    fisher = (fisher + fisher.T) / 2.0
    fisher_scaled = (
        scale_array[:, np.newaxis] * fisher * scale_array[np.newaxis, :]
    )
    covariance_rank = int(np.linalg.matrix_rank(fisher_scaled))
    try:
        covariance_condition = float(np.linalg.cond(fisher_scaled))
    except np.linalg.LinAlgError:
        covariance_condition = float("inf")
    covariance_valid = (
        covariance_rank == fitted.size
        and isfinite(covariance_condition)
        and covariance_condition <= 1e14
        and not active_bounds
    )
    try:
        covariance = np.linalg.inv(fisher)
        covariance = (covariance + covariance.T) / 2.0
    except np.linalg.LinAlgError:
        covariance = np.full((fitted.size, fitted.size), np.nan)
        covariance_valid = False

    target_labels = tuple(_TARGET_ROLES)
    target_area_indices = np.asarray(
        [
            component_parameter_indices[
                spec.components[target_component_indices[role]].name
            ][0]
            for role in target_labels
        ],
        dtype=np.int64,
    )
    target_areas = fitted[target_area_indices]
    target_area_covariance = covariance[np.ix_(target_area_indices, target_area_indices)]
    target_rates = target_areas / spectrum.live_time
    target_rate_covariance = target_area_covariance / (spectrum.live_time**2)

    ratio_labels = ("sep/fep", "dep/fep")
    if target_areas[0] > 0:
        ratios = np.asarray(
            [target_areas[1] / target_areas[0], target_areas[2] / target_areas[0]],
            dtype=np.float64,
        )
        ratio_jacobian = np.asarray(
            [
                [
                    -target_areas[1] / target_areas[0] ** 2,
                    1.0 / target_areas[0],
                    0.0,
                ],
                [
                    -target_areas[2] / target_areas[0] ** 2,
                    0.0,
                    1.0 / target_areas[0],
                ],
            ]
        )
        ratio_covariance = (
            ratio_jacobian @ target_area_covariance @ ratio_jacobian.T
        )
    else:
        ratios = np.full(2, np.nan)
        ratio_covariance = np.full((2, 2), np.nan)
        covariance_valid = False

    delta_calibration = fitted[:2] - calibration_mean
    penalty = 0.5 * float(
        delta_calibration @ calibration_precision @ delta_calibration
    )
    fitted_nll = poisson_nll(observed, fitted_expected) + penalty
    degrees_of_freedom = int(observed.size + 2 - fitted.size)
    return ParentGroupFitResult(
        success=bool(optimization.success and np.isfinite(fitted_nll)),
        message=str(optimization.message),
        parameter_names=tuple(names),
        parameter_values=fitted,
        covariance=covariance,
        covariance_valid=covariance_valid,
        covariance_rank=covariance_rank,
        covariance_condition=covariance_condition,
        active_bounds=tuple(active_bounds),
        bin_indices=all_indices,
        bin_window_names=tuple(bin_window_names),
        observed_counts=observed,
        expected_counts=fitted_expected,
        poisson_nll=fitted_nll,
        poisson_deviance=_poisson_deviance(observed, fitted_expected),
        degrees_of_freedom=degrees_of_freedom,
        target_labels=target_labels,
        target_areas_counts=target_areas,
        target_area_covariance=target_area_covariance,
        target_rates_per_s=target_rates,
        target_rate_covariance=target_rate_covariance,
        ratio_labels=ratio_labels,
        ratios=ratios,
        ratio_covariance=ratio_covariance,
    )
