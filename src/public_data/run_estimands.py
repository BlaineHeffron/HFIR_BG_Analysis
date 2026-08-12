"""Covariance-aware estimands for independently fitted spectrum yields.

These helpers operate on fitted detector full-energy-peak rates.  They do not
apply detector efficiency, response unfolding, or incident-flux conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, Sequence

import numpy as np
from scipy.stats import chi2


@dataclass(frozen=True)
class AggregateEstimands:
    """Exposure-summed fitted counts and exposure-weighted detected rates."""

    component_names: tuple[str, ...]
    summed_counts: np.ndarray
    summed_count_covariance: np.ndarray
    aggregate_rates_counts_per_s: np.ndarray
    aggregate_rate_covariance: np.ndarray
    count_jacobian: np.ndarray
    rate_jacobian: np.ndarray


@dataclass(frozen=True)
class RunRatioEstimands:
    """Per-run ratios and joint delta-method covariance."""

    labels: tuple[str, ...]
    values: np.ndarray
    covariance: np.ndarray
    jacobian: np.ndarray


@dataclass(frozen=True)
class GLSHeterogeneity:
    """Generalized least-squares constant-rate test."""

    estimate: float
    standard_deviation: float
    q_statistic: float
    degrees_of_freedom: int
    p_value: float
    covariance_rank: int


@dataclass(frozen=True)
class InteriorGLSHeterogeneity:
    """GLS result after excluding explicitly nonregular boundary inputs."""

    heterogeneity: GLSHeterogeneity | None
    included_indices: tuple[int, ...]
    excluded_boundary_indices: tuple[int, ...]
    status: Literal[
        "available_all_inputs_interior",
        "available_after_boundary_exclusion",
        "unavailable_fewer_than_two_interior_inputs",
    ]


def _validated_inputs(
    component_names: Sequence[str],
    live_times_s: Sequence[float],
    run_rates: np.ndarray,
    covariance: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    names = tuple(component_names)
    times = np.asarray(live_times_s, dtype=np.float64)
    rates = np.asarray(run_rates, dtype=np.float64)
    if not names or len(set(names)) != len(names):
        raise ValueError("component names must be nonempty and unique")
    if rates.ndim == 1:
        if rates.size % len(names):
            raise ValueError("flat run-rate vector does not divide into components")
        rates = rates.reshape((-1, len(names)))
    if rates.ndim != 2 or rates.shape[1] != len(names):
        raise ValueError("run rates must have shape (runs, components)")
    if times.shape != (rates.shape[0],) or np.any(times <= 0):
        raise ValueError("live times must be positive with one value per run")
    flat_size = rates.size
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.shape != (flat_size, flat_size):
        raise ValueError("covariance does not match run-major rate vector")
    if (
        not np.isfinite(times).all()
        or not np.isfinite(rates).all()
        or not np.isfinite(cov).all()
        or not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-14)
    ):
        raise ValueError("estimand inputs must be finite and covariance symmetric")
    return names, times, rates, cov


def aggregate_independent_run_rates(
    component_names: Sequence[str],
    live_times_s: Sequence[float],
    run_rates_counts_per_s: np.ndarray,
    run_rate_covariance: np.ndarray,
) -> AggregateEstimands:
    """Aggregate independent run yields with exact linear covariance maps.

    For component ``j``, summed fitted detector counts are
    ``N_j = sum_r t_r lambda_rj``.  The aggregate detected count rate is
    ``N_j / sum_r t_r``.  Full cross-run and cross-component covariance is
    retained through the two Jacobians.
    """

    names, times, rates, covariance = _validated_inputs(
        component_names,
        live_times_s,
        run_rates_counts_per_s,
        run_rate_covariance,
    )
    run_count, component_count = rates.shape
    count_jacobian = np.zeros(
        (component_count, run_count * component_count), dtype=np.float64
    )
    for run_index, live_time in enumerate(times):
        start = run_index * component_count
        count_jacobian[:, start : start + component_count] = (
            np.eye(component_count) * live_time
        )
    summed_counts = count_jacobian @ rates.ravel()
    summed_covariance = count_jacobian @ covariance @ count_jacobian.T
    total_live_time = float(times.sum())
    rate_jacobian = count_jacobian / total_live_time
    aggregate_rates = summed_counts / total_live_time
    aggregate_covariance = rate_jacobian @ covariance @ rate_jacobian.T
    return AggregateEstimands(
        names,
        summed_counts,
        (summed_covariance + summed_covariance.T) / 2.0,
        aggregate_rates,
        (aggregate_covariance + aggregate_covariance.T) / 2.0,
        count_jacobian,
        rate_jacobian,
    )


def per_run_ratios(
    component_names: Sequence[str],
    run_rates_counts_per_s: np.ndarray,
    run_rate_covariance: np.ndarray,
    reference_component: str,
) -> RunRatioEstimands:
    """Return every component/reference ratio separately for every run."""

    rates = np.asarray(run_rates_counts_per_s, dtype=np.float64)
    if rates.ndim != 2:
        raise ValueError("run rates must have shape (runs, components)")
    names, _, rates, covariance = _validated_inputs(
        component_names, np.ones(rates.shape[0]), rates, run_rate_covariance
    )
    try:
        reference_index = names.index(reference_component)
    except ValueError as error:
        raise KeyError(reference_component) from error
    if np.any(rates[:, reference_index] <= 0):
        raise ValueError("each per-run ratio denominator must be positive")
    run_count, component_count = rates.shape
    values = np.empty(run_count * component_count, dtype=np.float64)
    jacobian = np.zeros((values.size, values.size), dtype=np.float64)
    labels: list[str] = []
    for run_index in range(run_count):
        denominator_position = run_index * component_count + reference_index
        denominator = rates[run_index, reference_index]
        for component_index, component_name in enumerate(names):
            row = run_index * component_count + component_index
            numerator_position = run_index * component_count + component_index
            labels.append(f"spectrum.{run_index}.{component_name}/{reference_component}")
            if component_index == reference_index:
                values[row] = 1.0
                continue
            numerator = rates[run_index, component_index]
            values[row] = numerator / denominator
            jacobian[row, numerator_position] = 1.0 / denominator
            jacobian[row, denominator_position] = -numerator / denominator**2
    ratio_covariance = jacobian @ covariance @ jacobian.T
    return RunRatioEstimands(
        tuple(labels),
        values,
        (ratio_covariance + ratio_covariance.T) / 2.0,
        jacobian,
    )


def gls_constant_heterogeneity(
    values: Sequence[float], covariance: np.ndarray
) -> GLSHeterogeneity:
    """Test one common value while retaining correlated fit uncertainty."""

    vector = np.asarray(values, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    if vector.ndim != 1 or not vector.size or cov.shape != (vector.size, vector.size):
        raise ValueError("heterogeneity inputs have incompatible shapes")
    if (
        not np.isfinite(vector).all()
        or not np.isfinite(cov).all()
        or not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-14)
    ):
        raise ValueError("heterogeneity inputs must be finite and symmetric")
    precision = np.linalg.pinv(cov, rcond=1e-12)
    ones = np.ones(vector.size, dtype=np.float64)
    information = float(ones @ precision @ ones)
    if not isfinite(information) or information <= 0:
        raise ValueError("constant estimand has zero covariance information")
    estimate = float(ones @ precision @ vector / information)
    residual = vector - estimate
    q_statistic = float(residual @ precision @ residual)
    covariance_rank = int(np.linalg.matrix_rank(cov))
    degrees_of_freedom = max(covariance_rank - 1, 0)
    p_value = (
        float(chi2.sf(q_statistic, degrees_of_freedom))
        if degrees_of_freedom
        else float("nan")
    )
    return GLSHeterogeneity(
        estimate,
        float(np.sqrt(1.0 / information)),
        q_statistic,
        degrees_of_freedom,
        p_value,
        covariance_rank,
    )


def gls_constant_heterogeneity_interior(
    values: Sequence[float],
    covariance: np.ndarray,
    boundary_mask: Sequence[bool],
) -> InteriorGLSHeterogeneity:
    """Run constant-value GLS only on regular interior inputs.

    Boundary-pinned estimates and their local Fisher covariance remain available
    for descriptive products, but are excluded from this inferential diagnostic.
    """

    vector = np.asarray(values, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    boundary = np.asarray(boundary_mask)
    if vector.ndim != 1 or not vector.size or cov.shape != (
        vector.size,
        vector.size,
    ):
        raise ValueError("heterogeneity inputs have incompatible shapes")
    if boundary.shape != vector.shape:
        raise ValueError("boundary mask must have one value per estimand")
    if (
        not np.isfinite(vector).all()
        or not np.isfinite(cov).all()
        or not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-14)
    ):
        raise ValueError("heterogeneity inputs must be finite and symmetric")
    boundary = boundary.astype(bool)
    included = tuple(int(index) for index in np.flatnonzero(~boundary))
    excluded = tuple(int(index) for index in np.flatnonzero(boundary))
    if len(included) < 2:
        return InteriorGLSHeterogeneity(
            None,
            included,
            excluded,
            "unavailable_fewer_than_two_interior_inputs",
        )
    positions = np.asarray(included, dtype=np.int64)
    heterogeneity = gls_constant_heterogeneity(
        vector[positions], cov[np.ix_(positions, positions)]
    )
    return InteriorGLSHeterogeneity(
        heterogeneity,
        included,
        excluded,
        (
            "available_after_boundary_exclusion"
            if excluded
            else "available_all_inputs_interior"
        ),
    )


def temporal_model_identifiability(
    elapsed_times_s: Sequence[float],
    reactor_states: Sequence[str],
    *,
    reactor_powers_mw: Sequence[float] | None = None,
    source_history_available: bool = False,
    known_half_life_s: float | None = None,
) -> tuple[dict[str, object], ...]:
    """Declare which temporal rate models the supplied metadata can identify.

    This deliberately does not infer sub-day reactor power or an irradiation
    history absent from the released metadata.
    """

    times = np.asarray(elapsed_times_s, dtype=np.float64)
    states = tuple(str(value) for value in reactor_states)
    if times.ndim != 1 or times.size != len(states) or not np.isfinite(times).all():
        raise ValueError("elapsed times and reactor states must be finite and aligned")
    if np.unique(times).size != times.size:
        raise ValueError("elapsed times must be distinct")
    rows: list[dict[str, object]] = [
        {
            "model": "constant_detected_rate",
            "identifiable": bool(times.size >= 1),
            "reason": "one common detected-rate parameter",
        },
        {
            "model": "unconstrained_per_run_detected_rate",
            "identifiable": bool(times.size >= 1),
            "reason": "descriptive saturated run-yield model",
        },
    ]
    if reactor_powers_mw is None:
        powers = None
        power_reason = "no interval-resolved reactor-power covariate released"
    else:
        powers = np.asarray(reactor_powers_mw, dtype=np.float64)
        if powers.shape != times.shape or not np.isfinite(powers).all():
            raise ValueError("reactor powers must be finite with one value per run")
        power_reason = (
            "reactor-power covariate varies"
            if np.unique(powers).size >= 2
            else "reactor-power covariate is constant"
        )
    rows.append(
        {
            "model": "reactor_power_correlated_scale",
            "identifiable": bool(powers is not None and np.unique(powers).size >= 2),
            "reason": power_reason,
        }
    )
    known_valid = known_half_life_s is not None and isfinite(known_half_life_s) and known_half_life_s > 0
    rows.extend(
        (
            {
                "model": "activation_decay_known_half_life",
                "identifiable": bool(
                    source_history_available and known_valid and times.size >= 2
                ),
                "reason": (
                    "requires released source/irradiation history, a valid declared half-life, and at least two time points"
                ),
            },
            {
                "model": "activation_decay_free_half_life",
                "identifiable": bool(source_history_available and times.size >= 3),
                "reason": (
                    "requires released source/irradiation history and at least three time points with an informative time design"
                ),
            },
        )
    )
    return tuple(rows)
