"""Native-bin residual localization for fitted public peak models."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.stats import chi2

from src.public_data.browser import PublicSpectrum
from src.public_data.peak_likelihood import (
    CalibrationConstraint,
    JointPeakFitResult,
    JointPeakSpec,
    LinearResolution,
    calibrated_channel_energy,
    resolution_sigma_and_derivatives,
)


def poisson_deviance_contributions(
    observed: np.ndarray, expected: np.ndarray
) -> np.ndarray:
    """Return exact nonnegative per-bin Poisson-deviance contributions."""

    observed = np.asarray(observed, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if (
        observed.shape != expected.shape
        or not np.isfinite(observed).all()
        or not np.isfinite(expected).all()
        or np.any(observed < 0)
        or np.any(expected <= 0)
    ):
        raise ValueError(
            "observed/expected arrays must align, be finite, and define valid counts"
        )
    positive = observed > 0
    terms = expected - observed
    terms = terms.astype(np.float64, copy=True)
    terms[positive] += observed[positive] * np.log(
        observed[positive] / expected[positive]
    )
    return np.maximum(2.0 * terms, 0.0)


def residual_bin_diagnostics(
    spectra: Sequence[PublicSpectrum],
    result: JointPeakFitResult,
    spec: JointPeakSpec,
    calibration: CalibrationConstraint,
    resolution: LinearResolution,
) -> list[dict[str, Any]]:
    """Describe fitted native bins in fitted calibration coordinates."""

    contributions = poisson_deviance_contributions(
        result.observed_counts, result.expected_counts
    )
    signed_deviance = np.sign(
        result.observed_counts - result.expected_counts
    ) * np.sqrt(contributions)
    components_by_window = {
        window.name: tuple(
            component for component in spec.components
            if component.window == window.name
        )
        for window in spec.windows
    }
    resolution_intercept = result.parameter("resolution.intercept_keV")
    slope_name = (
        "resolution.linear_sigma_slope_keV_per_keV"
        if resolution.form == "linear"
        else "resolution.variance_slope_keV"
    )
    resolution_slope = result.parameter(slope_name)
    rows: list[dict[str, Any]] = []
    for row_index, (observed, expected) in enumerate(
        zip(result.observed_counts, result.expected_counts)
    ):
        spectrum_index = int(result.observation_spectrum_indices[row_index])
        spectrum = spectra[spectrum_index]
        window = result.observation_window_names[row_index]
        channel = int(result.observation_channel_indices[row_index])
        offset = result.parameter("calibration.offset_keV")
        stretch = result.parameter("calibration.fractional_gain_stretch")
        offset_name = f"spectrum.{spectrum_index}.calibration_offset_deviation_keV"
        stretch_name = (
            f"spectrum.{spectrum_index}.fractional_gain_stretch_deviation"
        )
        if offset_name in result.parameter_names:
            offset += result.parameter(offset_name)
            stretch += result.parameter(stretch_name)
        curvature_name = "calibration.quadratic_curvature_keV_at_domain_edges"
        curvature = (
            result.parameter(curvature_name)
            if curvature_name in result.parameter_names
            else 0.0
        )
        fitted_low, fitted_center, fitted_high = calibrated_channel_energy(
            spectrum,
            calibration,
            np.asarray((channel + 0.5, channel + 1.0, channel + 1.5)),
            offset,
            stretch,
            curvature,
        )
        nearest = min(
            components_by_window[window],
            key=lambda component: abs(component.energy_keV - fitted_center),
        )
        sigma = resolution_sigma_and_derivatives(
            resolution,
            nearest.energy_keV,
            resolution_intercept,
            resolution_slope,
        )[0]
        scale_name = (
            f"spectrum.{spectrum_index}.resolution_scale_relative_to_spectrum_0"
        )
        if scale_name in result.parameter_names:
            sigma *= result.parameter(scale_name)
        delta = fitted_center - nearest.energy_keV
        rows.append(
            {
                "spectrum_index": spectrum_index,
                "file_id": spectrum.file_id,
                "window": window,
                "channel_index_zero_based": channel,
                "nominal_calibrated_center_energy_keV": float(
                    spectrum.energy_keV[channel]
                ),
                "fitted_calibrated_low_edge_keV": fitted_low,
                "fitted_calibrated_center_energy_keV": fitted_center,
                "fitted_calibrated_high_edge_keV": fitted_high,
                "observed_counts_per_bin": float(observed),
                "expected_counts_per_bin": float(expected),
                "poisson_residual": float(
                    (observed - expected) / np.sqrt(max(expected, 1e-12))
                ),
                "poisson_deviance_contribution": float(contributions[row_index]),
                "signed_poisson_deviance_residual": float(
                    signed_deviance[row_index]
                ),
                "nearest_declared_component": nearest.name,
                "nearest_declared_component_energy_keV": nearest.energy_keV,
                "fitted_center_minus_nearest_component_keV": delta,
                "nearest_component_sigma_keV": sigma,
                "fitted_center_minus_nearest_component_sigma": delta / sigma,
            }
        )
    return rows


def dominant_signed_residual_cluster(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize the same-sign contiguous channel cluster with largest deviance."""

    clusters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_sign = 0.0
    for row in sorted(rows, key=lambda item: item["channel_index_zero_based"]):
        sign = float(np.sign(row["signed_poisson_deviance_residual"]))
        contiguous = bool(
            current
            and row["channel_index_zero_based"]
            == current[-1]["channel_index_zero_based"] + 1
        )
        if sign and current and sign == current_sign and contiguous:
            current.append(row)
        else:
            if current:
                clusters.append(current)
            current = [row] if sign else []
            current_sign = sign
    if current:
        clusters.append(current)
    if not clusters:
        return {
            "dominant_residual_cluster_sign": "none",
            "dominant_residual_cluster_start_channel_zero_based": "",
            "dominant_residual_cluster_end_channel_zero_based": "",
            "dominant_residual_cluster_low_edge_keV": "",
            "dominant_residual_cluster_high_edge_keV": "",
            "dominant_residual_cluster_poisson_deviance": "0",
            "dominant_residual_cluster_fraction_of_window_deviance": "0",
            "dominant_residual_cluster_observed_minus_expected_counts": "0",
            "dominant_residual_cluster_peak_signed_deviance_residual": "0",
            "dominant_residual_cluster_nearest_declared_component": "",
            "dominant_residual_cluster_peak_component_offset_sigma": "",
        }
    dominant = max(
        clusters,
        key=lambda cluster: sum(
            item["poisson_deviance_contribution"] for item in cluster
        ),
    )
    peak = max(dominant, key=lambda item: item["poisson_deviance_contribution"])
    deviance = sum(item["poisson_deviance_contribution"] for item in dominant)
    total_deviance = sum(item["poisson_deviance_contribution"] for item in rows)
    return {
        "dominant_residual_cluster_sign": (
            "positive"
            if dominant[0]["signed_poisson_deviance_residual"] > 0
            else "negative"
        ),
        "dominant_residual_cluster_start_channel_zero_based": dominant[0][
            "channel_index_zero_based"
        ],
        "dominant_residual_cluster_end_channel_zero_based": dominant[-1][
            "channel_index_zero_based"
        ],
        "dominant_residual_cluster_low_edge_keV": format(
            dominant[0]["fitted_calibrated_low_edge_keV"], ".12g"
        ),
        "dominant_residual_cluster_high_edge_keV": format(
            dominant[-1]["fitted_calibrated_high_edge_keV"], ".12g"
        ),
        "dominant_residual_cluster_poisson_deviance": format(deviance, ".12g"),
        "dominant_residual_cluster_fraction_of_window_deviance": format(
            deviance / max(total_deviance, 1e-300), ".12g"
        ),
        "dominant_residual_cluster_observed_minus_expected_counts": format(
            sum(
                item["observed_counts_per_bin"]
                - item["expected_counts_per_bin"]
                for item in dominant
            ),
            ".12g",
        ),
        "dominant_residual_cluster_peak_signed_deviance_residual": format(
            peak["signed_poisson_deviance_residual"], ".12g"
        ),
        "dominant_residual_cluster_nearest_declared_component": peak[
            "nearest_declared_component"
        ],
        "dominant_residual_cluster_peak_component_offset_sigma": format(
            peak["fitted_center_minus_nearest_component_sigma"], ".12g"
        ),
    }


def window_residual_diagnostics(
    result: JointPeakFitResult,
    spectra: Sequence[PublicSpectrum],
    spec: JointPeakSpec,
    calibration: CalibrationConstraint,
    resolution: LinearResolution,
    minimum_expected_counts: float,
) -> list[dict[str, Any]]:
    """Summarize signed residual structure on every fitted native window."""

    rows: list[dict[str, Any]] = []
    bin_rows = residual_bin_diagnostics(
        spectra, result, spec, calibration, resolution
    )
    component_counts = {
        window.name: sum(
            component.window == window.name for component in spec.components
        )
        for window in spec.windows
    }
    for spectrum_index, spectrum in enumerate(spectra):
        for window in spec.windows:
            mask = (
                (result.observation_spectrum_indices == spectrum_index)
                & (np.asarray(result.observation_window_names) == window.name)
            )
            observed = result.observed_counts[mask]
            expected = result.expected_counts[mask]
            selected = [
                row
                for row in bin_rows
                if row["spectrum_index"] == spectrum_index
                and row["window"] == window.name
            ]
            contributions = np.asarray(
                [row["poisson_deviance_contribution"] for row in selected]
            )
            signed = np.asarray(
                [row["signed_poisson_deviance_residual"] for row in selected]
            )
            residual = (observed - expected) / np.sqrt(
                np.maximum(expected, 1e-12)
            )
            eligible = expected >= minimum_expected_counts
            full_deviance = float(
                poisson_deviance_contributions(observed, expected).sum()
            )
            reference_deviance = float(
                poisson_deviance_contributions(
                    observed[eligible], expected[eligible]
                ).sum()
            )
            reference_dof = int(np.count_nonzero(eligible))
            absolute_residual = np.abs(residual[eligible])
            rows.append(
                {
                    "spectrum_index": spectrum_index,
                    "file_id": spectrum.file_id,
                    "window": window.name,
                    "low_keV": format(window.low_keV, ".12g"),
                    "high_keV": format(window.high_keV, ".12g"),
                    "raw_bin_count": int(mask.sum()),
                    "chi_square_reference_bin_count": reference_dof,
                    "excluded_low_expected_bin_count": int(
                        mask.sum() - reference_dof
                    ),
                    "chi_square_minimum_expected_counts_per_bin": format(
                        minimum_expected_counts, ".12g"
                    ),
                    "declared_component_count": component_counts[window.name],
                    "observed_counts": format(float(observed.sum()), ".12g"),
                    "expected_counts": format(float(expected.sum()), ".12g"),
                    "full_poisson_deviance": format(full_deviance, ".12g"),
                    "positive_residual_poisson_deviance": format(
                        float(contributions[signed > 0].sum()), ".12g"
                    ),
                    "negative_residual_poisson_deviance": format(
                        float(contributions[signed < 0].sum()), ".12g"
                    ),
                    **dominant_signed_residual_cluster(selected),
                    "chi_square_reference_poisson_deviance": format(
                        reference_deviance, ".12g"
                    ),
                    "chi_square_reference_degrees_of_freedom": reference_dof,
                    "chi_square_reference_p_value": format(
                        (
                            float(chi2.sf(reference_deviance, reference_dof))
                            if reference_dof > 0
                            else float("nan")
                        ),
                        ".12g",
                    ),
                    "largest_absolute_poisson_residual": format(
                        (
                            float(np.max(absolute_residual))
                            if reference_dof > 0
                            else float("nan")
                        ),
                        ".12g",
                    ),
                    "bins_with_absolute_residual_gt_4": int(
                        np.count_nonzero(absolute_residual > 4.0)
                    ),
                    "bins_with_absolute_residual_gt_5": int(
                        np.count_nonzero(absolute_residual > 5.0)
                    ),
                    "p_value_semantics": (
                        "diagnostic chi-square reference restricted to bins meeting "
                        "the declared minimum fitted expectation; shared fitted-parameter "
                        "allocation is not unique"
                    ),
                    "dominant_residual_cluster_definition": (
                        "contiguous native channels with the same nonzero "
                        "observed-minus-expected sign; selected by largest summed "
                        "exact Poisson-deviance contribution"
                    ),
                }
            )
    return rows
