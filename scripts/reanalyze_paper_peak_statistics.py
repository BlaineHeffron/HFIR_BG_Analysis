#!/usr/bin/env python3
"""Generate exploratory measured-data candidates for paper Tables 3 and 8.

The immutable public database and spectra are opened read-only.  The command
writes only to a caller-selected empty output directory.  It neither unfolds
spectra nor reads, fabricates, combines, or fits missing simulation products.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import binom, chi2, norm

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

from src.public_data.browser import PublicSpectrum, load_spectrum
from src.public_data.catalog import build_run_catalog
from src.public_data.peak_likelihood import (
    BootstrapSummary,
    CalibrationConstraint,
    FitWindow,
    JointPeakFitResult,
    JointPeakSpec,
    LineComponent,
    LinearResolution,
    ProfileInterval,
    RatioDefinition,
    fit_joint_peak_model,
    parametric_bootstrap,
    profile_linear_ratio_interval,
    profile_ratio_interval,
    ratio_values_and_covariance,
    ratios_from_fit,
)
from src.public_data.run_estimands import (
    aggregate_independent_run_rates,
    gls_constant_heterogeneity_interior,
    per_run_ratios,
    temporal_model_identifiability,
)


PUBLIC_V1_1_DB_SHA256 = (
    "c78bc8fa6ef7dbe1a8ea5d0189e69eb555c8a488fd582ff04b965a08aa1985e9"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _spectrum_path(data_root: Path, spectrum: PublicSpectrum) -> Path:
    name = spectrum.file_name
    if not name.lower().endswith(".txt"):
        name += ".txt"
    path = data_root / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty output: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_fit_bins(
    path: Path,
    spectra: Sequence[PublicSpectrum],
    result: JointPeakFitResult,
) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        fields = [
            "spectrum_index",
            "file_id",
            "window",
            "channel_index_zero_based",
            "observed_counts_per_bin",
            "expected_counts_per_bin",
            "poisson_residual",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row_index, (observed, expected) in enumerate(
            zip(result.observed_counts, result.expected_counts)
        ):
            spectrum_index = int(result.observation_spectrum_indices[row_index])
            writer.writerow(
                {
                    "spectrum_index": spectrum_index,
                    "file_id": spectra[spectrum_index].file_id,
                    "window": result.observation_window_names[row_index],
                    "channel_index_zero_based": int(
                        result.observation_channel_indices[row_index]
                    ),
                    "observed_counts_per_bin": format(float(observed), ".12g"),
                    "expected_counts_per_bin": format(float(expected), ".12g"),
                    "poisson_residual": format(
                        float((observed - expected) / np.sqrt(max(expected, 1e-12))),
                        ".12g",
                    ),
                }
            )


def _constraint(config: dict[str, Any]) -> CalibrationConstraint:
    return CalibrationConstraint(
        offset_mean_keV=float(config["offset_mean_keV"]),
        stretch_mean=float(config["stretch_mean"]),
        covariance=np.asarray(config["covariance"], dtype=np.float64),
        offset_bounds_keV=tuple(
            float(value) for value in config["offset_bounds_keV"]
        ),
        stretch_bounds=tuple(float(value) for value in config["stretch_bounds"]),
        per_run_deviation_covariance=(
            None
            if "per_run_deviation_covariance" not in config
            else np.asarray(
                config["per_run_deviation_covariance"], dtype=np.float64
            )
        ),
        per_run_offset_bounds_keV=tuple(
            float(value)
            for value in config.get("per_run_offset_bounds_keV", (-1.0, 1.0))
        ),
        per_run_stretch_bounds=tuple(
            float(value)
            for value in config.get("per_run_stretch_bounds", (-2e-4, 2e-4))
        ),
    )


def _resolution(config: dict[str, Any]) -> LinearResolution:
    return LinearResolution(
        intercept_keV=float(config["intercept_keV"]),
        slope=float(config["slope"]),
        tail_fraction=float(config["tail_fraction"]),
        tail_scale_in_sigma=float(config["tail_scale_in_sigma"]),
        form=config.get("form", "linear"),
        tail_model=config.get("tail_model", "constant"),
        per_run_scale_sigma=float(config.get("per_run_scale_sigma", 0.0)),
        per_run_scale_bounds=tuple(
            float(value)
            for value in config.get("per_run_scale_bounds", (0.7, 1.3))
        ),
    )


def _spec(table_config: dict[str, Any], name: str) -> JointPeakSpec:
    windows = tuple(
        FitWindow(
            item["name"],
            float(item["low_keV"]),
            float(item["high_keV"]),
            item.get("background_model", table_config.get("background_model", "affine")),
        )
        for item in table_config["windows"]
    )
    components = tuple(
        LineComponent(
            item["name"],
            float(item["energy_keV"]),
            item["window"],
            item["role"],
            item.get("parent", ""),
            item.get("origin_class", "shared"),
        )
        for item in table_config["components"]
    )
    return JointPeakSpec(name, windows, components)


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _input_record(
    spectrum: PublicSpectrum,
    data_root: Path,
    run_record: dict[str, Any],
) -> dict[str, Any]:
    path = _spectrum_path(data_root, spectrum)
    metadata = spectrum.metadata
    return {
        "file_id": spectrum.file_id,
        "run_id": spectrum.run_id,
        "run_name": spectrum.run_name,
        "file_name": spectrum.file_name,
        "start_time_unix_s": _plain(metadata.get("start_time")),
        "live_time_s": spectrum.live_time,
        "spectrum_value_unit": "measured detector counts per raw channel",
        "normalization": "none in likelihood input",
        "calibration_A0_keV": spectrum.calibration_A0,
        "calibration_A1_keV_per_channel": spectrum.calibration_A1,
        "calibration_group_id": _plain(metadata.get("calibration_group_id")),
        "calibration_group_name": _plain(metadata.get("calibration_group_name")),
        "reactor_cycle_inferred_from_name": _plain(run_record["reactor_cycle"]),
        "reactor_state_inferred_from_name": _plain(run_record["reactor_state"]),
        "calendar_cycle": _plain(run_record["calendar_cycle"]),
        "calendar_reactor_state": _plain(run_record["calendar_reactor_state"]),
        "calendar_period": _plain(run_record["calendar_period"]),
        "calendar_date_precision": _plain(run_record["calendar_date_precision"]),
        "coordinate_id": _plain(metadata.get("coordinate_id")),
        "coordinate_Rx": _plain(metadata.get("coordinate_Rx")),
        "coordinate_Rz": _plain(metadata.get("coordinate_Rz")),
        "coordinate_Lx": _plain(metadata.get("coordinate_Lx")),
        "coordinate_Lz": _plain(metadata.get("coordinate_Lz")),
        "detector_orientation_angle_deg": _plain(metadata.get("coordinate_angle")),
        "coordinate_track": _plain(metadata.get("coordinate_track")),
        "detector_configuration_id": _plain(metadata.get("detector_configuration_id")),
        "detector_id": _plain(metadata.get("detector_id")),
        "detector_type": _plain(metadata.get("detector_type")),
        "shield_id": _plain(metadata.get("shield_id")),
        "shield_name": _plain(metadata.get("shield_name")),
        "shield_description": _plain(metadata.get("shield_description")),
        "response_model_identity": "not applicable; measured detector counts",
        "spectrum_sha256": _sha256(path),
    }


def _covariance_rows(
    names: Sequence[str], covariance: np.ndarray, unit: str
) -> list[dict[str, Any]]:
    return [
        {
            "row": row_name,
            "column": column_name,
            "covariance": format(float(covariance[row, column]), ".12g"),
            "unit": unit,
        }
        for row, row_name in enumerate(names)
        for column, column_name in enumerate(names)
    ]


def _parameter_unit(name: str) -> str:
    if (
        name == "calibration.offset_keV"
        or name == "resolution.intercept_keV"
        or name.endswith("calibration_offset_deviation_keV")
    ):
        return "keV"
    if name == "resolution.variance_slope_keV":
        return "keV"
    if name == "resolution.linear_sigma_slope_keV_per_keV":
        return "keV/keV"
    if name.startswith("line."):
        return "counts/s"
    if name.startswith("background."):
        return "counts/(s keV)"
    return "dimensionless"


def _parameter_covariance_rows(
    result: JointPeakFitResult,
) -> list[dict[str, Any]]:
    return [
        {
            "row": row_name,
            "row_unit": _parameter_unit(row_name),
            "column": column_name,
            "column_unit": _parameter_unit(column_name),
            "covariance": format(float(result.covariance[row, column]), ".12g"),
        }
        for row, row_name in enumerate(result.parameter_names)
        for column, column_name in enumerate(result.parameter_names)
    ]


def _count_deviance(observed: np.ndarray, expected: np.ndarray) -> float:
    """Poisson deviance for a selected diagnostic subset."""

    positive = observed > 0
    terms = expected - observed
    terms = terms.astype(np.float64, copy=True)
    terms[positive] += observed[positive] * np.log(
        observed[positive] / expected[positive]
    )
    return float(2.0 * terms.sum())


def _window_diagnostics(
    result: JointPeakFitResult,
    spectra: Sequence[PublicSpectrum],
    spec: JointPeakSpec,
    minimum_expected_counts: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    component_counts = {
        window.name: sum(component.window == window.name for component in spec.components)
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
            residual = (observed - expected) / np.sqrt(np.maximum(expected, 1e-12))
            eligible = expected >= minimum_expected_counts
            full_window_deviance = _count_deviance(observed, expected)
            reference_deviance = _count_deviance(
                observed[eligible], expected[eligible]
            )
            reference_dof = int(np.count_nonzero(eligible))
            eligible_absolute_residual = np.abs(residual[eligible])
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
                    "full_poisson_deviance": format(
                        full_window_deviance, ".12g"
                    ),
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
                            float(np.max(eligible_absolute_residual))
                            if reference_dof > 0
                            else float("nan")
                        ),
                        ".12g",
                    ),
                    "bins_with_absolute_residual_gt_4": int(
                        np.count_nonzero(eligible_absolute_residual > 4.0)
                    ),
                    "bins_with_absolute_residual_gt_5": int(
                        np.count_nonzero(eligible_absolute_residual > 5.0)
                    ),
                    "p_value_semantics": (
                        "diagnostic chi-square reference restricted to bins meeting "
                        "the declared minimum fitted expectation; shared fitted-parameter "
                        "allocation is not unique"
                    ),
                }
            )
    return rows


def _reference_core_diagnostics(
    result: JointPeakFitResult,
    spectra: Sequence[PublicSpectrum],
    core_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Report the declared reference-line core balance run by run."""

    window_name = str(core_config["window"])
    low_keV = float(core_config["low_keV"])
    high_keV = float(core_config["high_keV"])
    rows: list[dict[str, Any]] = []
    observation_windows = np.asarray(result.observation_window_names)
    for spectrum_index, spectrum in enumerate(spectra):
        run_mask = (
            (result.observation_spectrum_indices == spectrum_index)
            & (observation_windows == window_name)
        )
        channels = result.observation_channel_indices[run_mask]
        nominal_energy = np.asarray(spectrum.energy_keV)[channels]
        core_mask = (nominal_energy >= low_keV) & (nominal_energy < high_keV)
        observed = result.observed_counts[run_mask][core_mask]
        expected = result.expected_counts[run_mask][core_mask]
        observed_total = float(observed.sum())
        expected_total = float(expected.sum())
        relative_balance = (
            (observed_total - expected_total) / expected_total
            if expected_total > 0
            else float("nan")
        )
        signed_poisson_residual = (
            (observed_total - expected_total) / np.sqrt(expected_total)
            if expected_total > 0
            else float("nan")
        )
        run_offset_name = (
            f"spectrum.{spectrum_index}.calibration_offset_deviation_keV"
        )
        run_stretch_name = (
            f"spectrum.{spectrum_index}.fractional_gain_stretch_deviation"
        )
        run_resolution_name = (
            f"spectrum.{spectrum_index}.resolution_scale_relative_to_spectrum_0"
        )
        rows.append(
            {
                "spectrum_index": spectrum_index,
                "file_id": spectrum.file_id,
                "window": window_name,
                "core_low_keV": format(low_keV, ".12g"),
                "core_high_keV": format(high_keV, ".12g"),
                "core_energy_convention": core_config["energy_convention"],
                "core_raw_bin_count": int(np.count_nonzero(core_mask)),
                "observed_core_counts": format(observed_total, ".12g"),
                "expected_core_counts": format(expected_total, ".12g"),
                "relative_core_balance": format(relative_balance, ".12g"),
                "signed_aggregate_poisson_residual": format(
                    signed_poisson_residual, ".12g"
                ),
                "fitted_calibration_offset_deviation_keV": format(
                    (
                        0.0
                        if run_offset_name not in result.parameter_names
                        else result.parameter(run_offset_name)
                    ),
                    ".12g",
                ),
                "fitted_fractional_gain_stretch_deviation": format(
                    (
                        0.0
                        if run_stretch_name not in result.parameter_names
                        else result.parameter(run_stretch_name)
                    ),
                    ".12g",
                ),
                "fitted_resolution_scale_relative_to_spectrum_0": format(
                    (
                        1.0
                        if run_resolution_name not in result.parameter_names
                        else result.parameter(run_resolution_name)
                    ),
                    ".12g",
                ),
            }
        )
    return rows


def _origin_scale_diagnostics(
    result: JointPeakFitResult,
    spectra: Sequence[PublicSpectrum],
) -> list[dict[str, Any]]:
    """Report fitted capture/decay composition changes relative to run zero."""

    rows: list[dict[str, Any]] = []
    for spectrum_index, spectrum in enumerate(spectra):
        if spectrum_index == 0:
            capture_scale = decay_scale = double_ratio = 1.0
            capture_error = decay_error = double_ratio_error = 0.0
        else:
            capture_name = (
                f"spectrum.{spectrum_index}.signal_scale.neutron_capture"
                "_relative_to_spectrum_0"
            )
            decay_name = (
                f"spectrum.{spectrum_index}.signal_scale.radioactive_decay"
                "_relative_to_spectrum_0"
            )
            capture_index = result.parameter_names.index(capture_name)
            decay_index = result.parameter_names.index(decay_name)
            capture_scale = float(result.parameter_values[capture_index])
            decay_scale = float(result.parameter_values[decay_index])
            capture_error = float(np.sqrt(result.covariance[capture_index, capture_index]))
            decay_error = float(np.sqrt(result.covariance[decay_index, decay_index]))
            double_ratio = capture_scale / decay_scale
            gradient = np.asarray(
                [1.0 / decay_scale, -capture_scale / decay_scale**2]
            )
            scale_covariance = result.covariance[
                np.ix_((capture_index, decay_index), (capture_index, decay_index))
            ]
            double_ratio_error = float(
                np.sqrt(max(float(gradient @ scale_covariance @ gradient), 0.0))
            )
        decay_to_capture_composition = 1.0 / double_ratio
        decay_to_capture_composition_error = (
            double_ratio_error / double_ratio**2
        )
        rows.append(
            {
                "spectrum_index": spectrum_index,
                "file_id": spectrum.file_id,
                "neutron_capture_scale_relative_to_spectrum_0": format(
                    capture_scale, ".12g"
                ),
                "neutron_capture_scale_fisher_uncertainty": format(
                    capture_error, ".12g"
                ),
                "radioactive_decay_scale_relative_to_spectrum_0": format(
                    decay_scale, ".12g"
                ),
                "radioactive_decay_scale_fisher_uncertainty": format(
                    decay_error, ".12g"
                ),
                "capture_to_decay_double_ratio_relative_to_spectrum_0": format(
                    double_ratio, ".12g"
                ),
                "double_ratio_fisher_uncertainty": format(
                    double_ratio_error, ".12g"
                ),
                "double_ratio_deviation_from_one_fisher_z": format(
                    (
                        0.0
                        if double_ratio_error == 0.0
                        else (double_ratio - 1.0) / double_ratio_error
                    ),
                    ".12g",
                ),
                "decay_to_capture_composition_relative_to_spectrum_0": format(
                    decay_to_capture_composition, ".12g"
                ),
                "decay_to_capture_composition_fisher_uncertainty": format(
                    decay_to_capture_composition_error, ".12g"
                ),
                "semantics": (
                    "fitted neutron-capture/decay composition relative to spectrum "
                    "index 0; index 0 is exactly one by parameter convention"
                ),
            }
        )
    return rows


def _fit_diagnostics(
    result: JointPeakFitResult,
    reporting: dict[str, Any],
) -> dict[str, Any]:
    standard_deviation = np.sqrt(np.maximum(np.diag(result.covariance), 0.0))
    deviance_per_dof = result.poisson_deviance / result.degrees_of_freedom
    minimum_expected_counts = float(
        reporting["chi_square_minimum_expected_counts_per_bin"]
    )
    residual = (result.observed_counts - result.expected_counts) / np.sqrt(
        np.maximum(result.expected_counts, 1e-12)
    )
    absolute_residual = np.abs(residual)
    chi_square_eligible = result.expected_counts >= minimum_expected_counts
    eligible_bin_count = int(np.count_nonzero(chi_square_eligible))
    excluded_low_expected_bin_count = int(
        result.observed_counts.size - eligible_bin_count
    )
    global_reference_deviance = _count_deviance(
        result.observed_counts[chi_square_eligible],
        result.expected_counts[chi_square_eligible],
    )
    global_reference_dof = max(
        eligible_bin_count - len(result.parameter_names), 1
    )
    global_p_value = float(
        chi2.sf(global_reference_deviance, global_reference_dof)
    )
    eligible_absolute_residual = absolute_residual[chi_square_eligible]
    bin_tests: list[dict[str, Any]] = []
    for threshold in reporting["per_bin_outlier_sigma_thresholds"]:
        threshold = float(threshold)
        observed_count = int(
            np.count_nonzero(eligible_absolute_residual > threshold)
        )
        null_probability = float(2.0 * norm.sf(threshold))
        bin_tests.append(
            {
                "absolute_residual_threshold_sigma": threshold,
                "eligible_bin_count": eligible_bin_count,
                "excluded_low_expected_bin_count": excluded_low_expected_bin_count,
                "minimum_expected_counts_per_bin": minimum_expected_counts,
                "observed_outlier_count": observed_count,
                "expected_outlier_count_under_normal_reference": (
                    eligible_bin_count * null_probability
                ),
                "binomial_excess_p_value": float(
                    binom.sf(
                        observed_count - 1,
                        eligible_bin_count,
                        null_probability,
                    )
                ),
            }
        )

    window_tests: list[dict[str, Any]] = []
    window_names = np.asarray(result.observation_window_names)
    for window_name in sorted(set(result.observation_window_names)):
        mask = window_names == window_name
        observed = result.observed_counts[mask]
        expected = result.expected_counts[mask]
        eligible = expected >= minimum_expected_counts
        deviance = _count_deviance(observed[eligible], expected[eligible])
        reference_dof = int(np.count_nonzero(eligible))
        window_tests.append(
            {
                "window": window_name,
                "full_poisson_deviance": _count_deviance(observed, expected),
                "chi_square_reference_poisson_deviance": deviance,
                "chi_square_reference_bin_count": reference_dof,
                "excluded_low_expected_bin_count": int(mask.sum() - reference_dof),
                "minimum_expected_counts_per_bin": minimum_expected_counts,
                "chi_square_reference_degrees_of_freedom": reference_dof,
                "chi_square_reference_p_value": float(
                    chi2.sf(deviance, reference_dof)
                    if reference_dof > 0
                    else float("nan")
                ),
                "largest_absolute_poisson_residual": float(
                    np.max(absolute_residual[mask][eligible])
                    if reference_dof > 0
                    else float("nan")
                ),
            }
        )

    global_alpha = float(reporting["global_deviance_p_value_minimum"])
    familywise_alpha = float(reporting["familywise_diagnostic_alpha"])
    window_alpha = familywise_alpha / max(len(window_tests), 1)
    bin_alpha = familywise_alpha / max(len(bin_tests), 1)
    applicability_reasons: list[str] = []
    optimizer_converged = bool(
        getattr(result, "optimizer_converged", result.success)
    )
    stationarity_valid = bool(
        getattr(result, "optimizer_stationarity_valid", result.success)
    )
    stationarity_tolerance = float(
        getattr(result, "stationarity_tolerance", float("nan"))
    )
    declared_stationarity_tolerance = float(
        reporting.get(
            "optimizer_scaled_projected_gradient_tolerance",
            stationarity_tolerance,
        )
    )
    if np.isfinite(stationarity_tolerance) and not np.isclose(
        stationarity_tolerance,
        declared_stationarity_tolerance,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError(
            "fitter stationarity tolerance differs from frozen reporting configuration"
        )
    optimizer_configuration = getattr(result, "optimizer_configuration", {})
    cone_diagnostics = {
        str(window_name): {
            str(key): float(value) for key, value in diagnostics.items()
        }
        for window_name, diagnostics in getattr(
            result, "quadratic_background_cone_diagnostics", {}
        ).items()
    }
    exact_cone_valid = bool(
        getattr(result, "quadratic_background_exact_cone_valid", True)
    )
    cone_active_windows = tuple(
        getattr(result, "quadratic_background_cone_active_windows", ())
    )
    constrained_cone_fallback_used = bool(
        getattr(
            result,
            "quadratic_background_constrained_fallback_used",
            False,
        )
    )
    declared_cone_feasibility_tolerance = float(
        reporting.get(
            "quadratic_background_cone_feasibility_relative_tolerance",
            getattr(
                result,
                "quadratic_background_cone_feasibility_relative_tolerance",
                float("nan"),
            ),
        )
    )
    declared_cone_activity_tolerance = float(
        reporting.get(
            "quadratic_background_cone_activity_relative_tolerance",
            getattr(
                result,
                "quadratic_background_cone_activity_relative_tolerance",
                float("nan"),
            ),
        )
    )
    for result_attribute, declared_value in (
        (
            "quadratic_background_cone_feasibility_relative_tolerance",
            declared_cone_feasibility_tolerance,
        ),
        (
            "quadratic_background_cone_activity_relative_tolerance",
            declared_cone_activity_tolerance,
        ),
    ):
        fitter_value = float(getattr(result, result_attribute, float("nan")))
        if np.isfinite(fitter_value) and not np.isclose(
            fitter_value, declared_value, rtol=0.0, atol=1.0e-15
        ):
            raise ValueError(
                f"fitter {result_attribute} differs from frozen reporting configuration"
            )
    declared_scoring_max_iterations = int(
        reporting.get(
            "optimizer_scoring_max_iterations",
            optimizer_configuration.get("scoring_max_iterations", 0),
        )
    )
    fitter_scoring_max_iterations = int(
        optimizer_configuration.get(
            "scoring_max_iterations", declared_scoring_max_iterations
        )
    )
    if fitter_scoring_max_iterations != declared_scoring_max_iterations:
        raise ValueError(
            "fitter scoring iteration cap differs from frozen reporting configuration"
        )
    declared_scoring_flat_nll_tolerance = float(
        reporting.get(
            "optimizer_scoring_numerically_flat_nll_tolerance",
            optimizer_configuration.get(
                "scoring_numerically_flat_nll_tolerance", float("nan")
            ),
        )
    )
    fitter_scoring_flat_nll_tolerance = float(
        optimizer_configuration.get(
            "scoring_numerically_flat_nll_tolerance",
            declared_scoring_flat_nll_tolerance,
        )
    )
    if np.isfinite(fitter_scoring_flat_nll_tolerance) and not np.isclose(
        fitter_scoring_flat_nll_tolerance,
        declared_scoring_flat_nll_tolerance,
        rtol=0.0,
        atol=1.0e-18,
    ):
        raise ValueError(
            "fitter scoring flat-NLL tolerance differs from frozen reporting configuration"
        )
    declared_scoring_flat_gradient_reduction = float(
        reporting.get(
            "optimizer_scoring_numerically_flat_gradient_reduction_factor",
            optimizer_configuration.get(
                "scoring_numerically_flat_gradient_reduction_factor",
                float("nan"),
            ),
        )
    )
    fitter_scoring_flat_gradient_reduction = float(
        optimizer_configuration.get(
            "scoring_numerically_flat_gradient_reduction_factor",
            declared_scoring_flat_gradient_reduction,
        )
    )
    if np.isfinite(fitter_scoring_flat_gradient_reduction) and not np.isclose(
        fitter_scoring_flat_gradient_reduction,
        declared_scoring_flat_gradient_reduction,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError(
            "fitter scoring flat-gradient reduction differs from frozen reporting configuration"
        )
    declared_polish_displacement_limit = float(
        reporting.get(
            "optimizer_polish_max_scaled_displacement_inf",
            optimizer_configuration.get(
                "polish_max_scaled_displacement_inf", float("nan")
            ),
        )
    )
    declared_polish_nll_decrease_limit = float(
        reporting.get(
            "optimizer_polish_max_stable_nll_decrease",
            optimizer_configuration.get(
                "polish_max_stable_nll_decrease", float("nan")
            ),
        )
    )
    for fitter_key, declared_value in (
        (
            "polish_max_scaled_displacement_inf",
            declared_polish_displacement_limit,
        ),
        (
            "polish_max_stable_nll_decrease",
            declared_polish_nll_decrease_limit,
        ),
    ):
        fitter_value = float(optimizer_configuration.get(fitter_key, float("nan")))
        if np.isfinite(fitter_value) and not np.isclose(
            fitter_value, declared_value, rtol=0.0, atol=1.0e-15
        ):
            raise ValueError(
                f"fitter {fitter_key} differs from frozen reporting configuration"
            )
    polish_basin_valid = bool(
        getattr(result, "optimizer_polish_basin_valid", True)
    )
    if not optimizer_converged:
        applicability_reasons.append("optimizer solver did not converge")
    if not stationarity_valid:
        applicability_reasons.append(
            "optimizer scaled projected-gradient stationarity gate failed"
        )
    if not polish_basin_valid:
        applicability_reasons.append("optimizer polish basin guard failed")
    if not exact_cone_valid:
        applicability_reasons.append(
            "quadratic background exact interval-nonnegativity cone check failed"
        )
    if cone_active_windows:
        applicability_reasons.append(
            "quadratic background exact cone is active and the fit is nonregular"
        )
    if not result.success and optimizer_converged and stationarity_valid:
        applicability_reasons.append("fit result is unsuccessful")
    if not result.fisher_covariance_valid:
        applicability_reasons.append("Fisher covariance rank/condition check failed")
    if result.active_bounds:
        applicability_reasons.append("one or more nuisance/yield bounds are active")
    if global_p_value < global_alpha:
        applicability_reasons.append(
            "global Poisson deviance rejects the fitted count model"
        )
    if any(
        np.isfinite(item["chi_square_reference_p_value"])
        and item["chi_square_reference_p_value"] < window_alpha
        for item in window_tests
    ):
        applicability_reasons.append(
            "one or more declared windows fail the Bonferroni-calibrated deviance diagnostic"
        )
    if any(item["binomial_excess_p_value"] < bin_alpha for item in bin_tests):
        applicability_reasons.append(
            "per-bin residual outliers exceed the familywise-calibrated normal reference"
        )
    applicability_reasons.extend(reporting.get("forced_applicability_reasons", []))
    return {
        "success": result.success,
        "optimizer_message": result.message,
        "optimizer_converged": optimizer_converged,
        "optimizer_method": getattr(result, "optimizer_method", "unrecorded"),
        "optimizer_stages": list(getattr(result, "optimizer_stages", ())),
        "optimizer_configuration": optimizer_configuration,
        "quadratic_background_cone_diagnostics": cone_diagnostics,
        "quadratic_background_exact_cone_valid": exact_cone_valid,
        "quadratic_background_cone_active_windows": list(
            cone_active_windows
        ),
        "quadratic_background_constrained_fallback_used": (
            constrained_cone_fallback_used
        ),
        "quadratic_background_cone_feasibility_relative_tolerance": (
            getattr(
                result,
                "quadratic_background_cone_feasibility_relative_tolerance",
                float("nan"),
            )
        ),
        "declared_quadratic_background_cone_feasibility_relative_tolerance": (
            declared_cone_feasibility_tolerance
        ),
        "quadratic_background_cone_activity_relative_tolerance": (
            getattr(
                result,
                "quadratic_background_cone_activity_relative_tolerance",
                float("nan"),
            )
        ),
        "declared_quadratic_background_cone_activity_relative_tolerance": (
            declared_cone_activity_tolerance
        ),
        "optimizer_scoring_iterations": getattr(
            result, "optimizer_scoring_iterations", 0
        ),
        "declared_optimizer_scoring_max_iterations": (
            declared_scoring_max_iterations
        ),
        "declared_optimizer_scoring_numerically_flat_nll_tolerance": (
            declared_scoring_flat_nll_tolerance
        ),
        "declared_optimizer_scoring_numerically_flat_gradient_reduction_factor": (
            declared_scoring_flat_gradient_reduction
        ),
        "optimizer_polish_basin_valid": polish_basin_valid,
        "optimizer_polish_basin_restart_used": getattr(
            result, "optimizer_polish_basin_restart_used", False
        ),
        "optimizer_polish_scaled_displacement_inf": getattr(
            result, "optimizer_polish_scaled_displacement_inf", float("nan")
        ),
        "optimizer_polish_stable_nll_decrease": getattr(
            result, "optimizer_polish_stable_nll_decrease", float("nan")
        ),
        "declared_optimizer_polish_max_scaled_displacement_inf": (
            declared_polish_displacement_limit
        ),
        "declared_optimizer_polish_max_stable_nll_decrease": (
            declared_polish_nll_decrease_limit
        ),
        "optimizer_stationarity_valid": stationarity_valid,
        "scaled_projected_gradient_inf_norm": getattr(
            result, "scaled_projected_gradient_inf_norm", float("nan")
        ),
        "stationarity_tolerance": getattr(
            result, "stationarity_tolerance", float("nan")
        ),
        "declared_stationarity_tolerance": declared_stationarity_tolerance,
        "optimizer_iterations": result.optimizer_iterations,
        "optimizer_evaluations": result.optimizer_evaluations,
        "data_poisson_nll_relative_to_saturated_data_constant": (
            result.data_poisson_nll
        ),
        "gaussian_nuisance_prior_deviance": (
            result.gaussian_nuisance_prior_deviance
        ),
        "calibration_prior_deviance": result.calibration_prior_deviance,
        "penalized_nll_relative_to_saturated_data_constant": result.penalized_nll,
        "poisson_deviance": result.poisson_deviance,
        "degrees_of_freedom": result.degrees_of_freedom,
        "degrees_of_freedom_convention": (
            "raw fitted bins minus all free fit parameters; calibration prior is excluded"
        ),
        "penalized_degrees_of_freedom": result.penalized_degrees_of_freedom,
        "penalized_degrees_of_freedom_convention": (
            "count-data degrees of freedom plus every declared Gaussian nuisance pseudo-observation"
        ),
        "poisson_deviance_per_dof": deviance_per_dof,
        "poisson_deviance_p_value": global_p_value,
        "poisson_deviance_p_value_semantics": (
            "asymptotic chi-square diagnostic on fitted bins whose expected count "
            "meets the declared threshold; degrees of freedom subtract all fit parameters"
        ),
        "chi_square_minimum_expected_counts_per_bin": minimum_expected_counts,
        "chi_square_reference_bin_count": eligible_bin_count,
        "excluded_low_expected_bin_count": excluded_low_expected_bin_count,
        "chi_square_reference_poisson_deviance": global_reference_deviance,
        "chi_square_reference_degrees_of_freedom": global_reference_dof,
        "global_deviance_p_value_minimum": global_alpha,
        "window_deviance_diagnostics": window_tests,
        "window_bonferroni_p_value_minimum": window_alpha,
        "per_bin_outlier_diagnostics": bin_tests,
        "per_bin_bonferroni_p_value_minimum": bin_alpha,
        "largest_absolute_poisson_residual": float(
            np.max(eligible_absolute_residual)
        ),
        "fisher_rank": result.fisher_rank,
        "fisher_condition": result.fisher_condition,
        "fisher_covariance_valid": result.fisher_covariance_valid,
        "fisher_covariance_inversion_coordinates": getattr(
            result,
            "fisher_covariance_inversion_coordinates",
            "unrecorded",
        ),
        "active_bounds": list(result.active_bounds),
        "parameters": [
            {
                "name": name,
                "value": float(value),
                "fisher_standard_deviation": float(error),
            }
            for name, value, error in zip(
                result.parameter_names, result.parameter_values, standard_deviation
            )
        ],
        "manuscript_replacement_applicable": not applicability_reasons,
        "applicability_reasons": applicability_reasons,
    }


def _bootstrap_rows(
    summary: BootstrapSummary, minimum_replicates_for_coverage: int
) -> list[dict[str, Any]]:
    coverage_status = (
        "descriptive_only_too_few_replicates"
        if summary.successful_replicates < minimum_replicates_for_coverage
        else "diagnostic_coverage_estimate"
    )
    return [
        {
            "line": name,
            "seed": summary.seed,
            "poisson_count_seed": summary.seed,
            "gaussian_pseudo_observation_seed": (
                summary.gaussian_pseudo_observation_seed
            ),
            "gaussian_pseudo_observation_parameter_count": len(
                summary.gaussian_pseudo_observation_parameter_names
            ),
            "requested_replicates": summary.requested_replicates,
            "successful_replicates": summary.successful_replicates,
            "empirical_standard_deviation_counts_per_s": format(
                float(summary.empirical_standard_deviation[index]), ".12g"
            ),
            "fisher_standard_deviation_counts_per_s": format(
                float(summary.fisher_standard_deviation[index]), ".12g"
            ),
            "fisher_68_percent_coverage": format(
                float(summary.fisher_68_percent_coverage[index]), ".12g"
            ),
            "fisher_95_percent_coverage": format(
                float(summary.fisher_95_percent_coverage[index]), ".12g"
            ),
            "boundary_fraction": format(float(summary.boundary_fraction[index]), ".12g"),
            "quadratic_background_constrained_fallback_replicates": (
                summary.quadratic_background_constrained_fallback_replicates
            ),
            "quadratic_background_exact_cone_invalid_replicates": (
                summary.quadratic_background_exact_cone_invalid_replicates
            ),
            "quadratic_background_cone_active_replicates": (
                summary.quadratic_background_cone_active_replicates
            ),
            "minimum_quadratic_background_normalized_cone_margin": format(
                summary.minimum_quadratic_background_normalized_cone_margin,
                ".12g",
            ),
            "coverage_status": coverage_status,
            "minimum_successful_replicates_for_coverage_assessment": (
                minimum_replicates_for_coverage
            ),
            "semantics": (
                summary.gaussian_pseudo_observation_convention
                + "; parametric-bootstrap diagnostic; not canonical interval; "
                "coverage is not tested below the declared replicate minimum"
            ),
        }
        for index, name in enumerate(summary.line_names)
    ]


def _bootstrap_provenance(summary: BootstrapSummary) -> dict[str, Any]:
    return {
        "poisson_count_seed": summary.seed,
        "gaussian_pseudo_observation_seed": (
            summary.gaussian_pseudo_observation_seed
        ),
        "gaussian_pseudo_observation_convention": (
            summary.gaussian_pseudo_observation_convention
        ),
        "gaussian_pseudo_observation_parameter_names": list(
            summary.gaussian_pseudo_observation_parameter_names
        ),
        "gaussian_pseudo_observation_generating_values": (
            summary.gaussian_pseudo_observation_generating_values.tolist()
        ),
        "gaussian_pseudo_observation_covariance": (
            summary.gaussian_pseudo_observation_covariance.tolist()
        ),
    }


def _profile_if_needed(
    spectra: Sequence[PublicSpectrum],
    spec: JointPeakSpec,
    calibration: CalibrationConstraint,
    resolution: LinearResolution,
    result: JointPeakFitResult,
    definition: RatioDefinition,
    weak_threshold: float,
    confidence_level: float,
    base_nll_consistency_tolerance: float,
) -> ProfileInterval | None:
    name_to_index = {name: index for index, name in enumerate(result.line_names)}
    numerator = name_to_index[definition.numerator]
    denominator = name_to_index[definition.denominator]
    if numerator == denominator:
        return profile_ratio_interval(
            spectra,
            spec,
            calibration,
            resolution,
            result,
            definition,
            confidence_level=confidence_level,
            base_nll_consistency_tolerance=(
                base_nll_consistency_tolerance
            ),
        )
    line_sd = np.sqrt(np.maximum(np.diag(result.line_rate_covariance), 0.0))
    numerator_z = (
        result.line_rates_counts_per_s[numerator] / line_sd[numerator]
        if line_sd[numerator] > 0
        else float("inf")
    )
    denominator_z = (
        result.line_rates_counts_per_s[denominator] / line_sd[denominator]
        if line_sd[denominator] > 0
        else float("inf")
    )
    active_names = {
        f"line.{definition.numerator}.rate_counts_per_s",
        f"line.{definition.denominator}.rate_counts_per_s",
    }
    if (
        min(numerator_z, denominator_z) >= weak_threshold
        and active_names.isdisjoint(result.active_bounds)
    ):
        return None
    return profile_ratio_interval(
        spectra,
        spec,
        calibration,
        resolution,
        result,
        definition,
        confidence_level=confidence_level,
        base_nll_consistency_tolerance=(
            base_nll_consistency_tolerance
        ),
    )


def _spec_with_background_model(
    spec: JointPeakSpec, background_model: str
) -> JointPeakSpec:
    return replace(
        spec,
        windows=tuple(
            replace(window, background_model=background_model)
            for window in spec.windows
        ),
    )


def _nonstandard_penalized_information_criteria(
    penalized_nll: float,
    parameter_count: int,
    observation_count: int,
) -> dict[str, str]:
    """Label AIC/BIC-shaped penalized-objective arithmetic honestly."""

    return {
        "nonstandard_penalized_objective_aic": format(
            2.0 * penalized_nll + 2.0 * parameter_count, ".12g"
        ),
        "nonstandard_penalized_objective_bic": format(
            2.0 * penalized_nll
            + parameter_count * np.log(observation_count),
            ".12g",
        ),
        "information_criterion_semantics": (
            "nonstandard descriptive arithmetic from penalized NLL including "
            "Gaussian constraint penalties; not standard AIC or BIC and not "
            "used for model promotion"
        ),
    }


def _table3_model_variant_comparison(
    spectra: Sequence[PublicSpectrum],
    config: dict[str, Any],
    spec: JointPeakSpec,
    calibration: CalibrationConstraint,
    canonical_resolution: LinearResolution,
    canonical_result: JointPeakFitResult,
) -> tuple[list[dict[str, Any]], dict[str, JointPeakFitResult]]:
    """Fit declared shape/background variants on identical raw bins."""

    variants = config["model_variants"]
    canonical_name = config["canonical_model_variant"]
    rows: list[dict[str, Any]] = []
    fitted: dict[str, JointPeakFitResult] = {}
    variant_specs: dict[str, JointPeakSpec] = {}
    for variant in variants:
        name = variant["name"]
        resolution_config = dict(config["resolution_initial"])
        resolution_config.update(variant.get("resolution_overrides", {}))
        resolution = _resolution(resolution_config)
        variant_spec = _spec_with_background_model(
            spec, variant.get("background_model", "affine")
        )
        if name == canonical_name:
            result = canonical_result
            resolution = canonical_resolution
            variant_spec = spec
        else:
            result = fit_joint_peak_model(
                spectra,
                variant_spec,
                calibration,
                resolution,
                warm_start=canonical_result,
                warm_start_source=(
                    f"Table 3 canonical model variant {canonical_name}"
                ),
            )
        fitted[name] = result
        variant_specs[name] = variant_spec
        shape_parameters = {
            parameter_name: float(parameter_value)
            for parameter_name, parameter_value in zip(
                result.parameter_names, result.parameter_values
            )
            if parameter_name.startswith("resolution.")
            or parameter_name.startswith("shape.")
        }
        parameter_count = len(result.parameter_names)
        observation_count = result.observed_counts.size
        rows.append(
            {
                "variant": name,
                "canonical": name == canonical_name,
                "resolution_form": resolution.form,
                "tail_model": resolution.tail_model,
                "background_model": variant_spec.windows[0].background_model,
                "success": result.success,
                "free_parameter_count": parameter_count,
                "raw_bin_count": observation_count,
                "penalized_nll": format(result.penalized_nll, ".12g"),
                "poisson_deviance": format(result.poisson_deviance, ".12g"),
                **_nonstandard_penalized_information_criteria(
                    result.penalized_nll,
                    parameter_count,
                    observation_count,
                ),
                "twice_delta_nll_vs_canonical": "",
                "delta_parameter_count_vs_canonical": "",
                "likelihood_ratio_reference_p_value": "",
                "comparison_semantics": "filled after all variants are fit",
                "fitted_shape_parameters_json": json.dumps(
                    shape_parameters, sort_keys=True, separators=(",", ":")
                ),
            }
        )

    canonical = fitted[canonical_name]
    canonical_parameters = len(canonical.parameter_names)
    for row in rows:
        name = str(row["variant"])
        result = fitted[name]
        delta_parameters = len(result.parameter_names) - canonical_parameters
        twice_delta_nll = 2.0 * (result.penalized_nll - canonical.penalized_nll)
        row["twice_delta_nll_vs_canonical"] = format(twice_delta_nll, ".12g")
        row["delta_parameter_count_vs_canonical"] = delta_parameters
        if (
            row["resolution_form"] == canonical_resolution.form
            and row["tail_model"] == canonical_resolution.tail_model
            and row["background_model"] == "affine"
            and spec.windows[0].background_model == "quadratic"
        ):
            improvement = max(twice_delta_nll, 0.0)
            degrees = canonical_parameters - len(result.parameter_names)
            row["likelihood_ratio_reference_p_value"] = format(
                float(chi2.sf(improvement, degrees)), ".12g"
            )
            row["comparison_semantics"] = (
                "affine background nested in the canonical quadratic Bernstein "
                "background; chi-square LRT reference"
            )
        elif name == canonical_name:
            row["comparison_semantics"] = "canonical model"
        elif row["tail_model"] != canonical_resolution.tail_model:
            row["comparison_semantics"] = (
                "tail/no-tail likelihood comparison; boundary and unidentified "
                "tail scale make ordinary chi-square LRT calibration invalid"
            )
        else:
            row["comparison_semantics"] = (
                "non-nested resolution-form likelihood comparison; no chi-square "
                "LRT p-value asserted"
            )
    return rows, fitted


def _table3_reference_window_variants(
    spectra: Sequence[PublicSpectrum],
    config: dict[str, Any],
    spec: JointPeakSpec,
    calibration: CalibrationConstraint,
    resolution: LinearResolution,
    canonical_result: JointPeakFitResult,
) -> tuple[list[dict[str, Any]], dict[str, JointPeakFitResult]]:
    """Test only predeclared, provenance-audited 558-keV model variants."""

    audit = config["reference_window_component_audit"]
    canonical_name = audit["canonical_variant"]
    results: dict[str, JointPeakFitResult] = {}
    rows: list[dict[str, Any]] = []
    for variant in audit["model_variants"]:
        name = variant["name"]
        reference_energy = float(variant["reference_energy_keV"])
        windows = spec.windows
        components = tuple(
            replace(component, energy_keV=reference_energy)
            if component.name == config["reference_component"]
            else component
            for component in spec.components
        )
        if variant["extend_for_tl208"]:
            windows = tuple(
                replace(window, high_keV=590.0)
                if window.name == "rd_558"
                else window
                for window in windows
            )
            candidate = next(
                item
                for item in audit["candidates"]
                if item["name"] == "tl208_583_187"
            )
            components += (
                LineComponent(
                    candidate["name"],
                    float(candidate["energy_keV"]),
                    "rd_558",
                    "contaminant",
                    "583.187",
                    "radioactive_decay",
                ),
            )
        variant_spec = JointPeakSpec(
            f"{spec.name}:{name}", tuple(windows), tuple(components)
        )
        result = (
            canonical_result
            if name == canonical_name
            else fit_joint_peak_model(
                spectra,
                variant_spec,
                calibration,
                resolution,
                yield_model="independent_runs",
                warm_start=canonical_result,
                warm_start_source=(
                    f"Table 3 canonical reference-window variant {canonical_name}"
                ),
            )
        )
        results[name] = result
        rows.append(
            {
                "variant": name,
                "canonical": name == canonical_name,
                "reference_energy_keV": format(reference_energy, ".12g"),
                "tl208_583_187_component": bool(variant["extend_for_tl208"]),
                "success": result.success,
                "free_parameter_count": len(result.parameter_names),
                "raw_bin_count": result.observed_counts.size,
                "penalized_nll": format(result.penalized_nll, ".12g"),
                "poisson_deviance": format(result.poisson_deviance, ".12g"),
                "twice_delta_nll_vs_canonical": (
                    format(
                        2.0
                        * (result.penalized_nll - canonical_result.penalized_nll),
                        ".12g",
                    )
                    if result.observed_counts.size
                    == canonical_result.observed_counts.size
                    else ""
                ),
                "interpretation": (
                    "predeclared nuclear-data/component sensitivity on identical bins; no automatic component promotion"
                    if result.observed_counts.size
                    == canonical_result.observed_counts.size
                    else "extended-window sensitivity uses different observations; NLL difference is not comparable and is intentionally blank"
                ),
            }
        )
    return rows, results


def _table3_independent_products(
    spectra: Sequence[PublicSpectrum],
    config: dict[str, Any],
    spec: JointPeakSpec,
    result: JointPeakFitResult,
    single_scale_result: JointPeakFitResult,
    shared_result: JointPeakFitResult,
    calibration: CalibrationConstraint,
    resolution: LinearResolution,
    reporting: dict[str, Any],
    bootstrap_replicates: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build phase-2 estimands from one fitted yield per run and component."""

    component_names = tuple(component.name for component in spec.components)
    run_count = len(spectra)
    component_count = len(component_names)
    expected_names = tuple(
        f"spectrum.{run_index}.{component_name}"
        for run_index in range(run_count)
        for component_name in component_names
    )
    if result.line_names != expected_names:
        raise RuntimeError("independent-run line ordering changed unexpectedly")
    rates = result.line_rates_counts_per_s.reshape((run_count, component_count))
    aggregate = aggregate_independent_run_rates(
        component_names,
        [spectrum.live_time for spectrum in spectra],
        rates,
        result.line_rate_covariance,
    )
    reference = str(config["reference_component"])
    definitions = tuple(
        RatioDefinition(f"{name}/{reference}", name, reference)
        for name in component_names
    )
    aggregate_ratios = ratio_values_and_covariance(
        component_names,
        aggregate.aggregate_rates_counts_per_s,
        aggregate.aggregate_rate_covariance,
        definitions,
    )
    reference_model_rows, reference_model_results = _table3_reference_window_variants(
        spectra,
        config,
        spec,
        calibration,
        resolution,
        result,
    )
    reference_ratio_vectors: list[np.ndarray] = []
    for variant_name, variant_result in reference_model_results.items():
        if (
            variant_name
            == config["reference_window_component_audit"]["canonical_variant"]
            or not variant_result.success
        ):
            continue
        variant_rates = np.asarray(
            [
                variant_result.line_rate(f"spectrum.{run_index}.{component_name}")
                for run_index in range(run_count)
                for component_name in component_names
            ]
        ).reshape((run_count, component_count))
        variant_positions = np.asarray(
            [
                variant_result.line_names.index(
                    f"spectrum.{run_index}.{component_name}"
                )
                for run_index in range(run_count)
                for component_name in component_names
            ]
        )
        variant_covariance = variant_result.line_rate_covariance[
            np.ix_(variant_positions, variant_positions)
        ]
        variant_aggregate = aggregate_independent_run_rates(
            component_names,
            [spectrum.live_time for spectrum in spectra],
            variant_rates,
            variant_covariance,
        )
        reference_ratio_vectors.append(
            ratio_values_and_covariance(
                component_names,
                variant_aggregate.aggregate_rates_counts_per_s,
                variant_aggregate.aggregate_rate_covariance,
                definitions,
            ).values
        )
    aggregate_ratio_model_covariance = np.zeros_like(aggregate_ratios.covariance)
    if reference_ratio_vectors:
        deviations = np.asarray(reference_ratio_vectors) - aggregate_ratios.values
        aggregate_ratio_model_covariance = deviations.T @ deviations / len(deviations)
    aggregate_ratio_total_covariance = (
        aggregate_ratios.covariance + aggregate_ratio_model_covariance
    )
    run_ratios = per_run_ratios(
        component_names, rates, result.line_rate_covariance, reference
    )
    component_config = {item["name"]: item for item in config["components"]}
    rate_sd = np.sqrt(np.maximum(np.diag(result.line_rate_covariance), 0.0)).reshape(
        (run_count, component_count)
    )
    active_parameters = set(result.active_bounds)

    def yield_is_on_bound(run_index: int, component_name: str) -> bool:
        return (
            f"spectrum.{run_index}.line.{component_name}.rate_counts_per_s"
            in active_parameters
        )

    per_run_rows: list[dict[str, Any]] = []
    ratio_rows: list[dict[str, Any]] = []
    for run_index, spectrum in enumerate(spectra):
        for component_index, component in enumerate(spec.components):
            flat_index = run_index * component_count + component_index
            rate_on_bound = yield_is_on_bound(run_index, component.name)
            ratio_on_bound = rate_on_bound or yield_is_on_bound(
                run_index, reference
            )
            per_run_rows.append(
                {
                    "spectrum_index": run_index,
                    "file_id": spectrum.file_id,
                    "start_time_unix_s": spectrum.metadata.get("start_time", ""),
                    "live_time_s": format(spectrum.live_time, ".12g"),
                    "component": component.name,
                    "energy_keV": format(component.energy_keV, ".12g"),
                    "identity": component_config[component.name]["identity"],
                    "origin_class": component.origin_class,
                    "detected_full_peak_rate_counts_per_s": format(
                        float(rates[run_index, component_index]), ".12g"
                    ),
                    "fisher_uncertainty_counts_per_s": format(
                        float(rate_sd[run_index, component_index]), ".12g"
                    ),
                    "fitted_counts_for_run": format(
                        float(rates[run_index, component_index] * spectrum.live_time),
                        ".12g",
                    ),
                    "boundary_status": (
                        "yield_on_bound" if rate_on_bound else "interior"
                    ),
                    "fisher_uncertainty_policy": (
                        "diagnostic_only_boundary_nonregular"
                        if rate_on_bound
                        else "regular_interior_local_approximation"
                    ),
                    "heterogeneity_policy": (
                        "excluded_from_rate_gls_boundary_nonregular"
                        if rate_on_bound
                        else "included_in_rate_gls"
                    ),
                    "estimand": "run-specific detected full-energy-peak rate; no efficiency correction or unfolding",
                }
            )
            ratio_rows.append(
                {
                    "spectrum_index": run_index,
                    "file_id": spectrum.file_id,
                    "component": component.name,
                    "reference_component": reference,
                    "ratio": format(float(run_ratios.values[flat_index]), ".12g"),
                    "fisher_uncertainty": format(
                        float(np.sqrt(max(run_ratios.covariance[flat_index, flat_index], 0.0))),
                        ".12g",
                    ),
                    "boundary_status": (
                        "numerator_or_denominator_yield_on_bound"
                        if ratio_on_bound
                        else "interior"
                    ),
                    "heterogeneity_policy": (
                        "not_applicable_algebraic_self_ratio"
                        if component.name == reference
                        else (
                            "excluded_from_ratio_gls_boundary_nonregular"
                            if ratio_on_bound
                            else "included_in_ratio_gls"
                        )
                    ),
                    "interval_policy": (
                        "exact_self_ratio"
                        if component.name == reference
                        else (
                            "diagnostic_point_only; constrained profile required before inferential use"
                            if ratio_on_bound
                            else "Fisher diagnostic; per-run ratios are secondary"
                        )
                    ),
                    "estimand": "within-run detected full-peak rate ratio; detector efficiency does not cancel across energies",
                }
            )

    aggregate_rate_sd = np.sqrt(
        np.maximum(np.diag(aggregate.aggregate_rate_covariance), 0.0)
    )
    aggregate_count_sd = np.sqrt(
        np.maximum(np.diag(aggregate.summed_count_covariance), 0.0)
    )
    aggregate_ratio_sd = np.sqrt(
        np.maximum(np.diag(aggregate_ratios.covariance), 0.0)
    )
    aggregate_ratio_model_sd = np.sqrt(
        np.maximum(np.diag(aggregate_ratio_model_covariance), 0.0)
    )
    aggregate_ratio_total_sd = np.sqrt(
        np.maximum(np.diag(aggregate_ratio_total_covariance), 0.0)
    )
    aggregate_profiles: dict[str, ProfileInterval] = {}
    live_times = np.asarray([spectrum.live_time for spectrum in spectra])
    reference_component_index = component_names.index(reference)
    for component_index, component in enumerate(spec.components):
        if component.name == reference:
            continue
        component_parameter_names = {
            f"spectrum.{run_index}.line.{component.name}.rate_counts_per_s"
            for run_index in range(run_count)
        }
        reference_parameter_names = {
            f"spectrum.{run_index}.line.{reference}.rate_counts_per_s"
            for run_index in range(run_count)
        }
        weak = (
            aggregate.aggregate_rates_counts_per_s[component_index]
            / max(aggregate_rate_sd[component_index], 1e-300)
            < float(reporting["weak_line_fisher_z_threshold"])
        )
        aggregate_boundary = all(
            name in active_parameters for name in component_parameter_names
        ) or all(name in active_parameters for name in reference_parameter_names)
        if not weak and not aggregate_boundary:
            continue
        numerator_weights = np.zeros(run_count * component_count)
        denominator_weights = np.zeros_like(numerator_weights)
        for run_index, live_time in enumerate(live_times):
            numerator_weights[
                run_index * component_count + component_index
            ] = live_time
            denominator_weights[
                run_index * component_count + reference_component_index
            ] = live_time
        aggregate_profiles[component.name] = profile_linear_ratio_interval(
            spectra,
            spec,
            calibration,
            resolution,
            result,
            numerator_weights,
            denominator_weights,
            ratio_name=definitions[component_index].name,
            yield_model="independent_runs",
            confidence_level=float(reporting["profile_confidence_level"]),
            max_evaluations=24,
            base_nll_consistency_tolerance=float(
                reporting["profile_base_nll_consistency_tolerance"]
            ),
        )
    aggregate_rows = [
        {
            "paper_table": 3,
            "component": component.name,
            "energy_keV": format(component.energy_keV, ".12g"),
            "identity": component_config[component.name]["identity"],
            "origin_class": component.origin_class,
            "summed_fitted_detector_counts": format(
                float(aggregate.summed_counts[index]), ".12g"
            ),
            "summed_counts_fisher_uncertainty": format(
                float(aggregate_count_sd[index]), ".12g"
            ),
            "aggregate_detected_rate_counts_per_s": format(
                float(aggregate.aggregate_rates_counts_per_s[index]), ".12g"
            ),
            "aggregate_rate_fisher_uncertainty_counts_per_s": format(
                float(aggregate_rate_sd[index]), ".12g"
            ),
            "aggregate_ratio_to_558_5_keV": format(
                float(aggregate_ratios.values[index]), ".12g"
            ),
            "aggregate_ratio_fisher_uncertainty": format(
                float(aggregate_ratio_sd[index]), ".12g"
            ),
            "aggregate_ratio_reference_model_rms_systematic": format(
                float(aggregate_ratio_model_sd[index]), ".12g"
            ),
            "aggregate_ratio_total_exploratory_uncertainty": format(
                float(aggregate_ratio_total_sd[index]), ".12g"
            ),
            "interval_kind": (
                f"profile_{aggregate_profiles[component.name].kind}_statistical_only_model_covariance_separate"
                if component.name in aggregate_profiles
                else (
                    "self_ratio_exact"
                    if component.name == reference
                    else "symmetric_fisher_plus_reference_model_rms"
                )
            ),
            "interval_confidence_level": reporting["profile_confidence_level"],
            "interval_lower": format(
                float(
                    aggregate_profiles[component.name].lower
                    if component.name in aggregate_profiles
                    else (
                        1.0
                        if component.name == reference
                        else max(
                            0.0,
                            aggregate_ratios.values[index]
                            - 1.959963984540054 * aggregate_ratio_total_sd[index],
                        )
                    )
                ),
                ".12g",
            ),
            "interval_upper": format(
                float(
                    aggregate_profiles[component.name].upper
                    if component.name in aggregate_profiles
                    else (
                        1.0
                        if component.name == reference
                        else aggregate_ratios.values[index]
                        + 1.959963984540054 * aggregate_ratio_total_sd[index]
                    )
                ),
                ".12g",
            ),
            "estimand": "exposure-summed fitted detector counts and their total-live-time rate; no efficiency correction or unfolding",
            "result_semantics": "new phase-2 measured-data calculation; not approved for manuscript",
        }
        for index, component in enumerate(spec.components)
    ]
    for excluded in config["excluded_components"]:
        aggregate_rows.append(
            {
                "paper_table": 3,
                "component": excluded["name"],
                "energy_keV": format(float(excluded["energy_keV"]), ".12g"),
                "identity": excluded.get("identity", excluded["name"]),
                "origin_class": excluded.get("origin_class", ""),
                "summed_fitted_detector_counts": "",
                "summed_counts_fisher_uncertainty": "",
                "aggregate_detected_rate_counts_per_s": "",
                "aggregate_rate_fisher_uncertainty_counts_per_s": "",
                "aggregate_ratio_to_558_5_keV": "",
                "aggregate_ratio_fisher_uncertainty": "",
                "aggregate_ratio_reference_model_rms_systematic": "",
                "aggregate_ratio_total_exploratory_uncertainty": "",
                "interval_kind": excluded.get(
                    "interval_kind", "unavailable_non_photopeak_model"
                ),
                "interval_confidence_level": "",
                "interval_lower": "",
                "interval_upper": "",
                "estimand": "unavailable excluded component",
                "result_semantics": excluded["reason"],
            }
        )

    def heterogeneity_run_metadata(selection: Any) -> dict[str, Any]:
        included = selection.included_indices
        excluded = selection.excluded_boundary_indices
        return {
            "input_run_count": run_count,
            "included_run_count": len(included),
            "included_spectrum_indices_json": json.dumps(
                list(included), separators=(",", ":")
            ),
            "included_file_ids_json": json.dumps(
                [spectra[index].file_id for index in included],
                separators=(",", ":"),
            ),
            "excluded_boundary_spectrum_indices_json": json.dumps(
                list(excluded), separators=(",", ":")
            ),
            "excluded_boundary_file_ids_json": json.dumps(
                [spectra[index].file_id for index in excluded],
                separators=(",", ":"),
            ),
            "test_status": selection.status,
        }

    def heterogeneity_statistics(test: Any) -> dict[str, Any]:
        if test is None:
            return {
                "gls_constant_estimate": "",
                "gls_standard_deviation": "",
                "q_statistic": "",
                "degrees_of_freedom": "",
                "p_value": "",
            }
        return {
            "gls_constant_estimate": format(test.estimate, ".12g"),
            "gls_standard_deviation": format(test.standard_deviation, ".12g"),
            "q_statistic": format(test.q_statistic, ".12g"),
            "degrees_of_freedom": test.degrees_of_freedom,
            "p_value": format(test.p_value, ".12g"),
        }

    def heterogeneity_interpretation(
        selection: Any, estimand_description: str
    ) -> str:
        if selection.heterogeneity is None:
            return (
                "unavailable: fewer than two regular interior runs remain after "
                f"boundary exclusion; {estimand_description} not tested"
            )
        if selection.excluded_boundary_indices:
            return (
                f"interior-only GLS tests {estimand_description}; boundary-pinned "
                "runs are excluded from the test but retained in per-run and "
                "aggregate products"
            )
        return f"GLS tests {estimand_description} across all interior runs"

    heterogeneity_rows: list[dict[str, Any]] = []
    rate_heterogeneity_by_component: dict[str, Any] = {}
    ratio_heterogeneity_by_component: dict[str, Any] = {}
    for component_index, component in enumerate(spec.components):
        positions = np.asarray(
            [run * component_count + component_index for run in range(run_count)]
        )
        rate_selection = gls_constant_heterogeneity_interior(
            rates[:, component_index],
            result.line_rate_covariance[np.ix_(positions, positions)],
            [
                yield_is_on_bound(run_index, component.name)
                for run_index in range(run_count)
            ],
        )
        rate_heterogeneity_by_component[component.name] = rate_selection
        heterogeneity_rows.append(
            {
                "estimand": "detected_full_peak_rate_counts_per_s",
                "component": component.name,
                **heterogeneity_statistics(rate_selection.heterogeneity),
                **heterogeneity_run_metadata(rate_selection),
                "interpretation": heterogeneity_interpretation(
                    rate_selection,
                    "equality of run-specific detected rates using fitted covariance",
                ),
            }
        )
        if component.name == reference:
            heterogeneity_rows.append(
                {
                    "estimand": "within_run_ratio_to_reference",
                    "component": component.name,
                    "gls_constant_estimate": "1",
                    "gls_standard_deviation": "0",
                    "q_statistic": "0",
                    "degrees_of_freedom": 0,
                    "p_value": "",
                    "input_run_count": run_count,
                    "included_run_count": "",
                    "included_spectrum_indices_json": "[]",
                    "included_file_ids_json": "[]",
                    "excluded_boundary_spectrum_indices_json": "[]",
                    "excluded_boundary_file_ids_json": "[]",
                    "test_status": "not_applicable_algebraic_self_ratio",
                    "interpretation": "algebraic self-ratio; not a heterogeneity test",
                }
            )
        else:
            ratio_positions = positions
            ratio_selection = gls_constant_heterogeneity_interior(
                run_ratios.values[ratio_positions],
                run_ratios.covariance[np.ix_(ratio_positions, ratio_positions)],
                [
                    yield_is_on_bound(run_index, component.name)
                    or yield_is_on_bound(run_index, reference)
                    for run_index in range(run_count)
                ],
            )
            ratio_heterogeneity_by_component[component.name] = ratio_selection
            heterogeneity_rows.append(
                {
                    "estimand": "within_run_ratio_to_reference",
                    "component": component.name,
                    **heterogeneity_statistics(ratio_selection.heterogeneity),
                    **heterogeneity_run_metadata(ratio_selection),
                    "interpretation": heterogeneity_interpretation(
                        ratio_selection,
                        "equality of within-run detected-rate ratios using full covariance",
                    ),
                }
            )

    for row in aggregate_rows:
        component_name = str(row["component"])
        rate_selection = rate_heterogeneity_by_component.get(component_name)
        ratio_selection = ratio_heterogeneity_by_component.get(component_name)
        if rate_selection is None:
            row.update(
                {
                    "rate_heterogeneity_q": "",
                    "rate_heterogeneity_dof": "",
                    "rate_heterogeneity_p_value": "",
                    "rate_heterogeneity_status": "unavailable_component_not_fitted",
                    "rate_heterogeneity_included_file_ids_json": "[]",
                    "rate_heterogeneity_excluded_boundary_file_ids_json": "[]",
                    "ratio_heterogeneity_q": "",
                    "ratio_heterogeneity_dof": "",
                    "ratio_heterogeneity_p_value": "",
                    "ratio_heterogeneity_status": "unavailable_component_not_fitted",
                    "ratio_heterogeneity_included_file_ids_json": "[]",
                    "ratio_heterogeneity_excluded_boundary_file_ids_json": "[]",
                    "time_variation_qualifier": "unavailable; component not fitted",
                    "aggregate_time_semantics": "unavailable excluded component",
                }
            )
            continue
        rate_test = rate_selection.heterogeneity
        if component_name != reference and ratio_selection is None:
            raise RuntimeError("missing non-reference ratio heterogeneity result")
        ratio_test = (
            None if ratio_selection is None else ratio_selection.heterogeneity
        )
        rate_flag = bool(
            rate_test is not None
            and np.isfinite(rate_test.p_value)
            and rate_test.p_value < 0.05
        )
        ratio_flag = bool(
            ratio_test is not None
            and np.isfinite(ratio_test.p_value)
            and ratio_test.p_value < 0.05
        )
        diagnostic_unavailable = bool(
            rate_test is None
            or (component_name != reference and ratio_test is None)
        )
        excluded_indices = list(rate_selection.excluded_boundary_indices)
        if ratio_selection is not None:
            excluded_indices.extend(ratio_selection.excluded_boundary_indices)
        excluded_indices = list(dict.fromkeys(excluded_indices))
        exclusion_prefix = (
            "interior-only GLS excludes boundary-pinned file(s) "
            + ", ".join(str(spectra[index].file_id) for index in excluded_indices)
            + "; full aggregate retains all runs; "
            if excluded_indices
            else ""
        )
        if diagnostic_unavailable:
            variation_qualifier = (
                "one or more heterogeneity diagnostics unavailable after "
                "boundary exclusion; stationarity not established"
            )
        elif rate_flag or ratio_flag:
            variation_qualifier = (
                "run-to-run heterogeneity diagnostic p<0.05; aggregate is not "
                "a stationary-rate estimate"
            )
        else:
            variation_qualifier = (
                "no unadjusted p<0.05 run-to-run heterogeneity diagnostic; "
                "stationarity still not assumed"
            )
        row.update(
            {
                "rate_heterogeneity_q": (
                    "" if rate_test is None else format(rate_test.q_statistic, ".12g")
                ),
                "rate_heterogeneity_dof": (
                    "" if rate_test is None else rate_test.degrees_of_freedom
                ),
                "rate_heterogeneity_p_value": (
                    "" if rate_test is None else format(rate_test.p_value, ".12g")
                ),
                "rate_heterogeneity_status": rate_selection.status,
                "rate_heterogeneity_included_file_ids_json": json.dumps(
                    [
                        spectra[index].file_id
                        for index in rate_selection.included_indices
                    ],
                    separators=(",", ":"),
                ),
                "rate_heterogeneity_excluded_boundary_file_ids_json": json.dumps(
                    [
                        spectra[index].file_id
                        for index in rate_selection.excluded_boundary_indices
                    ],
                    separators=(",", ":"),
                ),
                "ratio_heterogeneity_q": (
                    "0"
                    if component_name == reference
                    else (
                        ""
                        if ratio_test is None
                        else format(ratio_test.q_statistic, ".12g")
                    )
                ),
                "ratio_heterogeneity_dof": (
                    0
                    if component_name == reference
                    else (
                        "" if ratio_test is None else ratio_test.degrees_of_freedom
                    )
                ),
                "ratio_heterogeneity_p_value": (
                    ""
                    if component_name == reference or ratio_test is None
                    else format(ratio_test.p_value, ".12g")
                ),
                "ratio_heterogeneity_status": (
                    "not_applicable_algebraic_self_ratio"
                    if component_name == reference
                    else ratio_selection.status
                ),
                "ratio_heterogeneity_included_file_ids_json": (
                    "[]"
                    if component_name == reference
                    else json.dumps(
                        [
                            spectra[index].file_id
                            for index in ratio_selection.included_indices
                        ],
                        separators=(",", ":"),
                    )
                ),
                "ratio_heterogeneity_excluded_boundary_file_ids_json": (
                    "[]"
                    if component_name == reference
                    else json.dumps(
                        [
                            spectra[index].file_id
                            for index in ratio_selection.excluded_boundary_indices
                        ],
                        separators=(",", ":"),
                    )
                ),
                "time_variation_qualifier": exclusion_prefix
                + variation_qualifier,
                "aggregate_time_semantics": "exposure-specific aggregate across four consecutive runs; stationarity not assumed",
            }
        )

    start_times = np.asarray(
        [float(spectrum.metadata["start_time"]) for spectrum in spectra]
    )
    temporal_policy = config["temporal_model_policy"]
    temporal_rows = list(
        temporal_model_identifiability(
            start_times - start_times.min(),
            ["on"] * run_count,
            source_history_available=False,
            known_half_life_s=float(
                temporal_policy["known_half_life_case"]["half_life_s"]
            ),
        )
    )
    for row in temporal_rows:
        row.update(
            {
                "input_scope": "four consecutive Cycle493 reactor-on spectra",
                "power_history": "official cycle calendar supplies no interval-resolved power values",
                "calendar_source": temporal_policy["calendar_source"],
                "calendar_authority": temporal_policy["calendar_authority"],
                "calendar_doi": temporal_policy["calendar_doi"],
                "known_half_life_case_json": json.dumps(
                    temporal_policy["known_half_life_case"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "selection": (
                    "canonical" if row["model"] == "unconstrained_per_run_detected_rate" else "diagnostic_only"
                ),
            }
        )
    single_to_two_delta_parameters = (
        len(shared_result.parameter_names) - len(single_scale_result.parameter_names)
    )
    single_to_two_lrt = 2.0 * (
        single_scale_result.penalized_nll - shared_result.penalized_nll
    )
    delta_parameters = len(result.parameter_names) - len(shared_result.parameter_names)
    lrt = 2.0 * (shared_result.penalized_nll - result.penalized_nll)
    yield_model_rows = [
        {
            "model": "single_shared_scale_comparison",
            "canonical": False,
            "free_parameter_count": len(single_scale_result.parameter_names),
            "penalized_nll": format(single_scale_result.penalized_nll, ".12g"),
            "poisson_deviance": format(single_scale_result.poisson_deviance, ".12g"),
            "twice_nll_improvement_vs_previous": "",
            "delta_parameter_count": 0,
            "reference_chi_square_p_value": "",
            "semantics": "restrictive comparison; every line shares one run scale",
        },
        {
            "model": "shared_origin_scales_phase1",
            "canonical": False,
            "free_parameter_count": len(shared_result.parameter_names),
            "penalized_nll": format(shared_result.penalized_nll, ".12g"),
            "poisson_deviance": format(shared_result.poisson_deviance, ".12g"),
            "twice_nll_improvement_vs_previous": format(
                single_to_two_lrt, ".12g"
            ),
            "delta_parameter_count": single_to_two_delta_parameters,
            "reference_chi_square_p_value": format(
                float(
                    chi2.sf(
                        max(single_to_two_lrt, 0.0),
                        single_to_two_delta_parameters,
                    )
                ),
                ".12g",
            ),
            "semantics": "two origin-class run scales; nested LRT versus the single shared run scale",
        },
        {
            "model": "independent_run_yields_phase2",
            "canonical": True,
            "free_parameter_count": len(result.parameter_names),
            "penalized_nll": format(result.penalized_nll, ".12g"),
            "poisson_deviance": format(result.poisson_deviance, ".12g"),
            "twice_nll_improvement_vs_previous": format(lrt, ".12g"),
            "delta_parameter_count": delta_parameters,
            "reference_chi_square_p_value": format(
                float(chi2.sf(max(lrt, 0.0), delta_parameters)), ".12g"
            ),
            "semantics": "canonical descriptive model; chi-square LRT is diagnostic because nonnegative yield boundaries may invalidate regular calibration",
        },
    ]
    reference_audit = config["reference_window_component_audit"]
    canonical_reference = reference_audit["canonical_component"]
    reference_index = component_names.index(reference)
    reference_audit_rows: list[dict[str, Any]] = [
        {
            "candidate": reference,
            "energy_keV": format(float(canonical_reference["paper_energy_keV"]), ".12g"),
            "authoritative_energy_keV": format(
                float(canonical_reference["authoritative_energy_keV"]), ".12g"
            ),
            "nuclide_reaction": canonical_reference["nuclide_reaction"],
            "classification": canonical_reference["classification"],
            "source_plausibility": "paper-declared Cd-113 reference",
            "fit_treatment": "canonical paper energy plus explicit IAEA-energy sensitivity",
            "aggregate_fitted_rate_counts_per_s": format(
                float(aggregate.aggregate_rates_counts_per_s[reference_index]), ".12g"
            ),
            "aggregate_fisher_uncertainty_counts_per_s": format(
                float(aggregate_rate_sd[reference_index]), ".12g"
            ),
            "authoritative_sources_json": json.dumps(
                reference_audit["authoritative_sources"],
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
    ]
    tl_result = reference_model_results.get("paper_energy_plus_tl208_583")
    for candidate in reference_audit["candidates"]:
        fitted_rate = fitted_sd = float("nan")
        if (
            candidate["name"] == "tl208_583_187"
            and tl_result is not None
            and tl_result.success
        ):
            tl_positions = np.asarray(
                [
                    tl_result.line_names.index(
                        f"spectrum.{run_index}.tl208_583_187"
                    )
                    for run_index in range(run_count)
                ]
            )
            weights = np.asarray([spectrum.live_time for spectrum in spectra])
            weights /= weights.sum()
            fitted_rate = float(
                weights @ tl_result.line_rates_counts_per_s[tl_positions]
            )
            fitted_sd = float(
                np.sqrt(
                    max(
                        weights
                        @ tl_result.line_rate_covariance[
                            np.ix_(tl_positions, tl_positions)
                        ]
                        @ weights,
                        0.0,
                    )
                )
            )
        reference_audit_rows.append(
            {
                "candidate": candidate["name"],
                "energy_keV": format(float(candidate["energy_keV"]), ".12g"),
                "authoritative_energy_keV": format(
                    float(candidate["energy_keV"]), ".12g"
                ),
                "nuclide_reaction": candidate["nuclide_reaction"],
                "classification": candidate["classification"],
                "source_plausibility": candidate["source_plausibility"],
                "fit_treatment": candidate["fit_treatment"],
                "aggregate_fitted_rate_counts_per_s": (
                    "" if not np.isfinite(fitted_rate) else format(fitted_rate, ".12g")
                ),
                "aggregate_fisher_uncertainty_counts_per_s": (
                    "" if not np.isfinite(fitted_sd) else format(fitted_sd, ".12g")
                ),
                "authoritative_sources_json": json.dumps(
                    reference_audit["authoritative_sources"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    bootstrap = parametric_bootstrap(
        spectra,
        spec,
        calibration,
        resolution,
        result,
        case_identity="paper-table-3-independent-run-yields-phase2",
        replicates=bootstrap_replicates,
        yield_model="independent_runs",
    )
    products = {
        "table3_candidate.csv": aggregate_rows,
        "table3_per_run_line_rates.csv": per_run_rows,
        "table3_per_run_line_rate_covariance.csv": _covariance_rows(
            result.line_names, result.line_rate_covariance, "(counts/s)^2"
        ),
        "table3_full_parameter_covariance.csv": _parameter_covariance_rows(result),
        "table3_aggregate_count_covariance.csv": _covariance_rows(
            component_names, aggregate.summed_count_covariance, "counts^2"
        ),
        "table3_aggregate_rate_covariance.csv": _covariance_rows(
            component_names, aggregate.aggregate_rate_covariance, "(counts/s)^2"
        ),
        "table3_aggregate_ratio_covariance.csv": _covariance_rows(
            aggregate_ratios.labels, aggregate_ratios.covariance, "dimensionless^2"
        ),
        "table3_aggregate_ratio_reference_model_covariance.csv": _covariance_rows(
            aggregate_ratios.labels,
            aggregate_ratio_model_covariance,
            "dimensionless^2 (IAEA reference-energy and Tl-208 window-extension RMS)",
        ),
        "table3_aggregate_ratio_total_covariance.csv": _covariance_rows(
            aggregate_ratios.labels,
            aggregate_ratio_total_covariance,
            "dimensionless^2 (Fisher plus declared reference-window model RMS)",
        ),
        "table3_per_run_ratios.csv": ratio_rows,
        "table3_per_run_ratio_covariance.csv": _covariance_rows(
            run_ratios.labels, run_ratios.covariance, "dimensionless^2"
        ),
        "table3_run_heterogeneity.csv": heterogeneity_rows,
        "table3_temporal_model_identifiability.csv": temporal_rows,
        "table3_yield_model_comparison.csv": yield_model_rows,
        "table3_reference_window_component_audit.csv": reference_audit_rows,
        "table3_reference_window_model_comparison.csv": reference_model_rows,
        "table3_aggregate_ratio_profile_intervals.csv": [
            asdict(profile) for profile in aggregate_profiles.values()
        ]
        or [
            {
                "ratio": "none",
                "estimate": "",
                "confidence_level": reporting["profile_confidence_level"],
                "kind": "not_required",
                "lower": "",
                "upper": "",
                "threshold_delta_nll": "",
                "evaluations": 0,
                "message": "all aggregate ratios regular and interior",
            }
        ],
        "table3_bootstrap_diagnostics.csv": _bootstrap_rows(
            bootstrap,
            int(reporting["minimum_bootstrap_replicates_for_coverage_assessment"]),
        ),
        "table3_window_diagnostics.csv": _window_diagnostics(
            result,
            spectra,
            spec,
            float(reporting["chi_square_minimum_expected_counts_per_bin"]),
        ),
    }
    return products, {
        "canonical_yield_model": "independent_runs",
        "yield_model_fit_diagnostics": {
            "phase2_independent_runs": _fit_diagnostics(result, reporting),
            "phase1_origin_class_scales": _fit_diagnostics(
                shared_result, reporting
            ),
            "single_shared_run_scale": _fit_diagnostics(
                single_scale_result, reporting
            ),
        },
        "reference_window_variant_fit_diagnostics": {
            variant_name: _fit_diagnostics(variant_result, reporting)
            for variant_name, variant_result in reference_model_results.items()
        },
        "aggregate_estimand": (
            "sum_r live_time_r * fitted_detected_rate_r, with full fitted covariance; "
            "aggregate rate divides by total live time"
        ),
        "efficiency_or_unfolding_applied": False,
        "yield_model_comparison": yield_model_rows,
        "reference_window_model_comparison": reference_model_rows,
        "reference_window_promotion_policy": reference_audit["promotion_policy"],
        "aggregate_profile_intervals": [
            asdict(profile) for profile in aggregate_profiles.values()
        ],
        "temporal_model_identifiability": temporal_rows,
        "bootstrap_seed": bootstrap.seed,
        "bootstrap_pseudo_observation_provenance": (
            _bootstrap_provenance(bootstrap)
        ),
        "bootstrap_requested_replicates": bootstrap.requested_replicates,
        "bootstrap_successful_replicates": bootstrap.successful_replicates,
        "bootstrap_quadratic_background_constrained_fallback_replicates": (
            bootstrap.quadratic_background_constrained_fallback_replicates
        ),
        "bootstrap_quadratic_background_exact_cone_invalid_replicates": (
            bootstrap.quadratic_background_exact_cone_invalid_replicates
        ),
        "bootstrap_quadratic_background_cone_active_replicates": (
            bootstrap.quadratic_background_cone_active_replicates
        ),
        "bootstrap_minimum_quadratic_background_normalized_cone_margin": (
            bootstrap.minimum_quadratic_background_normalized_cone_margin
        ),
    }


def _table3_products(
    spectra: Sequence[PublicSpectrum],
    config: dict[str, Any],
    reporting: dict[str, Any],
    bootstrap_replicates: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    spec = _spec(config, "paper-table-3-measured-data-v1")
    calibration = _constraint(config["calibration_constraint"])
    resolution = _resolution(config["resolution_initial"])
    result = fit_joint_peak_model(spectra, spec, calibration, resolution)
    if not result.success:
        raise RuntimeError(f"Table 3 fit failed: {result.message}")
    single_scale_spec = replace(
        spec,
        name=f"{spec.name}:single-shared-run-scale",
        components=tuple(
            replace(component, origin_class="shared")
            for component in spec.components
        ),
    )
    single_scale_result = fit_joint_peak_model(
        spectra,
        single_scale_spec,
        calibration,
        resolution,
        warm_start=result,
        warm_start_source="Table 3 repaired phase-1 origin-class fit",
    )
    if not single_scale_result.success:
        raise RuntimeError(
            f"Table 3 single-scale comparison failed: {single_scale_result.message}"
        )
    independent_result = fit_joint_peak_model(
        spectra,
        spec,
        calibration,
        resolution,
        yield_model="independent_runs",
        warm_start=result,
        warm_start_source="Table 3 repaired phase-1 origin-class fit",
    )
    if not independent_result.success:
        raise RuntimeError(
            f"Table 3 independent-run fit failed: {independent_result.message}"
        )
    model_variants, model_variant_results = _table3_model_variant_comparison(
        spectra,
        config,
        spec,
        calibration,
        resolution,
        result,
    )
    reference = config["reference_component"]
    definitions = tuple(
        RatioDefinition(f"{component.name}/{reference}", component.name, reference)
        for component in spec.components
    )
    ratios = ratios_from_fit(result, definitions)
    reference_core_rows = _reference_core_diagnostics(
        result, spectra, config["reference_core_diagnostic"]
    )
    origin_scale_rows = _origin_scale_diagnostics(result, spectra)
    reference_minimum_live_time = float(
        config["reference_core_systematic"]["minimum_run_live_time_s"]
    )
    reference_core_eligible_rows = [
        row
        for row, spectrum in zip(reference_core_rows, spectra)
        if spectrum.live_time >= reference_minimum_live_time
    ]
    if not reference_core_eligible_rows:
        raise ValueError("no run is eligible for the reference-core systematic")
    reference_core_fractional_systematic = max(
        abs(float(row["relative_core_balance"]))
        for row in reference_core_eligible_rows
    )
    composition_minimum_live_time = float(
        config["origin_composition_systematic"]["minimum_run_live_time_s"]
    )
    composition_eligible_rows = [
        row
        for row, spectrum in zip(origin_scale_rows, spectra)
        if row["spectrum_index"] != 0
        and spectrum.live_time >= composition_minimum_live_time
    ]
    if not composition_eligible_rows:
        raise ValueError("no non-reference run is eligible for composition systematic")
    origin_composition_fractional_systematic = max(
        abs(
            float(
                row["decay_to_capture_composition_relative_to_spectrum_0"]
            )
            - 1.0
        )
        for row in composition_eligible_rows
    )
    line_sd = np.sqrt(np.maximum(np.diag(result.line_rate_covariance), 0.0))
    ratio_sd = np.sqrt(np.maximum(np.diag(ratios.covariance), 0.0))
    variant_ratio_vectors = [
        ratios_from_fit(variant_result, definitions).values
        for variant_name, variant_result in model_variant_results.items()
        if variant_name != config["canonical_model_variant"]
        and variant_result.success
    ]
    if not variant_ratio_vectors:
        ratio_model_covariance = np.zeros_like(ratios.covariance)
    else:
        ratio_deviations = np.asarray(variant_ratio_vectors) - ratios.values
        ratio_model_covariance = (
            ratio_deviations.T @ ratio_deviations / len(ratio_deviations)
        )
    reference_systematic_sensitivity = ratios.values.copy()
    for index, definition in enumerate(definitions):
        if definition.numerator == definition.denominator:
            reference_systematic_sensitivity[index] = 0.0
    ratio_reference_core_covariance = (
        reference_core_fractional_systematic**2
        * np.outer(
            reference_systematic_sensitivity,
            reference_systematic_sensitivity,
        )
    )
    composition_systematic_sensitivity = np.zeros_like(ratios.values)
    for index, component in enumerate(spec.components):
        if component.origin_class == "radioactive_decay":
            composition_systematic_sensitivity[index] = ratios.values[index]
    ratio_origin_composition_covariance = (
        origin_composition_fractional_systematic**2
        * np.outer(
            composition_systematic_sensitivity,
            composition_systematic_sensitivity,
        )
    )
    ratio_total_covariance = (
        ratios.covariance
        + ratio_model_covariance
        + ratio_reference_core_covariance
        + ratio_origin_composition_covariance
    )
    ratio_model_sd = np.sqrt(
        np.maximum(np.diag(ratio_model_covariance), 0.0)
    )
    ratio_reference_core_sd = np.sqrt(
        np.maximum(np.diag(ratio_reference_core_covariance), 0.0)
    )
    ratio_origin_composition_sd = np.sqrt(
        np.maximum(np.diag(ratio_origin_composition_covariance), 0.0)
    )
    ratio_total_sd = np.sqrt(
        np.maximum(np.diag(ratio_total_covariance), 0.0)
    )
    component_config = {item["name"]: item for item in config["components"]}
    rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    for index, (component, definition) in enumerate(zip(spec.components, definitions)):
        is_cross_origin = component.origin_class == "radioactive_decay"
        ratio_estimand = (
            "reference-run-relative radioactive-decay/neutron-capture composition; not run invariant"
            if is_cross_origin
            else "within-neutron-capture-class ratio; run scales cancel"
        )
        profile = _profile_if_needed(
            spectra,
            spec,
            calibration,
            resolution,
            result,
            definition,
            float(reporting["weak_line_fisher_z_threshold"]),
            float(reporting["profile_confidence_level"]),
            float(reporting["profile_base_nll_consistency_tolerance"]),
        )
        if profile is not None:
            profiles.append(asdict(profile))
        if definition.numerator == definition.denominator:
            interval_kind = "self_ratio_exact"
            lower = upper = 1.0
        elif profile is not None:
            interval_kind = (
                f"profile_{profile.kind}_statistical_only_"
                "systematic_covariances_tabulated_separately"
            )
            lower, upper = profile.lower, profile.upper
        else:
            interval_kind = (
                "symmetric_statistical_plus_declared_class_aware_systematics"
            )
            lower = max(
                0.0,
                ratios.values[index] - 1.959963984540054 * ratio_total_sd[index],
            )
            upper = ratios.values[index] + 1.959963984540054 * ratio_total_sd[index]
        rows.append(
            {
                "paper_table": 3,
                "component": component.name,
                "energy_keV": format(component.energy_keV, ".12g"),
                "identity": component_config[component.name]["identity"],
                "origin_class": component.origin_class,
                "declared_window": component.window,
                "line_rate_counts_per_s": format(
                    float(result.line_rates_counts_per_s[index]), ".12g"
                ),
                "line_rate_fisher_uncertainty_counts_per_s": format(
                    float(line_sd[index]), ".12g"
                ),
                "ratio_to_558_5_keV": format(float(ratios.values[index]), ".12g"),
                "ratio_fisher_uncertainty": format(float(ratio_sd[index]), ".12g"),
                "ratio_model_variant_rms_systematic": format(
                    float(ratio_model_sd[index]), ".12g"
                ),
                "ratio_reference_core_systematic": format(
                    float(ratio_reference_core_sd[index]), ".12g"
                ),
                "ratio_origin_composition_systematic": format(
                    float(ratio_origin_composition_sd[index]), ".12g"
                ),
                "ratio_total_exploratory_uncertainty": format(
                    float(ratio_total_sd[index]), ".12g"
                ),
                "interval_kind": interval_kind,
                "interval_confidence_level": reporting["profile_confidence_level"],
                "interval_lower": format(float(lower), ".12g"),
                "interval_upper": format(float(upper), ".12g"),
                "ratio_estimand": ratio_estimand,
                "result_semantics": (
                    "new exploratory measured-data calculation; cross-origin ratio is a manuscript blocker"
                    if is_cross_origin
                    else "new exploratory measured-data calculation"
                ),
            }
        )
    for excluded in config["excluded_components"]:
        rows.append(
            {
                "paper_table": 3,
                "component": excluded["name"],
                "energy_keV": format(float(excluded["energy_keV"]), ".12g"),
                "identity": excluded.get("identity", excluded["name"]),
                "origin_class": excluded.get("origin_class", ""),
                "declared_window": "",
                "line_rate_counts_per_s": "",
                "line_rate_fisher_uncertainty_counts_per_s": "",
                "ratio_to_558_5_keV": "",
                "ratio_fisher_uncertainty": "",
                "ratio_model_variant_rms_systematic": "",
                "ratio_reference_core_systematic": "",
                "ratio_origin_composition_systematic": "",
                "ratio_total_exploratory_uncertainty": "",
                "interval_kind": excluded.get(
                    "interval_kind", "unavailable_non_photopeak_model"
                ),
                "interval_confidence_level": "",
                "interval_lower": "",
                "interval_upper": "",
                "ratio_estimand": "unavailable excluded component",
                "result_semantics": excluded["reason"],
            }
        )
    products = {
        "table3_candidate.csv": rows,
        "table3_line_rate_covariance.csv": _covariance_rows(
            result.line_names,
            result.line_rate_covariance,
            "(counts/s)^2",
        ),
        "table3_full_parameter_covariance.csv": _parameter_covariance_rows(result),
        "table3_ratio_covariance.csv": _covariance_rows(
            ratios.labels, ratios.covariance, "dimensionless^2"
        ),
        "table3_ratio_model_variant_covariance.csv": _covariance_rows(
            ratios.labels,
            ratio_model_covariance,
            "dimensionless^2 (declared model-variant RMS sensitivity)",
        ),
        "table3_ratio_reference_core_systematic_covariance.csv": _covariance_rows(
            ratios.labels,
            ratio_reference_core_covariance,
            "dimensionless^2 (declared reference-core residual envelope)",
        ),
        "table3_ratio_origin_composition_systematic_covariance.csv": _covariance_rows(
            ratios.labels,
            ratio_origin_composition_covariance,
            "dimensionless^2 (declared cross-origin run-composition envelope)",
        ),
        "table3_ratio_total_covariance.csv": _covariance_rows(
            ratios.labels,
            ratio_total_covariance,
            "dimensionless^2 (Fisher statistical plus model-variant RMS plus reference-core and origin-composition systematics)",
        ),
        "table3_window_diagnostics.csv": _window_diagnostics(
            result,
            spectra,
            spec,
            float(reporting["chi_square_minimum_expected_counts_per_bin"]),
        ),
        "table3_reference_core_diagnostics.csv": reference_core_rows,
        "table3_origin_scale_diagnostics.csv": origin_scale_rows,
        "table3_model_variant_comparison.csv": model_variants,
    }
    diagnostics = _fit_diagnostics(result, reporting)
    diagnostics["profile_intervals"] = profiles
    diagnostics["bootstrap_status"] = (
        "not run for phase1 shared-scale comparison; bootstrap budget is applied "
        "only to the canonical phase2 independent-run fit"
    )
    diagnostics["canonical_model_variant"] = config["canonical_model_variant"]
    diagnostics["run_drift_nuisance_constraints"] = {
        "relative_convention": (
            "spectrum index 0 fixed at zero calibration deviation and unit "
            "resolution scale; independent constrained deviations for later spectra"
        ),
        "per_run_calibration_deviation_covariance": config[
            "calibration_constraint"
        ]["per_run_deviation_covariance"],
        "per_run_calibration_offset_bounds_keV": config[
            "calibration_constraint"
        ]["per_run_offset_bounds_keV"],
        "per_run_fractional_stretch_bounds": config[
            "calibration_constraint"
        ]["per_run_stretch_bounds"],
        "per_run_resolution_scale_sigma": config["resolution_initial"][
            "per_run_scale_sigma"
        ],
        "per_run_resolution_scale_bounds": config["resolution_initial"][
            "per_run_scale_bounds"
        ],
        "assumption": (
            "release supplies no run-drift covariance; these are declared "
            "applicability/sensitivity assumptions, not released metrology"
        ),
    }
    diagnostics["reference_core_diagnostics"] = products[
        "table3_reference_core_diagnostics.csv"
    ]
    diagnostics["origin_scale_diagnostics"] = origin_scale_rows
    diagnostics["origin_class_policy"] = config["origin_class_policy"]
    diagnostics["reference_core_systematic"] = {
        **config["reference_core_systematic"],
        "fractional_scale": reference_core_fractional_systematic,
        "eligible_file_ids": [
            int(row["file_id"]) for row in reference_core_eligible_rows
        ],
        "added_to_total_ratio_covariance": True,
    }
    diagnostics["origin_composition_systematic"] = {
        **config["origin_composition_systematic"],
        "fractional_scale": origin_composition_fractional_systematic,
        "eligible_file_ids": [
            int(row["file_id"]) for row in composition_eligible_rows
        ],
        "added_to_total_ratio_covariance": True,
    }
    diagnostics["residual_provenance_notes"] = config.get(
        "residual_provenance_notes", []
    )
    diagnostics["model_variant_comparison"] = model_variants
    diagnostics["model_variant_fit_diagnostics"] = {
        variant_name: _fit_diagnostics(variant_result, reporting)
        for variant_name, variant_result in model_variant_results.items()
    }
    diagnostics["ratio_model_variant_systematic"] = {
        "definition": (
            "positive-semidefinite RMS outer-product covariance of each declared "
            "successful alternative model's ratio shift from the canonical fit"
        ),
        "included_alternative_variants": [
            name
            for name, variant_result in model_variant_results.items()
            if name != config["canonical_model_variant"] and variant_result.success
        ],
        "canonical_fisher_covariance_kept_separate": True,
        "total_covariance_definition": (
            "canonical Fisher statistical covariance plus declared model-variant "
            "RMS sensitivity covariance plus declared reference-core residual-envelope "
            "covariance plus cross-origin run-composition covariance"
        ),
        "maximum_model_systematic_to_fisher_sd_ratio": float(
            np.max(
                np.divide(
                    ratio_model_sd,
                    ratio_sd,
                    out=np.zeros_like(ratio_model_sd),
                    where=ratio_sd > 0,
                )
            )
        ),
    }
    phase2_products, phase2_diagnostics = _table3_independent_products(
        spectra,
        config,
        spec,
        independent_result,
        single_scale_result,
        result,
        calibration,
        resolution,
        reporting,
        bootstrap_replicates,
    )
    products["table3_phase1_shared_scale_candidate.csv"] = products.pop(
        "table3_candidate.csv"
    )
    products["table3_phase1_shared_scale_line_rate_covariance.csv"] = products.pop(
        "table3_line_rate_covariance.csv"
    )
    products["table3_phase1_shared_scale_full_parameter_covariance.csv"] = products.pop(
        "table3_full_parameter_covariance.csv"
    )
    products["table3_phase1_shared_scale_ratio_covariance.csv"] = products.pop(
        "table3_ratio_covariance.csv"
    )
    for old_name in (
        "table3_ratio_model_variant_covariance.csv",
        "table3_ratio_reference_core_systematic_covariance.csv",
        "table3_ratio_origin_composition_systematic_covariance.csv",
        "table3_ratio_total_covariance.csv",
        "table3_reference_core_diagnostics.csv",
        "table3_origin_scale_diagnostics.csv",
        "table3_model_variant_comparison.csv",
    ):
        phase1_name = old_name.replace(
            "table3_", "table3_phase1_shared_scale_", 1
        )
        products[phase1_name] = products.pop(old_name)
    products.update(phase2_products)
    phase1_diagnostics = {
        key: value
        for key, value in diagnostics.items()
        if key not in {"fit_result", "spec"}
    }
    diagnostics = _fit_diagnostics(independent_result, reporting)
    diagnostics["phase1_shared_scale_diagnostics"] = phase1_diagnostics
    diagnostics["phase2"] = phase2_diagnostics
    diagnostics["historical_reconstruction_record"] = (
        "config/table3_historical_reconstruction.json"
    )
    diagnostics["fit_result"] = independent_result
    diagnostics["spec"] = spec
    return products, diagnostics


def _table8_ratio_definitions(
    spec: JointPeakSpec, target_parents: Sequence[str] | None = None
) -> tuple[RatioDefinition, ...]:
    by_parent: dict[str, dict[str, str]] = {}
    for component in spec.components:
        by_parent.setdefault(component.parent, {})[component.role] = component.name
    definitions: list[RatioDefinition] = []
    selected = set(by_parent) if target_parents is None else set(target_parents)
    for parent in sorted(selected, key=float):
        if parent not in by_parent:
            raise ValueError(f"Table 8 target parent {parent} is absent")
        roles = by_parent[parent]
        if set(roles) != {"fep", "sep", "dep"}:
            raise ValueError(f"Table 8 parent {parent} lacks one declared target role")
        definitions.extend(
            (
                RatioDefinition(f"{parent}:fep/sep", roles["fep"], roles["sep"]),
                RatioDefinition(f"{parent}:fep/dep", roles["fep"], roles["dep"]),
                RatioDefinition(f"{parent}:sep/dep", roles["sep"], roles["dep"]),
            )
        )
    return tuple(definitions)


def _assert_table8_canonical_fit_identity(
    result: JointPeakFitResult,
    spec: JointPeakSpec,
    resolution: LinearResolution,
) -> None:
    """Fail if downstream Table 8 inputs do not reconstruct the fitted model."""

    expected_line_names = tuple(component.name for component in spec.components)
    if result.line_names != expected_line_names:
        raise RuntimeError(
            "canonical Table 8 result/component identity differs from the "
            "spec passed to profile and bootstrap calculations"
        )
    parameter_names = set(result.parameter_names)
    expected_slope = (
        "resolution.linear_sigma_slope_keV_per_keV"
        if resolution.form == "linear"
        else "resolution.variance_slope_keV"
    )
    other_slope = (
        "resolution.variance_slope_keV"
        if resolution.form == "linear"
        else "resolution.linear_sigma_slope_keV_per_keV"
    )
    if expected_slope not in parameter_names or other_slope in parameter_names:
        raise RuntimeError(
            "canonical Table 8 result/resolution-form identity differs from "
            "the resolution passed to profile and bootstrap calculations"
        )
    tail_names = {
        "shape.low_energy_tail_fraction",
        "shape.low_energy_tail_scale_in_sigma",
    }
    fitted_tail_names = tail_names.intersection(parameter_names)
    expected_tail_names = (
        tail_names if resolution.tail_model == "constant" else set()
    )
    if fitted_tail_names != expected_tail_names:
        raise RuntimeError(
            "canonical Table 8 result/tail-model identity differs from the "
            "resolution passed to profile and bootstrap calculations"
        )
    for window in spec.windows:
        middle_name = (
            f"background.{window.name}.middle_counts_per_s_per_keV"
        )
        if (middle_name in parameter_names) != (
            window.background_model == "quadratic"
        ):
            raise RuntimeError(
                "canonical Table 8 result/background identity differs from "
                "the spec passed to profile and bootstrap calculations"
            )


def _table8_guarded_restart_settings(
    variant_name: str,
    guarded_restart_rules: dict[str, dict[str, Any]],
) -> tuple[bool, str]:
    """Resolve the frozen per-variant restart rule without name heuristics."""

    restart_rule = guarded_restart_rules.get(variant_name, {})
    allowed = bool(restart_rule.get("allowed", False))
    source = str(restart_rule.get("initial_source", ""))
    if allowed != bool(source):
        raise ValueError(
            f"Table 8 guarded restart rule for {variant_name!r} must declare "
            "both allowed=true and a nonempty initial_source"
        )
    return allowed, source


def _table8_component_variants(
    spectra: Sequence[PublicSpectrum],
    config: dict[str, Any],
    base_spec: JointPeakSpec,
    calibration: CalibrationConstraint,
    resolution: LinearResolution,
) -> tuple[
    list[dict[str, Any]],
    dict[str, JointPeakFitResult],
    dict[str, JointPeakSpec],
    dict[str, LinearResolution],
]:
    audit = config["component_audit"]
    candidates = {item["name"]: item for item in audit["candidates"]}
    specs: dict[str, JointPeakSpec] = {}
    variant_resolutions: dict[str, LinearResolution] = {}
    variant_metadata: dict[str, dict[str, Any]] = {}
    guarded_restart_rules: dict[str, dict[str, Any]] = {}
    for variant in audit["variants"]:
        additions = tuple(
            LineComponent(
                name,
                float(candidates[name]["energy_keV"]),
                str(candidates[name]["window"]),
                "contaminant",
                str(candidates[name]["parent"]),
                "shared",
            )
            for name in variant["candidate_components"]
        )
        variant_spec = replace(
            base_spec,
            name=f"{base_spec.name}:{variant['name']}",
            components=base_spec.components + additions,
        )
        specs[variant["name"]] = variant_spec
        variant_resolutions[variant["name"]] = resolution
        variant_metadata[variant["name"]] = {
            "candidate_components": list(variant["candidate_components"]),
            "background_model": "affine",
            "tail_model": resolution.tail_model,
            "comparison_kind": "declared component set",
        }
        guarded_restart_rules[variant["name"]] = dict(
            variant.get("guarded_basin_restart", {})
        )

    canonical_component_name = config["canonical_component_variant"]
    canonical_spec = specs[canonical_component_name]
    no_tail_name = f"{canonical_component_name}_no_tail"
    no_tail_resolution = replace(resolution, tail_model="none")
    specs[no_tail_name] = canonical_spec
    variant_resolutions[no_tail_name] = no_tail_resolution
    variant_metadata[no_tail_name] = {
        "candidate_components": variant_metadata[canonical_component_name][
            "candidate_components"
        ],
        "background_model": "affine",
        "tail_model": "none",
        "comparison_kind": "line-shape sensitivity",
    }
    quadratic_name = f"{canonical_component_name}_quadratic_background"
    quadratic_spec = _spec_with_background_model(canonical_spec, "quadratic")
    specs[quadratic_name] = quadratic_spec
    variant_resolutions[quadratic_name] = resolution
    variant_metadata[quadratic_name] = {
        "candidate_components": variant_metadata[canonical_component_name][
            "candidate_components"
        ],
        "background_model": "quadratic",
        "tail_model": resolution.tail_model,
        "comparison_kind": "background-shape sensitivity",
    }

    canonical_model_name = config["canonical_model_variant"]
    if canonical_model_name not in specs:
        raise ValueError("canonical Table 8 model variant was not generated")
    canonical_restart_allowed, canonical_restart_source = (
        _table8_guarded_restart_settings(
            canonical_model_name, guarded_restart_rules
        )
    )
    canonical = fit_joint_peak_model(
        spectra,
        specs[canonical_model_name],
        calibration,
        variant_resolutions[canonical_model_name],
        allow_guarded_basin_restart=canonical_restart_allowed,
        guarded_basin_restart_source=canonical_restart_source,
    )
    results: dict[str, JointPeakFitResult] = {}
    for name, variant_spec in specs.items():
        if name == canonical_model_name:
            results[name] = canonical
            continue
        use_default_restart_rule, restart_source = (
            _table8_guarded_restart_settings(name, guarded_restart_rules)
        )
        results[name] = fit_joint_peak_model(
            spectra,
            variant_spec,
            calibration,
            variant_resolutions[name],
            warm_start=(
                canonical
                if canonical.success and not use_default_restart_rule
                else None
            ),
            warm_start_source=(
                f"Table 8 canonical model variant {canonical_model_name}"
                if canonical.success and not use_default_restart_rule
                else ""
            ),
            allow_guarded_basin_restart=use_default_restart_rule,
            guarded_basin_restart_source=restart_source,
        )
    rows: list[dict[str, Any]] = []
    for name, result in results.items():
        parameter_count = len(result.parameter_names)
        observation_count = result.observed_counts.size
        rows.append(
            {
                "variant": name,
                "canonical": name == canonical_model_name,
                "candidate_components_json": json.dumps(
                    variant_metadata[name]["candidate_components"],
                    separators=(",", ":"),
                ),
                "background_model": variant_metadata[name]["background_model"],
                "tail_model": variant_metadata[name]["tail_model"],
                "comparison_kind": variant_metadata[name]["comparison_kind"],
                "success": result.success,
                "free_parameter_count": parameter_count,
                "raw_bin_count": observation_count,
                "penalized_nll": format(result.penalized_nll, ".12g"),
                "poisson_deviance": format(result.poisson_deviance, ".12g"),
                "calibration_offset_keV": format(
                    result.parameter("calibration.offset_keV"), ".12g"
                ),
                "calibration_fractional_gain_stretch": format(
                    result.parameter("calibration.fractional_gain_stretch"),
                    ".12g",
                ),
                **_nonstandard_penalized_information_criteria(
                    result.penalized_nll,
                    parameter_count,
                    observation_count,
                ),
                "twice_delta_nll_vs_canonical": format(
                    2.0 * (result.penalized_nll - canonical.penalized_nll),
                    ".12g",
                ),
                "interpretation": (
                    "sensitivity comparison; component boundaries and non-nested variants preclude automatic chi-square promotion"
                ),
            }
        )
    return rows, results, specs, variant_resolutions


def _table8_products(
    spectrum: PublicSpectrum,
    config: dict[str, Any],
    reporting: dict[str, Any],
    bootstrap_replicates: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    spectra = (spectrum,)
    base_spec = _spec(config, "paper-table-8-measured-data-v2")
    calibration = _constraint(config["calibration_constraint"])
    base_resolution = _resolution(config["resolution_initial"])
    (
        model_rows,
        model_results,
        model_specs,
        model_resolutions,
    ) = _table8_component_variants(
        spectra, config, base_spec, calibration, base_resolution
    )
    canonical_name = config["canonical_model_variant"]
    canonical_component_name = config["canonical_component_variant"]
    result = model_results[canonical_name]
    spec = model_specs[canonical_name]
    resolution = model_resolutions[canonical_name]
    _assert_table8_canonical_fit_identity(result, spec, resolution)
    if not result.success:
        raise RuntimeError(f"Table 8 fit failed: {result.message}")
    definitions = _table8_ratio_definitions(
        spec, config["target_parents_keV"]
    )
    ratios = ratios_from_fit(result, definitions)
    ratio_sd = np.sqrt(np.maximum(np.diag(ratios.covariance), 0.0))
    all_variant_fit_diagnostics = {
        variant_name: _fit_diagnostics(variant_result, reporting)
        for variant_name, variant_result in model_results.items()
    }
    variant_fit_diagnostics = {
        variant_name: variant_diagnostic
        for variant_name, variant_diagnostic in all_variant_fit_diagnostics.items()
        if model_results[variant_name].success
    }
    for model_row in model_rows:
        variant_diagnostic = variant_fit_diagnostics.get(str(model_row["variant"]))
        model_row["fit_quality_acceptable"] = bool(
            variant_diagnostic
            and variant_diagnostic["manuscript_replacement_applicable"]
        )
        model_row["fit_quality_reasons_json"] = json.dumps(
            (
                variant_diagnostic["applicability_reasons"]
                if variant_diagnostic is not None
                else ["optimizer did not converge"]
            ),
            separators=(",", ":"),
        )
    variant_ratio_values = {
        variant_name: ratios_from_fit(variant_result, definitions).values
        for variant_name, variant_result in model_results.items()
        if variant_name != canonical_name and variant_result.success
    }
    all_variant_names = list(variant_ratio_values)
    acceptable_variant_names = [
        name
        for name in all_variant_names
        if variant_fit_diagnostics[name]["manuscript_replacement_applicable"]
    ]
    ratio_model_covariance = np.zeros_like(ratios.covariance)
    all_variant_deviations = np.empty((0, len(definitions)), dtype=np.float64)
    if all_variant_names:
        all_variant_deviations = (
            np.asarray([variant_ratio_values[name] for name in all_variant_names])
            - ratios.values
        )
        deviations = all_variant_deviations
        ratio_model_covariance = deviations.T @ deviations / len(deviations)
    ratio_acceptable_model_covariance: np.ndarray | None = None
    if acceptable_variant_names:
        acceptable_deviations = (
            np.asarray(
                [variant_ratio_values[name] for name in acceptable_variant_names]
            )
            - ratios.values
        )
        ratio_acceptable_model_covariance = (
            acceptable_deviations.T
            @ acceptable_deviations
            / len(acceptable_deviations)
        )
    ratio_total_covariance = ratios.covariance + ratio_model_covariance
    ratio_model_sd = np.sqrt(
        np.maximum(np.diag(ratio_model_covariance), 0.0)
    )
    ratio_acceptable_model_sd = (
        np.sqrt(
            np.maximum(np.diag(ratio_acceptable_model_covariance), 0.0)
        )
        if ratio_acceptable_model_covariance is not None
        else np.full(len(definitions), np.nan)
    )
    ratio_total_sd = np.sqrt(
        np.maximum(np.diag(ratio_total_covariance), 0.0)
    )
    all_variant_driver: dict[str, Any] | None = None
    if all_variant_names:
        systematic_to_fisher = np.divide(
            ratio_model_sd,
            ratio_sd,
            out=np.zeros_like(ratio_model_sd),
            where=ratio_sd > 0,
        )
        driver_ratio_index = int(np.argmax(systematic_to_fisher))
        driver_variant_index = int(
            np.argmax(np.abs(all_variant_deviations[:, driver_ratio_index]))
        )
        all_variant_driver = {
            "variant": all_variant_names[driver_variant_index],
            "ratio": ratios.labels[driver_ratio_index],
            "absolute_ratio_shift": float(
                abs(all_variant_deviations[driver_variant_index, driver_ratio_index])
            ),
            "all_variant_rms_to_fisher_sd_ratio": float(
                systematic_to_fisher[driver_ratio_index]
            ),
        }

    al_variant_name = "al27_6711_alternative"
    ge_variant_name = "target_plus_fe_cu_ge70"
    al_result = model_results[al_variant_name]
    ge_result = model_results[ge_variant_name]
    al_ge_discrimination_rows = [
        {
            "al27_variant": al_variant_name,
            "ge70_variant": ge_variant_name,
            "same_raw_bin_count": (
                al_result.observed_counts.size == ge_result.observed_counts.size
            ),
            "same_free_parameter_count": (
                len(al_result.parameter_names) == len(ge_result.parameter_names)
            ),
            "al27_penalized_nll": format(al_result.penalized_nll, ".12g"),
            "ge70_penalized_nll": format(ge_result.penalized_nll, ".12g"),
            "twice_nll_al27_minus_ge70": format(
                2.0 * (al_result.penalized_nll - ge_result.penalized_nll),
                ".12g",
            ),
            "difference_direction": "positive favors the Ge-70 variant; descriptive non-nested comparison",
            "al27_calibration_offset_keV": format(
                al_result.parameter("calibration.offset_keV"), ".12g"
            ),
            "ge70_calibration_offset_keV": format(
                ge_result.parameter("calibration.offset_keV"), ".12g"
            ),
            "al27_calibration_fractional_gain_stretch": format(
                al_result.parameter("calibration.fractional_gain_stretch"),
                ".12g",
            ),
            "ge70_calibration_fractional_gain_stretch": format(
                ge_result.parameter("calibration.fractional_gain_stretch"),
                ".12g",
            ),
            "al27_independent_presence_evidence": "Al-27 target parents at 7693.398 and 7724.034 keV occur in Table 8",
            "ge70_independent_presence_evidence": "HPGe self-capture is physically plausible but not independently established in this spectrum",
            "target_measurand_overlap": "Al-27 at 6710.700 keV and Ge-70 at 6707.450 keV both reallocate the 6702.034-keV target DEP",
            "canonical_policy": "neither ambiguous component promoted; both retained as equal-status sensitivities",
        }
    ]
    rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    for index, definition in enumerate(definitions):
        profile = _profile_if_needed(
            spectra,
            spec,
            calibration,
            resolution,
            result,
            definition,
            float(reporting["weak_line_fisher_z_threshold"]),
            float(reporting["profile_confidence_level"]),
            float(reporting["profile_base_nll_consistency_tolerance"]),
        )
        if profile is not None:
            profiles.append(asdict(profile))
            interval_kind = f"profile_{profile.kind}"
            lower, upper = profile.lower, profile.upper
        else:
            interval_kind = "symmetric_fisher_plus_all_declared_model_rms"
            lower = max(0.0, ratios.values[index] - 1.959963984540054 * ratio_total_sd[index])
            upper = ratios.values[index] + 1.959963984540054 * ratio_total_sd[index]
        parent, ratio_name = definition.name.split(":", 1)
        rows.append(
            {
                "paper_table": 8,
                "file_id": spectrum.file_id,
                "parent_energy_keV": parent,
                "ratio": ratio_name,
                "numerator_component": definition.numerator,
                "denominator_component": definition.denominator,
                "ratio_value": format(float(ratios.values[index]), ".12g"),
                "ratio_fisher_uncertainty": format(float(ratio_sd[index]), ".12g"),
                "ratio_model_variant_rms_systematic": format(
                    float(ratio_model_sd[index]), ".12g"
                ),
                "ratio_all_declared_variant_rms_systematic": format(
                    float(ratio_model_sd[index]), ".12g"
                ),
                "ratio_acceptable_variant_rms_systematic": (
                    ""
                    if not np.isfinite(ratio_acceptable_model_sd[index])
                    else format(float(ratio_acceptable_model_sd[index]), ".12g")
                ),
                "acceptable_variant_systematic_status": (
                    "available"
                    if acceptable_variant_names
                    else "unavailable; no noncanonical variant passes declared fit-quality applicability checks"
                ),
                "ratio_total_exploratory_uncertainty": format(
                    float(ratio_total_sd[index]), ".12g"
                ),
                "interval_kind": interval_kind,
                "interval_confidence_level": reporting["profile_confidence_level"],
                "interval_lower": format(float(lower), ".12g"),
                "interval_upper": format(float(upper), ".12g"),
                "simulation_status": "not evaluated; missing monoenergetic inputs",
                "result_semantics": "new exploratory measured-data calculation",
            }
        )
    line_sd = np.sqrt(np.maximum(np.diag(result.line_rate_covariance), 0.0))
    line_rows = [
        {
            "component": component.name,
            "parent_energy_keV": component.parent,
            "role": component.role,
            "line_energy_keV": format(component.energy_keV, ".12g"),
            "declared_window": component.window,
            "line_rate_counts_per_s": format(
                float(result.line_rates_counts_per_s[index]), ".12g"
            ),
            "line_rate_fisher_uncertainty_counts_per_s": format(
                float(line_sd[index]), ".12g"
            ),
        }
        for index, component in enumerate(spec.components)
    ]
    audit_config = config["component_audit"]
    audit_rows: list[dict[str, Any]] = []
    for candidate in audit_config["candidates"]:
        candidate_name = candidate["name"]
        selected_variant = next(
            (
                variant["name"]
                for variant in audit_config["variants"]
                if candidate_name in variant["candidate_components"]
                and model_results[variant["name"]].success
            ),
            "",
        )
        selected_result = model_results.get(selected_variant)
        if selected_result is not None and candidate_name in selected_result.line_names:
            candidate_index = selected_result.line_names.index(candidate_name)
            fitted_rate = selected_result.line_rates_counts_per_s[candidate_index]
            fitted_sd = np.sqrt(
                max(selected_result.line_rate_covariance[candidate_index, candidate_index], 0.0)
            )
            target_only = model_results["target_only"]
            twice_improvement = 2.0 * (
                target_only.penalized_nll - selected_result.penalized_nll
            )
        else:
            fitted_rate = fitted_sd = twice_improvement = float("nan")
        audit_rows.append(
            {
                "candidate": candidate_name,
                "energy_keV": format(float(candidate["energy_keV"]), ".12g"),
                "nuclide_reaction": candidate["nuclide_reaction"],
                "classification": candidate["classification"],
                "source_plausibility": candidate["source_plausibility"],
                "residual_evidence": candidate["residual_evidence"],
                "authoritative_source": audit_config["authoritative_nuclear_source"]["url"],
                "included_in_canonical": candidate_name
                in next(
                    variant["candidate_components"]
                    for variant in audit_config["variants"]
                    if variant["name"] == canonical_component_name
                ),
                "representative_variant": selected_variant,
                "representative_fitted_rate_counts_per_s": (
                    "" if not np.isfinite(fitted_rate) else format(float(fitted_rate), ".12g")
                ),
                "representative_fisher_uncertainty_counts_per_s": (
                    "" if not np.isfinite(fitted_sd) else format(float(fitted_sd), ".12g")
                ),
                "twice_nll_improvement_vs_target_only_for_representative_variant": (
                    ""
                    if not np.isfinite(twice_improvement)
                    else format(float(twice_improvement), ".12g")
                ),
                "promotion_rule": "requires authoritative energy, plausible material source, residual support, and stable target ratios",
            }
        )
    bootstrap = parametric_bootstrap(
        spectra,
        spec,
        calibration,
        resolution,
        result,
        case_identity="paper-table-8-measured-data-v1",
        replicates=bootstrap_replicates,
    )
    products = {
        "table8_candidate.csv": rows,
        "table8_line_rates.csv": line_rows,
        "table8_line_rate_covariance.csv": _covariance_rows(
            result.line_names, result.line_rate_covariance, "(counts/s)^2"
        ),
        "table8_full_parameter_covariance.csv": _parameter_covariance_rows(result),
        "table8_ratio_covariance.csv": _covariance_rows(
            ratios.labels, ratios.covariance, "dimensionless^2"
        ),
        "table8_ratio_model_variant_covariance.csv": _covariance_rows(
            ratios.labels,
            ratio_model_covariance,
            "dimensionless^2 (all declared component/shape/background variants, including rejected fits)",
        ),
        "table8_ratio_acceptable_model_variant_covariance.csv": (
            _covariance_rows(
                ratios.labels,
                ratio_acceptable_model_covariance,
                "dimensionless^2 (fit-quality-acceptable declared variants only)",
            )
            if ratio_acceptable_model_covariance is not None
            else [
                {
                    "row": row_label,
                    "column": column_label,
                    "covariance": "",
                    "unit": "unavailable; no noncanonical variant passes declared fit-quality applicability checks",
                }
                for row_label in ratios.labels
                for column_label in ratios.labels
            ]
        ),
        "table8_ratio_total_covariance.csv": _covariance_rows(
            ratios.labels,
            ratio_total_covariance,
            "dimensionless^2 (Fisher plus declared model RMS)",
        ),
        "table8_component_audit.csv": audit_rows,
        "table8_al27_ge70_discrimination.csv": al_ge_discrimination_rows,
        "table8_model_variant_comparison.csv": model_rows,
        "table8_window_diagnostics.csv": _window_diagnostics(
            result,
            spectra,
            spec,
            float(reporting["chi_square_minimum_expected_counts_per_bin"]),
        ),
        "table8_bootstrap_diagnostics.csv": _bootstrap_rows(
            bootstrap,
            int(reporting["minimum_bootstrap_replicates_for_coverage_assessment"]),
        ),
    }
    diagnostics = _fit_diagnostics(result, reporting)
    diagnostics["profile_intervals"] = profiles
    diagnostics["bootstrap_seed"] = bootstrap.seed
    diagnostics["bootstrap_pseudo_observation_provenance"] = (
        _bootstrap_provenance(bootstrap)
    )
    diagnostics["bootstrap_requested_replicates"] = bootstrap.requested_replicates
    diagnostics["bootstrap_successful_replicates"] = bootstrap.successful_replicates
    diagnostics[
        "bootstrap_quadratic_background_constrained_fallback_replicates"
    ] = bootstrap.quadratic_background_constrained_fallback_replicates
    diagnostics[
        "bootstrap_quadratic_background_exact_cone_invalid_replicates"
    ] = bootstrap.quadratic_background_exact_cone_invalid_replicates
    diagnostics[
        "bootstrap_quadratic_background_cone_active_replicates"
    ] = bootstrap.quadratic_background_cone_active_replicates
    diagnostics[
        "bootstrap_minimum_quadratic_background_normalized_cone_margin"
    ] = bootstrap.minimum_quadratic_background_normalized_cone_margin
    diagnostics["bootstrap_coverage_assessment_status"] = (
        "descriptive_only_too_few_replicates"
        if bootstrap.successful_replicates
        < int(reporting["minimum_bootstrap_replicates_for_coverage_assessment"])
        else "diagnostic_coverage_estimate"
    )
    diagnostics["canonical_component_variant"] = canonical_component_name
    diagnostics["canonical_model_variant"] = canonical_name
    diagnostics["phase2_window_change"] = config["phase2_window_change"]
    diagnostics["component_audit_policy"] = config["contaminant_policy"]
    diagnostics["authoritative_nuclear_source"] = config["component_audit"][
        "authoritative_nuclear_source"
    ]
    diagnostics["model_variant_comparison"] = model_rows
    diagnostics["model_variant_fit_diagnostics"] = all_variant_fit_diagnostics
    diagnostics["al27_ge70_discrimination"] = al_ge_discrimination_rows[0]
    diagnostics["ratio_model_variant_systematic"] = {
        "definition": "all-declared positive-semidefinite RMS outer-product covariance; includes successful variants rejected by fit-quality diagnostics",
        "included_variants": all_variant_names,
        "all_declared_definition": "positive-semidefinite RMS outer-product covariance of every successful declared component, line-shape, and background variant, including fits rejected by applicability diagnostics",
        "all_declared_included_variants": all_variant_names,
        "fit_quality_acceptable_definition": "same RMS construction restricted to noncanonical variants that pass every declared fit-quality applicability check",
        "fit_quality_acceptable_included_variants": acceptable_variant_names,
        "fit_quality_acceptable_status": (
            "available"
            if acceptable_variant_names
            else "unavailable; no noncanonical variant passes declared fit-quality applicability checks"
        ),
        "maximum_model_systematic_to_fisher_sd_ratio": float(
            np.max(
                np.divide(
                    ratio_model_sd,
                    ratio_sd,
                    out=np.zeros_like(ratio_model_sd),
                    where=ratio_sd > 0,
                )
            )
        ),
        "all_declared_rms_driver": all_variant_driver,
    }
    diagnostics["fit_result"] = result
    diagnostics["spec"] = spec
    return products, diagnostics


def _serializable_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in diagnostics.items()
        if key not in {"fit_result", "spec"}
    }


def _git_revision(repo_root: Path) -> dict[str, Any]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, text=True
        ).strip()
    )
    return {"commit": revision, "working_tree_dirty": dirty}


def _write_manifest(
    path: Path,
    *,
    repo_root: Path,
    config: dict[str, Any],
    config_path: Path,
    database_sha256: str,
    input_records: dict[str, list[dict[str, Any]]],
    diagnostics_by_table: dict[str, dict[str, Any]],
    output_hashes: dict[str, str],
    bootstrap_replicates: int,
    code_revision: dict[str, Any] | None = None,
) -> None:
    """Write the self-contained provenance manifest used by the CLI."""

    manifest = {
        "workflow": "paper peak-statistics measured-data correction phase 2",
        "result_semantics": config["result_semantics"],
        "input_release": "HFIRBG_public_data_v1.1.0",
        "database_sha256": database_sha256,
        "database_access": "SQLite URI mode=ro plus PRAGMA query_only",
        "spectrum_access": "read-only calibrated text counts",
        "input_spectra": input_records,
        "configuration": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
            "schema_version": config["schema_version"],
        },
        "reporting_configuration": config["reporting"],
        "code_revision": (
            _git_revision(repo_root)
            if code_revision is None
            else code_revision
        ),
        "likelihood": config["likelihood"],
        "fit_definitions": {
            f"table{table}": config[f"table{table}"]
            for table in diagnostics_by_table
        },
        "bootstrap_replicates_per_table": bootstrap_replicates,
        "diagnostics": diagnostics_by_table,
        "output_sha256": output_hashes,
        "scientific_scope": {
            "measured_detector_counts": True,
            "unfolding_performed": False,
            "cadmium_abundance_inferred": False,
            "neutron_simulation_run": False,
            "missing_simulation_products_combined": False,
            "candidate_numbers_approved_for_manuscript": False,
        },
    }
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        required=True,
        type=Path,
        help="immutable HFIRBG_public_data_v1.1.0 directory",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="empty output directory"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "config" / "paper_peak_statistics.json",
    )
    parser.add_argument("--table", choices=("3", "8", "all"), default="all")
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=None,
        help="deterministic diagnostic refits per table (default from config)",
    )
    args = parser.parse_args()

    bundle = args.bundle.expanduser().resolve()
    db_path = bundle / "HFIRBG.db"
    data_root = bundle / "spectra"
    if not db_path.is_file() or not data_root.is_dir():
        raise FileNotFoundError("bundle must contain HFIRBG.db and spectra/")
    db_hash = _sha256(db_path)
    if db_hash != PUBLIC_V1_1_DB_SHA256:
        raise RuntimeError("database hash does not match immutable public v1.1.0")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise RuntimeError("output directory must be empty")
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_catalog = build_run_catalog(db_path)
    run_records = {
        int(record["run_id"]): record
        for record in run_catalog.to_dict(orient="records")
    }
    reporting = config["reporting"]
    bootstrap_replicates = (
        int(reporting["default_bootstrap_replicates"])
        if args.bootstrap_replicates is None
        else args.bootstrap_replicates
    )
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap replicates must be positive")

    products: dict[str, list[dict[str, Any]]] = {}
    diagnostics_by_table: dict[str, dict[str, Any]] = {}
    input_records: dict[str, list[dict[str, Any]]] = {}
    fit_bin_jobs: list[
        tuple[str, tuple[PublicSpectrum, ...], JointPeakFitResult]
    ] = []

    if args.table in {"3", "all"}:
        configured_ids = tuple(config["table3"]["selection"]["file_ids_in_time_order"])
        spectra = tuple(load_spectrum(file_id, db_path, data_root) for file_id in configured_ids)
        if tuple(spectrum.file_id for spectrum in spectra) != configured_ids:
            raise RuntimeError("Table 3 file selection/order changed unexpectedly")
        expected_run = config["table3"]["selection"]["run_name"]
        if any(spectrum.run_name != expected_run for spectrum in spectra):
            raise RuntimeError("a configured Table 3 file no longer belongs to the declared run")
        table_products, diagnostics = _table3_products(
            spectra, config["table3"], reporting, bootstrap_replicates
        )
        products.update(table_products)
        diagnostics_by_table["3"] = _serializable_diagnostics(diagnostics)
        input_records["3"] = [
            _input_record(spectrum, data_root, run_records[spectrum.run_id])
            for spectrum in spectra
        ]
        fit_bin_jobs.append(("table3_fit_bins.csv.gz", spectra, diagnostics["fit_result"]))

    if args.table in {"8", "all"}:
        file_id = int(config["table8"]["selection"]["file_id"])
        spectrum = load_spectrum(file_id, db_path, data_root)
        table_products, diagnostics = _table8_products(
            spectrum, config["table8"], reporting, bootstrap_replicates
        )
        products.update(table_products)
        diagnostics_by_table["8"] = _serializable_diagnostics(diagnostics)
        input_records["8"] = [
            _input_record(spectrum, data_root, run_records[spectrum.run_id])
        ]
        fit_bin_jobs.append(
            ("table8_fit_bins.csv.gz", (spectrum,), diagnostics["fit_result"])
        )

    for filename, rows in products.items():
        _write_csv(output_dir / filename, rows)
    for filename, spectra, result in fit_bin_jobs:
        _write_fit_bins(output_dir / filename, spectra, result)
    for table, diagnostics in diagnostics_by_table.items():
        (output_dir / f"table{table}_fit_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if "3" in diagnostics_by_table:
        historical_path = repo_root / "config" / "table3_historical_reconstruction.json"
        historical = json.loads(historical_path.read_text(encoding="utf-8"))
        historical_files = historical["selection_and_order"]["files"]
        observed_files = input_records["3"]
        if [item["file_id"] for item in historical_files] != [
            item["file_id"] for item in observed_files
        ] or [item["spectrum_sha256"] for item in historical_files] != [
            item["spectrum_sha256"] for item in observed_files
        ]:
            raise RuntimeError("historical Table 3 reconstruction inputs changed")
        historical_selection = historical["selection_and_order"]
        shared_metadata = historical_selection["shared_metadata"]
        if (
            [float(item["live_time_s"]) for item in historical_files]
            != [float(item["live_time_s"]) for item in observed_files]
            or sum(float(item["live_time_s"]) for item in observed_files)
            != float(historical_selection["total_live_time_s"])
            or any(
                int(item["run_id"]) != int(historical_selection["run_id"])
                or float(item["calibration_A0_keV"])
                != float(historical_selection["calibration_A0_keV"])
                or float(item["calibration_A1_keV_per_channel"])
                != float(historical_selection["calibration_A1_keV_per_channel"])
                for item in observed_files
            )
            or any(
                item[key] != value
                for item in observed_files
                for key, value in shared_metadata.items()
                if key in item and key != "raw_channel_axis_identity"
            )
        ):
            raise RuntimeError("historical Table 3 metadata record changed")
        historical["runtime_verification"] = {
            "database_sha256_matches": True,
            "ordered_spectrum_ids_and_hashes_match": True,
            "live_times_calibration_and_shared_metadata_match": True,
            "record_source_path": str(historical_path),
            "record_source_sha256": _sha256(historical_path),
        }
        (output_dir / "table3_historical_reconstruction.json").write_text(
            json.dumps(historical, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    output_hashes = {
        path.name: _sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    _write_manifest(
        output_dir / "manifest.json",
        repo_root=repo_root,
        config=config,
        config_path=config_path,
        database_sha256=db_hash,
        input_records=input_records,
        diagnostics_by_table=diagnostics_by_table,
        output_hashes=output_hashes,
        bootstrap_replicates=bootstrap_replicates,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
