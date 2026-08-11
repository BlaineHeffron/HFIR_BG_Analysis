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
    profile_ratio_interval,
    ratios_from_fit,
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
    if not result.success:
        applicability_reasons.append("optimizer did not converge")
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
            "coverage_status": coverage_status,
            "minimum_successful_replicates_for_coverage_assessment": (
                minimum_replicates_for_coverage
            ),
            "semantics": (
                "parametric-bootstrap diagnostic; not canonical interval; "
                "coverage is not tested below the declared replicate minimum"
            ),
        }
        for index, name in enumerate(summary.line_names)
    ]


def _profile_if_needed(
    spectra: Sequence[PublicSpectrum],
    spec: JointPeakSpec,
    calibration: CalibrationConstraint,
    resolution: LinearResolution,
    result: JointPeakFitResult,
    definition: RatioDefinition,
    weak_threshold: float,
    confidence_level: float,
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
                spectra, variant_spec, calibration, resolution
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
                "akaike_information_criterion": format(
                    2.0 * result.penalized_nll + 2.0 * parameter_count, ".12g"
                ),
                "bayesian_information_criterion": format(
                    2.0 * result.penalized_nll
                    + parameter_count * np.log(observation_count),
                    ".12g",
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
    bootstrap = parametric_bootstrap(
        spectra,
        spec,
        calibration,
        resolution,
        result,
        case_identity="paper-table-3-measured-data-v1",
        replicates=bootstrap_replicates,
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
        "table3_bootstrap_diagnostics.csv": _bootstrap_rows(
            bootstrap,
            int(reporting["minimum_bootstrap_replicates_for_coverage_assessment"]),
        ),
        "table3_model_variant_comparison.csv": model_variants,
    }
    diagnostics = _fit_diagnostics(result, reporting)
    diagnostics["profile_intervals"] = profiles
    diagnostics["bootstrap_seed"] = bootstrap.seed
    diagnostics["bootstrap_requested_replicates"] = bootstrap.requested_replicates
    diagnostics["bootstrap_successful_replicates"] = bootstrap.successful_replicates
    diagnostics["bootstrap_coverage_assessment_status"] = (
        "descriptive_only_too_few_replicates"
        if bootstrap.successful_replicates
        < int(reporting["minimum_bootstrap_replicates_for_coverage_assessment"])
        else "diagnostic_coverage_estimate"
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
    diagnostics["fit_result"] = result
    diagnostics["spec"] = spec
    return products, diagnostics


def _table8_ratio_definitions(spec: JointPeakSpec) -> tuple[RatioDefinition, ...]:
    by_parent: dict[str, dict[str, str]] = {}
    for component in spec.components:
        by_parent.setdefault(component.parent, {})[component.role] = component.name
    definitions: list[RatioDefinition] = []
    for parent in sorted(by_parent, key=float):
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


def _table8_products(
    spectrum: PublicSpectrum,
    config: dict[str, Any],
    reporting: dict[str, Any],
    bootstrap_replicates: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    spectra = (spectrum,)
    spec = _spec(config, "paper-table-8-measured-data-v1")
    calibration = _constraint(config["calibration_constraint"])
    resolution = _resolution(config["resolution_initial"])
    result = fit_joint_peak_model(spectra, spec, calibration, resolution)
    if not result.success:
        raise RuntimeError(f"Table 8 fit failed: {result.message}")
    definitions = _table8_ratio_definitions(spec)
    ratios = ratios_from_fit(result, definitions)
    ratio_sd = np.sqrt(np.maximum(np.diag(ratios.covariance), 0.0))
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
        )
        if profile is not None:
            profiles.append(asdict(profile))
            interval_kind = f"profile_{profile.kind}"
            lower, upper = profile.lower, profile.upper
        else:
            interval_kind = "fisher_symmetric_regular"
            lower = max(0.0, ratios.values[index] - 1.959963984540054 * ratio_sd[index])
            upper = ratios.values[index] + 1.959963984540054 * ratio_sd[index]
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
    diagnostics["bootstrap_requested_replicates"] = bootstrap.requested_replicates
    diagnostics["bootstrap_successful_replicates"] = bootstrap.successful_replicates
    diagnostics["bootstrap_coverage_assessment_status"] = (
        "descriptive_only_too_few_replicates"
        if bootstrap.successful_replicates
        < int(reporting["minimum_bootstrap_replicates_for_coverage_assessment"])
        else "diagnostic_coverage_estimate"
    )
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

    output_hashes = {
        path.name: _sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    manifest = {
        "workflow": "paper peak-statistics measured-data correction phase 1",
        "result_semantics": config["result_semantics"],
        "input_release": "HFIRBG_public_data_v1.1.0",
        "database_sha256": db_hash,
        "database_access": "SQLite URI mode=ro plus PRAGMA query_only",
        "spectrum_access": "read-only calibrated text counts",
        "input_spectra": input_records,
        "configuration": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
            "schema_version": config["schema_version"],
        },
        "code_revision": _git_revision(repo_root),
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
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output_dir)


if __name__ == "__main__":
    main()
