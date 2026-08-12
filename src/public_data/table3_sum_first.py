"""Corrected sum-first/local-window products for descriptive paper Table 3.

The workflow reuses the normalized Poisson peak likelihood.  It never unfolds
the measured counts and never interprets fitted detector prominence as source
emission, activity, abundance, flux, or detector-response validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha256
from math import isfinite, sqrt
from typing import Any, Mapping, Sequence

import numpy as np

from src.public_data.browser import (
    PublicSpectrum,
    accumulate_spectra_exact,
)
from src.public_data.peak_likelihood import (
    CalibrationConstraint,
    FitWindow,
    JointPeakFitResult,
    JointPeakSpec,
    LineComponent,
    LinearResolution,
    RatioDefinition,
    fit_joint_peak_model,
    ratios_from_fit,
    resolution_sigma_and_derivatives,
)
from src.public_data.peak_residuals import (
    poisson_deviance_contributions,
    window_residual_diagnostics,
)


REFERENCE = "rd_558_5_reference"
STATUS_QUANTITATIVE = "supported quantitative relative detected count"
STATUS_UNRELIABLE = "visible/identified but quantitative area unreliable"
STATUS_BROAD = "broad physical feature requiring a non-Gaussian template"
STATUS_BLEND = "unresolved blend"
STATUS_UNAVAILABLE = "unsupported/unavailable"


@dataclass(frozen=True)
class Table3SumFirstAnalysis:
    """Generated rows plus the canonical fit needed for residual-bin export."""

    products: Mapping[str, list[dict[str, Any]]]
    diagnostics: Mapping[str, Any]
    accumulated: PublicSpectrum
    canonical_result: JointPeakFitResult
    canonical_spec: JointPeakSpec
    calibration: CalibrationConstraint
    resolution: LinearResolution


def _array_sha256(values: np.ndarray, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
    return sha256(array.tobytes()).hexdigest()


def _paper_energy(item: Mapping[str, Any]) -> float:
    return float(item.get("paper_energy_keV", item["energy_keV"]))


def _window_name(energies: Sequence[float]) -> str:
    labels = "_".join(f"{value:g}".replace(".", "p") for value in energies)
    return f"sum_first_{labels}"


def _historical_windows(
    accumulated: PublicSpectrum,
    historical: Mapping[str, Any],
    background_model: str,
) -> tuple[dict[float, FitWindow], dict[str, tuple[int, int]]]:
    by_energy: dict[float, FitWindow] = {}
    channel_ranges: dict[str, tuple[int, int]] = {}
    for record in historical["historical_local_windows"]:
        energies = tuple(float(value) for value in record["paper_energies_keV"])
        first = int(record["first_channel_1_based"])
        last = int(record["last_channel_1_based"])
        if first < 1 or last < first or last > accumulated.counts.size:
            raise RuntimeError("historical Table 3 channel window is invalid")
        name = _window_name(energies)
        low = accumulated.calibration_A0 + accumulated.calibration_A1 * (first - 0.5)
        high = accumulated.calibration_A0 + accumulated.calibration_A1 * (last + 0.5)
        window = FitWindow(name, low, high, background_model)
        selected = np.flatnonzero(
            (accumulated.energy_keV >= low)
            & (accumulated.energy_keV < high)
        )
        expected = np.arange(first - 1, last, dtype=np.int64)
        if not np.array_equal(selected, expected):
            raise RuntimeError("historical Table 3 native-bin window changed")
        channel_ranges[name] = (first, last)
        for energy in energies:
            key = round(energy, 6)
            if key in by_energy:
                raise RuntimeError("historical Table 3 row appears in two windows")
            by_energy[key] = window
    return by_energy, channel_ranges


def _component_metadata(
    table_config: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[float, dict[str, Any]]]:
    by_name: dict[str, dict[str, Any]] = {}
    by_paper_energy: dict[float, dict[str, Any]] = {}
    sources = (
        list(table_config["components"])
        + list(table_config["sum_first_local_workflow"]["isolated_candidate_components"])
        + list(table_config["excluded_components"])
    )
    for source in sources:
        item = dict(source)
        if item.get("role", "line") == "contaminant":
            continue
        paper = round(_paper_energy(item), 6)
        if "name" in item and str(item["name"]).startswith("rd_"):
            by_name[str(item["name"])] = item
        by_paper_energy.setdefault(paper, item)
    return by_name, by_paper_energy


def _primary_spec(
    accumulated: PublicSpectrum,
    table_config: Mapping[str, Any],
    historical: Mapping[str, Any],
    *,
    background_model: str,
    name: str,
) -> tuple[JointPeakSpec, dict[str, dict[str, Any]], dict[str, tuple[int, int]]]:
    windows_by_energy, channel_ranges = _historical_windows(
        accumulated, historical, background_model
    )
    records = [
        dict(item)
        for item in table_config["components"]
        if item.get("role", "line") == "line"
    ]
    records.extend(
        dict(item)
        for item in table_config["sum_first_local_workflow"][
            "isolated_candidate_components"
        ]
    )
    records.sort(key=_paper_energy)
    components: list[LineComponent] = []
    metadata: dict[str, dict[str, Any]] = {}
    selected_windows: dict[str, FitWindow] = {}
    for item in records:
        paper = float(item.get("window_paper_energy_keV", _paper_energy(item)))
        try:
            window = windows_by_energy[round(paper, 6)]
        except KeyError as error:
            raise RuntimeError(f"no historical window for Table 3 row {paper:g}") from error
        selected_windows[window.name] = window
        component = LineComponent(
            str(item["name"]),
            float(item["energy_keV"]),
            window.name,
            str(item.get("role", "line")),
            "",
            str(item.get("origin_class", "shared")),
        )
        components.append(component)
        metadata[component.name] = item
    windows = tuple(sorted(selected_windows.values(), key=lambda item: item.low_keV))
    return JointPeakSpec(name, windows, tuple(components)), metadata, channel_ranges


def _assert_accumulation(
    spectra: Sequence[PublicSpectrum],
    accumulated: PublicSpectrum,
    table_config: Mapping[str, Any],
    historical: Mapping[str, Any],
) -> dict[str, Any]:
    selection = historical["selection_and_order"]
    expected_ids = tuple(
        int(value) for value in table_config["selection"]["file_ids_in_time_order"]
    )
    historical_ids = tuple(int(item["file_id"]) for item in selection["files"])
    observed_ids = tuple(spectrum.file_id for spectrum in spectra)
    if observed_ids != expected_ids or observed_ids != historical_ids:
        raise RuntimeError("sum-first Table 3 input identity/order changed")
    if (
        table_config["selection"]["release"]
        != historical["public_release"]["name"]
        or any(spectrum.run_id != int(selection["run_id"]) for spectrum in spectra)
    ):
        raise RuntimeError("sum-first Table 3 release/run identity changed")
    expected_file_names = tuple(str(item["file_name"]) for item in selection["files"])
    expected_live_times = tuple(float(item["live_time_s"]) for item in selection["files"])
    if (
        tuple(spectrum.file_name for spectrum in spectra) != expected_file_names
        or tuple(spectrum.live_time for spectrum in spectra) != expected_live_times
    ):
        raise RuntimeError("sum-first Table 3 file name/live-time record changed")
    group_ids = tuple(
        int(spectrum.metadata["calibration_group_id"]) for spectrum in spectra
    )
    if group_ids != (90, 90, 90, 90):
        raise RuntimeError("sum-first Table 3 calibration group is not exactly 90")
    if any(
        spectrum.calibration_A0 != float(selection["calibration_A0_keV"])
        or spectrum.calibration_A1
        != float(selection["calibration_A1_keV_per_channel"])
        or spectrum.counts.size != int(selection["channel_count"])
        for spectrum in spectra
    ):
        raise RuntimeError("sum-first Table 3 calibration/channel identity changed")
    shared_key_map = {
        "calibration_group_name": "calibration_group_name",
        "coordinate_id": "coordinate_id",
        "coordinate_Rx": "coordinate_Rx",
        "coordinate_Rz": "coordinate_Rz",
        "coordinate_Lx": "coordinate_Lx",
        "coordinate_Lz": "coordinate_Lz",
        "coordinate_track": "coordinate_track",
        "detector_orientation_angle_deg": "coordinate_angle",
        "detector_configuration_id": "detector_configuration_id",
        "detector_id": "detector_id",
        "detector_type": "detector_type",
        "shield_id": "shield_id",
        "shield_name": "shield_name",
        "shield_description": "shield_description",
    }
    expected_shared = selection["shared_metadata"]
    for output_key, metadata_key in shared_key_map.items():
        expected = expected_shared[output_key]
        if any(spectrum.metadata.get(metadata_key) != expected for spectrum in spectra):
            raise RuntimeError(
                f"sum-first Table 3 shared metadata changed: {output_key}"
            )
    count_hash = _array_sha256(accumulated.counts, "<i8")
    energy_hash = _array_sha256(accumulated.energy_keV, "<f8")
    if (
        count_hash != selection["summed_counts_sha256_int64_little_endian"]
        or energy_hash != selection["energy_grid_sha256_float64_little_endian"]
        or int(accumulated.counts.sum()) != int(selection["summed_detector_counts"])
        or int(accumulated.counts.max())
        != int(selection["maximum_summed_channel_count"])
        or accumulated.live_time != float(selection["total_live_time_s"])
    ):
        raise RuntimeError("sum-first Table 3 exact accumulation record changed")
    return {
        "release": table_config["selection"]["release"],
        "ordered_file_ids": list(observed_ids),
        "ordered_file_names": list(expected_file_names),
        "ordered_live_times_s": list(expected_live_times),
        "run_id": int(selection["run_id"]),
        "run_name": str(selection["run_name"]),
        "reactor_cycle_classification": (
            f"Cycle {historical['reconstructed_input_cycle']}; reactor-on selection"
        ),
        "calibration_group_ids": list(group_ids),
        "calibration_A0_keV": accumulated.calibration_A0,
        "calibration_A1_keV_per_channel": accumulated.calibration_A1,
        "channel_count": accumulated.counts.size,
        "total_live_time_s": accumulated.live_time,
        "summed_detector_counts": int(accumulated.counts.sum()),
        "maximum_summed_channel_count": int(accumulated.counts.max()),
        "summed_counts_sha256_int64_little_endian": count_hash,
        "energy_grid_sha256_float64_little_endian": energy_hash,
        "detector_and_geometry": {
            key: expected_shared[key] for key in shared_key_map
        },
        "normalization": "none; measured detector counts per raw channel",
        "response_model_identity": expected_shared["response_model_identity"],
        "summation": "independent Poisson counts added exactly in int64; no interpolation or rebinning",
    }


def _fit_variants(
    accumulated: PublicSpectrum,
    table_config: Mapping[str, Any],
    historical: Mapping[str, Any],
    calibration: CalibrationConstraint,
    resolution: LinearResolution,
) -> tuple[
    dict[str, JointPeakFitResult],
    dict[str, JointPeakSpec],
    dict[str, LinearResolution],
    dict[str, dict[str, Any]],
    dict[str, tuple[int, int]],
]:
    results: dict[str, JointPeakFitResult] = {}
    specs: dict[str, JointPeakSpec] = {}
    resolutions: dict[str, LinearResolution] = {}
    metadata: dict[str, dict[str, Any]] = {}
    channel_ranges: dict[str, tuple[int, int]] = {}
    warm: JointPeakFitResult | None = None
    for variant in table_config["sum_first_local_workflow"]["model_variants"]:
        variant_name = str(variant["name"])
        spec, current_metadata, current_ranges = _primary_spec(
            accumulated,
            table_config,
            historical,
            background_model=str(variant["background_model"]),
            name=f"paper-table-3-sum-first:{variant_name}",
        )
        fit_resolution = replace(
            resolution,
            tail_model=str(variant["tail_model"]),
            tail_fraction=float(
                variant.get("tail_fraction_initial", resolution.tail_fraction)
            ),
            per_run_scale_sigma=0.0,
        )
        result = fit_joint_peak_model(
            (accumulated,),
            spec,
            calibration,
            fit_resolution,
            warm_start=warm,
            warm_start_source=(
                "preceding declared sum-first model variant" if warm else ""
            ),
        )
        if not result.success:
            raise RuntimeError(
                f"Table 3 sum-first variant {variant_name} failed: {result.message}"
            )
        results[variant_name] = result
        specs[variant_name] = spec
        resolutions[variant_name] = fit_resolution
        metadata = current_metadata
        channel_ranges = current_ranges
        warm = result
    return results, specs, resolutions, metadata, channel_ranges


def _window_deviance(result: JointPeakFitResult, window: str) -> float:
    mask = np.asarray(result.observation_window_names) == window
    return float(
        poisson_deviance_contributions(
            result.observed_counts[mask], result.expected_counts[mask]
        ).sum()
    )


def _model_comparison_rows(
    results: Mapping[str, JointPeakFitResult],
    specs: Mapping[str, JointPeakSpec],
    table_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    variants = {
        str(item["name"]): item
        for item in table_config["sum_first_local_workflow"]["model_variants"]
    }
    canonical_name = "no_tail_quadratic"
    canonical = results[canonical_name]
    rows: list[dict[str, Any]] = []
    for name, result in results.items():
        tail_fraction = (
            result.parameter("shape.low_energy_tail_fraction")
            if "shape.low_energy_tail_fraction" in result.parameter_names
            else float("nan")
        )
        tail_scale = (
            result.parameter("shape.low_energy_tail_scale_in_sigma")
            if "shape.low_energy_tail_scale_in_sigma" in result.parameter_names
            else float("nan")
        )
        common = {
            "variant": name,
            "background_model": variants[name]["background_model"],
            "tail_model": variants[name]["tail_model"],
            "selected_canonical": name == canonical_name,
            "success": result.success,
            "fisher_covariance_valid": result.fisher_covariance_valid,
            "active_bounds": ";".join(result.active_bounds),
            "tail_fraction": tail_fraction,
            "tail_scale_in_sigma": tail_scale,
            "tail_interior": (
                isfinite(tail_fraction)
                and not any(bound.startswith("shape.") for bound in result.active_bounds)
            ),
            "parameter_count_global": len(result.parameter_names),
            "native_bin_count_global": result.observed_counts.size,
        }
        rows.append(
            {
                **common,
                "scope": "all_primary_windows",
                "window": "all",
                "poisson_deviance": result.poisson_deviance,
                "delta_deviance_from_canonical_same_scope": (
                    result.poisson_deviance - canonical.poisson_deviance
                ),
                "comparison_semantics": (
                    "same accumulated spectrum and primary native bins; descriptive "
                    "model comparison, not a row-wise p-value selector"
                ),
            }
        )
        canonical_windows = {window.name for window in specs[canonical_name].windows}
        if {window.name for window in specs[name].windows} != canonical_windows:
            raise RuntimeError("sum-first model variants changed their native windows")
        for window in specs[name].windows:
            deviance = _window_deviance(result, window.name)
            rows.append(
                {
                    **common,
                    "scope": "declared_local_window",
                    "window": window.name,
                    "poisson_deviance": deviance,
                    "delta_deviance_from_canonical_same_scope": (
                        deviance - _window_deviance(canonical, window.name)
                    ),
                    "comparison_semantics": (
                        "identical native bins; shared global nuisance allocation "
                        "prevents interpreting D/window as an independent fit rank"
                    ),
                }
            )
    return rows


def _ratio_result(result: JointPeakFitResult):
    definitions = tuple(
        RatioDefinition(name, name, REFERENCE) for name in result.line_names
    )
    return ratios_from_fit(result, definitions)


def _constant_local_resolution(
    resolution: LinearResolution,
    fitted: JointPeakFitResult,
    energies_keV: Sequence[float],
) -> LinearResolution:
    energy = float(np.mean(np.asarray(energies_keV, dtype=np.float64)))
    sigma = resolution_sigma_and_derivatives(
        resolution,
        energy,
        fitted.parameter("resolution.intercept_keV"),
        fitted.parameter("resolution.variance_slope_keV"),
    )[0]
    return replace(
        resolution,
        intercept_keV=sigma,
        slope=0.0,
        form="constant",
        tail_model="none",
        tail_fraction=0.0,
        per_run_scale_sigma=0.0,
    )


def _per_window_sensitivity(
    accumulated: PublicSpectrum,
    canonical_result: JointPeakFitResult,
    canonical_spec: JointPeakSpec,
    calibration: CalibrationConstraint,
    resolution: LinearResolution,
    movement_sigma_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    local_results: dict[str, JointPeakFitResult] = {}
    for window in canonical_spec.windows:
        components = tuple(
            component
            for component in canonical_spec.components
            if component.window == window.name
        )
        local_spec = JointPeakSpec(
            f"{canonical_spec.name}:independent:{window.name}",
            (window,),
            components,
        )
        local_resolution = _constant_local_resolution(
            resolution,
            canonical_result,
            [component.energy_keV for component in components],
        )
        local_results[window.name] = fit_joint_peak_model(
            (accumulated,),
            local_spec,
            calibration,
            local_resolution,
            warm_start=canonical_result,
            warm_start_source="assembled sum-first canonical fit",
        )
    reference_component = next(
        component for component in canonical_spec.components if component.name == REFERENCE
    )
    reference_result = local_results[reference_component.window]
    reference_index = reference_result.line_names.index(REFERENCE)
    reference_rate = float(reference_result.line_rates_counts_per_s[reference_index])
    reference_variance = float(
        reference_result.line_rate_covariance[reference_index, reference_index]
    )
    canonical_ratios = _ratio_result(canonical_result)
    canonical_lookup = {
        name: (float(canonical_ratios.values[index]), sqrt(max(float(canonical_ratios.covariance[index, index]), 0.0)))
        for index, name in enumerate(canonical_ratios.labels)
    }
    rows: list[dict[str, Any]] = []
    lookup: dict[str, dict[str, Any]] = {}
    for component in canonical_spec.components:
        result = local_results[component.window]
        index = result.line_names.index(component.name)
        rate = float(result.line_rates_counts_per_s[index])
        variance = max(float(result.line_rate_covariance[index, index]), 0.0)
        if component.name == REFERENCE:
            ratio = 1.0
            ratio_variance = 0.0
            covariance_statement = "exact self-ratio; zero Jacobian and variance"
        else:
            ratio = rate / reference_rate if reference_rate > 0 else float("nan")
            ratio_variance = (
                variance / reference_rate**2
                + rate**2 * reference_variance / reference_rate**4
                if reference_rate > 0
                else float("nan")
            )
            covariance_statement = (
                "numerator and reference use disjoint independently fitted native "
                "windows; conditional covariance set exactly to zero"
            )
        intercept = result.parameter("resolution.intercept_keV")
        sigma = intercept
        canonical_ratio, canonical_sigma = canonical_lookup[component.name]
        movement = abs(ratio - canonical_ratio)
        model_sensitive = bool(
            isfinite(movement)
            and isfinite(canonical_sigma)
            and movement > movement_sigma_threshold * canonical_sigma
        )
        row = {
            "component": component.name,
            "window": component.window,
            "fit_success": result.success,
            "fisher_covariance_valid": result.fisher_covariance_valid,
            "fisher_note": (
                "one free constant effective width; full local conditional Fisher covariance"
            ),
            "active_bounds": ";".join(result.active_bounds),
            "effective_sigma_keV": sigma,
            "full_line_rate_counts_per_s": rate,
            "full_line_rate_conditional_uncertainty_counts_per_s": sqrt(variance),
            "ratio_to_558_456_keV": ratio,
            "ratio_conditional_uncertainty": sqrt(max(ratio_variance, 0.0)),
            "canonical_ratio": canonical_ratio,
            "canonical_ratio_conditional_uncertainty": canonical_sigma,
            "absolute_ratio_movement": movement,
            "movement_exceeds_canonical_conditional_1sigma": model_sensitive,
            "numerator_denominator_covariance_statement": covariance_statement,
            "role": "declared per-window free-width sensitivity; never selectable answer",
        }
        rows.append(row)
        lookup[component.name] = row
    return rows, lookup


def _covariance_rows(
    names: Sequence[str], covariance: np.ndarray, unit: str
) -> list[dict[str, Any]]:
    return [
        {
            "row": row_name,
            "column": column_name,
            "covariance": float(covariance[row, column]),
            "unit": unit,
        }
        for row, row_name in enumerate(names)
        for column, column_name in enumerate(names)
    ]


def _parameter_covariance_rows(
    result: JointPeakFitResult,
) -> list[dict[str, Any]]:
    return [
        {
            "row": row_name,
            "column": column_name,
            "covariance": float(result.covariance[row, column]),
            "semantics": "complete conditional fitted-parameter covariance",
        }
        for row, row_name in enumerate(result.parameter_names)
        for column, column_name in enumerate(result.parameter_names)
    ]


def _correlation(
    result: JointPeakFitResult, left: str, right: str
) -> float:
    i = result.line_names.index(left)
    j = result.line_names.index(right)
    denominator = sqrt(
        max(float(result.line_rate_covariance[i, i]), 0.0)
        * max(float(result.line_rate_covariance[j, j]), 0.0)
    )
    return (
        float(result.line_rate_covariance[i, j]) / denominator
        if denominator > 0
        else float("nan")
    )


def _cluster_bounds(
    record: Mapping[str, Any],
    local_windows: Mapping[float, FitWindow],
) -> tuple[float, float]:
    if "window_paper_energy_keV" in record:
        window = local_windows[round(float(record["window_paper_energy_keV"]), 6)]
        return window.low_keV, window.high_keV
    return float(record["low_keV"]), float(record["high_keV"])


def _cluster_diagnostics(
    spectra: Sequence[PublicSpectrum],
    accumulated: PublicSpectrum,
    table_config: Mapping[str, Any],
    historical: Mapping[str, Any],
    canonical_result: JointPeakFitResult,
    calibration: CalibrationConstraint,
    resolution: LinearResolution,
) -> list[dict[str, Any]]:
    by_name, _ = _component_metadata(table_config)
    local_windows, _ = _historical_windows(
        accumulated, historical, "quadratic"
    )
    rows: list[dict[str, Any]] = []
    for record in table_config["sum_first_local_workflow"][
        "diagnostic_clusters"
    ]:
        cluster = str(record["name"])
        low, high = _cluster_bounds(record, local_windows)
        window = FitWindow(f"diagnostic_{cluster}", low, high, "quadratic")
        targets: list[LineComponent] = []
        for name in record["reported_components"]:
            item = by_name[str(name)]
            targets.append(
                LineComponent(
                    str(name),
                    float(item["energy_keV"]),
                    window.name,
                    "line",
                    "",
                    str(item.get("origin_class", "shared")),
                )
            )
        candidates = tuple(
            LineComponent(
                str(item["name"]),
                float(item["energy_keV"]),
                window.name,
                "contaminant",
                "",
                "shared",
            )
            for item in record.get("candidate_components", [])
        )
        base_spec = JointPeakSpec(
            f"table3-sum-first-diagnostic:{cluster}:targets",
            (window,),
            tuple(targets),
        )
        full_spec = replace(
            base_spec,
            name=f"table3-sum-first-diagnostic:{cluster}:declared-candidates",
            components=tuple(targets) + candidates,
        )
        local_resolution = _constant_local_resolution(
            resolution,
            canonical_result,
            [component.energy_keV for component in full_spec.components],
        )
        base = fit_joint_peak_model(
            (accumulated,),
            base_spec,
            calibration,
            local_resolution,
            warm_start=canonical_result,
            warm_start_source="assembled sum-first canonical fit",
        )
        full = (
            fit_joint_peak_model(
                (accumulated,),
                full_spec,
                calibration,
                local_resolution,
                warm_start=base,
                warm_start_source=f"{cluster} target-only diagnostic",
            )
            if candidates
            else base
        )
        candidate_details: list[dict[str, Any]] = []
        maximum_target_candidate_correlation = 0.0
        candidate_interior = False
        recurrence_records: list[dict[str, Any]] = []
        for candidate, source in zip(
            candidates, record.get("candidate_components", [])
        ):
            index = full.line_names.index(candidate.name)
            rate = float(full.line_rates_counts_per_s[index])
            sigma = sqrt(
                max(float(full.line_rate_covariance[index, index]), 0.0)
            )
            bound = any(
                name == f"line.{candidate.name}.rate_counts_per_s"
                for name in full.active_bounds
            )
            correlations = [
                abs(_correlation(full, target.name, candidate.name))
                for target in targets
            ]
            finite_correlations = [value for value in correlations if isfinite(value)]
            maximum_target_candidate_correlation = max(
                maximum_target_candidate_correlation,
                max(finite_correlations, default=0.0),
            )
            candidate_interior = candidate_interior or (
                not bound and sigma > 0 and rate / sigma >= 2.0
            )
            candidate_details.append(
                {
                    "name": candidate.name,
                    "energy_keV": candidate.energy_keV,
                    "identity": source["identity"],
                    "source": source["authoritative_source"],
                    "intensity": source.get(
                        "intensity_percent",
                        source.get("capgam_relative_intensity", "not supplied"),
                    ),
                    "treatment": source["treatment"],
                    "rate_counts_per_s": rate,
                    "conditional_sigma_counts_per_s": sigma,
                    "conditional_z": rate / sigma if sigma > 0 else float("nan"),
                    "active_bound": bound,
                }
            )
            if "unidentified" in str(source["treatment"]):
                for spectrum in spectra[:3]:
                    recurrence = fit_joint_peak_model(
                        (spectrum,),
                        full_spec,
                        calibration,
                        local_resolution,
                        warm_start=full,
                        warm_start_source=f"accumulated {cluster} diagnostic",
                    )
                    recurrence_index = recurrence.line_names.index(candidate.name)
                    recurrence_rate = float(
                        recurrence.line_rates_counts_per_s[recurrence_index]
                    )
                    recurrence_sigma = sqrt(
                        max(
                            float(
                                recurrence.line_rate_covariance[
                                    recurrence_index, recurrence_index
                                ]
                            ),
                            0.0,
                        )
                    )
                    recurrence_bound = any(
                        name == f"line.{candidate.name}.rate_counts_per_s"
                        for name in recurrence.active_bounds
                    )
                    recurrence_records.append(
                        {
                            "file_id": spectrum.file_id,
                            "fit_success": recurrence.success,
                            "rate_counts_per_s": recurrence_rate,
                            "conditional_sigma_counts_per_s": recurrence_sigma,
                            "conditional_z": (
                                recurrence_rate / recurrence_sigma
                                if recurrence_sigma > 0
                                else float("nan")
                            ),
                            "active_bound": recurrence_bound,
                        }
                    )
        target_impacts: list[dict[str, Any]] = []
        for target in targets:
            base_index = base.line_names.index(target.name)
            full_index = full.line_names.index(target.name)
            base_rate = float(base.line_rates_counts_per_s[base_index])
            full_rate = float(full.line_rates_counts_per_s[full_index])
            full_sigma = sqrt(
                max(float(full.line_rate_covariance[full_index, full_index]), 0.0)
            )
            target_impacts.append(
                {
                    "component": target.name,
                    "target_only_rate_counts_per_s": base_rate,
                    "declared_candidate_rate_counts_per_s": full_rate,
                    "absolute_shift_counts_per_s": abs(full_rate - base_rate),
                    "shift_in_full_conditional_sigma": (
                        abs(full_rate - base_rate) / full_sigma
                        if full_sigma > 0
                        else float("nan")
                    ),
                }
            )
        unmodeled = record.get("unmodeled_feature")
        if unmodeled is not None:
            outcome = "unidentified"
            outcome_reason = (
                "known physical structure lacks a released normalized template; "
                "target-only Gaussian fit cannot explain it"
            )
        elif candidate_interior:
            outcome = "absorbed_as_flexible_background"
            outcome_reason = (
                "one or more freely fitted catalog/unidentified amplitudes is "
                "interior; this is flexibility, not a branching prediction"
            )
        else:
            outcome = "unidentified"
            outcome_reason = "no constrained prediction and no supported free candidate"
        consequence = {
            "338_352": "351.9 local row evaluated independently; free Ac-228 component is not an explanation",
            "595_609": "609.3 visible/identified but quantitative area remains response-template-sensitive",
            "707_725": "707.4 evaluated in its isolated window; 725 unresolved",
            "1364_1408": "1364.3 and 1377.7 evaluated in isolated windows; 1399.6 unresolved",
            "5433": "5433.1 unsupported/unavailable",
            "5825": "5824.6 unsupported/unavailable",
            "7368": "7367.9 target unsupported; neighboring structure remains unidentified",
            "7916": "7916.3 unsupported/unavailable",
        }[cluster]
        rows.append(
            {
                "cluster": cluster,
                "low_keV": low,
                "high_keV": high,
                "reported_components": ";".join(item.name for item in targets),
                "reported_line_records": json.dumps(
                    [
                        {
                            "name": target.name,
                            "energy_keV": target.energy_keV,
                            "identity": by_name[target.name].get("identity", ""),
                            "source": by_name[target.name].get(
                                "authoritative_source", "declared primary component"
                            ),
                            "intensity": by_name[target.name].get(
                                "intensity_percent",
                                by_name[target.name].get(
                                    "capgam_relative_intensity",
                                    by_name[target.name].get(
                                        "authoritative_relative_intensity",
                                        "not supplied",
                                    ),
                                ),
                            ),
                        }
                        for target in targets
                    ],
                    sort_keys=True,
                ),
                "candidate_lines": json.dumps(candidate_details, sort_keys=True),
                "unmodeled_feature": (
                    "" if unmodeled is None else json.dumps(unmodeled, sort_keys=True)
                ),
                "candidate_amplitude_status": (
                    "none"
                    if not candidates
                    else "independent nonnegative free nuisance; never constrained"
                ),
                "predictive_test": "not testable without response/geometry model",
                "target_only_success": base.success,
                "declared_candidate_success": full.success,
                "target_only_poisson_deviance": base.poisson_deviance,
                "declared_candidate_poisson_deviance": full.poisson_deviance,
                "delta_deviance_full_minus_target_only": (
                    full.poisson_deviance - base.poisson_deviance
                ),
                "maximum_absolute_target_candidate_correlation": (
                    maximum_target_candidate_correlation
                ),
                "target_yield_impacts": json.dumps(target_impacts, sort_keys=True),
                "unidentified_recurrence_three_long_files": json.dumps(
                    recurrence_records, sort_keys=True
                ),
                "outcome": outcome,
                "outcome_reason": outcome_reason,
                "explained_by_constrained_held_line_prediction": False,
                "row_consequence": consequence,
            }
        )
    return rows


def _nominal_centroid(
    line_energy_keV: float,
    spectrum: PublicSpectrum,
    offset_keV: float,
    stretch: float,
) -> float:
    return (
        line_energy_keV - offset_keV + stretch * spectrum.calibration_A0
    ) / (1.0 + stretch)


def _drift_diagnostics(
    spectra: Sequence[PublicSpectrum],
    accumulated: PublicSpectrum,
    canonical_result: JointPeakFitResult,
    canonical_spec: JointPeakSpec,
    calibration: CalibrationConstraint,
    resolution: LinearResolution,
    workflow: Mapping[str, Any],
    per_window_lookup: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    component_lookup = {component.name: component for component in canonical_spec.components}
    window_lookup = {window.name: window for window in canonical_spec.windows}
    file_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for name in workflow["drift_diagnostic"]["anchor_components"]:
        component = component_lookup[str(name)]
        window = window_lookup[component.window]
        local_spec = JointPeakSpec(
            f"table3-drift:{component.name}",
            (window,),
            (component,),
        )
        local_resolution = _constant_local_resolution(
            resolution, canonical_result, [component.energy_keV]
        )
        records: list[dict[str, Any]] = []
        for spectrum in spectra:
            result = fit_joint_peak_model(
                (spectrum,),
                local_spec,
                calibration,
                local_resolution,
                warm_start=canonical_result,
                warm_start_source="accumulated sum-first canonical fit",
            )
            index = result.line_names.index(component.name)
            rate = float(result.line_rates_counts_per_s[index])
            rate_sigma = sqrt(
                max(float(result.line_rate_covariance[index, index]), 0.0)
            )
            bound = any(
                item == f"line.{component.name}.rate_counts_per_s"
                for item in result.active_bounds
            )
            offset = result.parameter("calibration.offset_keV")
            stretch = result.parameter("calibration.fractional_gain_stretch")
            sigma = result.parameter("resolution.intercept_keV")
            offset_index = result.parameter_names.index("calibration.offset_keV")
            stretch_index = result.parameter_names.index(
                "calibration.fractional_gain_stretch"
            )
            sigma_index = result.parameter_names.index("resolution.intercept_keV")
            centroid = _nominal_centroid(
                component.energy_keV, spectrum, offset, stretch
            )
            centroid_gradient = np.asarray(
                (
                    -1.0 / (1.0 + stretch),
                    (
                        spectrum.calibration_A0
                        - component.energy_keV
                        + offset
                    )
                    / (1.0 + stretch) ** 2,
                )
            )
            centroid_covariance = result.covariance[
                np.ix_((offset_index, stretch_index), (offset_index, stretch_index))
            ]
            centroid_uncertainty = sqrt(
                max(
                    float(
                        centroid_gradient @ centroid_covariance @ centroid_gradient
                    ),
                    0.0,
                )
            )
            sigma_uncertainty = sqrt(
                max(float(result.covariance[sigma_index, sigma_index]), 0.0)
            )
            available = bool(
                result.success
                and result.fisher_covariance_valid
                and not result.active_bounds
                and rate_sigma > 0
                and rate / rate_sigma >= 2
            )
            record = {
                "component": component.name,
                "authoritative_energy_keV": component.energy_keV,
                "file_id": spectrum.file_id,
                "live_time_s": spectrum.live_time,
                "fit_success": result.success,
                "fisher_covariance_valid": result.fisher_covariance_valid,
                "available_for_drift": available,
                "unavailability_reason": (
                    ""
                    if available
                    else "fit failed, Fisher covariance invalid, boundary-active, or line below 2 conditional sigma"
                ),
                "fitted_centroid_on_released_grid_keV": centroid,
                "fitted_centroid_conditional_uncertainty_keV": centroid_uncertainty,
                "centroid_minus_authoritative_keV": centroid - component.energy_keV,
                "centroid_shift_native_channels": (
                    (centroid - component.energy_keV) / spectrum.calibration_A1
                ),
                "centroid_shift_fitted_sigma": (
                    (centroid - component.energy_keV) / sigma
                    if sigma > 0
                    else float("nan")
                ),
                "fitted_sigma_keV": sigma,
                "fitted_sigma_conditional_uncertainty_keV": sigma_uncertainty,
                "full_line_rate_counts_per_s": rate,
                "full_line_rate_conditional_uncertainty_counts_per_s": rate_sigma,
                "yield_active_bound": bound,
                "all_active_bounds": ";".join(result.active_bounds),
                "covariance_note": (
                    "local conditional fit with one free constant width; centroid "
                    "uncertainty includes fitted offset/stretch covariance"
                ),
            }
            records.append(record)
            file_rows.append(record)
        usable = [record for record in records if record["available_for_drift"]]
        if len(usable) < 2:
            summaries.append(
                {
                    "component": component.name,
                    "available_file_count": len(usable),
                    "centroid_spread_keV": float("nan"),
                    "centroid_spread_native_channels": float("nan"),
                    "centroid_spread_fraction_of_median_sigma": float("nan"),
                    "centroid_spread_combined_conditional_sigma": float("nan"),
                    "centroid_threshold_keV": float("nan"),
                    "centroid_threshold_exceeded": False,
                    "maximum_pair_high_file_id": "",
                    "maximum_pair_low_file_id": "",
                    "base_mixture_sigma_keV": float("nan"),
                    "centroid_mixed_sigma_keV": float("nan"),
                    "mixture_sigma_fractional_increase": float("nan"),
                    "mixture_broadening_exceeds_5_percent": False,
                    "width_spread_fraction_of_median": float("nan"),
                    "width_spread_combined_conditional_sigma": float("nan"),
                    "maximum_width_file_id": "",
                    "minimum_width_file_id": "",
                }
            )
            continue
        centroids = np.asarray(
            [float(record["fitted_centroid_on_released_grid_keV"]) for record in usable]
        )
        sigmas = np.asarray([float(record["fitted_sigma_keV"]) for record in usable])
        centroid_uncertainties = np.asarray(
            [
                float(record["fitted_centroid_conditional_uncertainty_keV"])
                for record in usable
            ]
        )
        sigma_uncertainties = np.asarray(
            [
                float(record["fitted_sigma_conditional_uncertainty_keV"])
                for record in usable
            ]
        )
        weights = np.asarray(
            [
                float(record["full_line_rate_counts_per_s"])
                * float(record["live_time_s"])
                for record in usable
            ]
        )
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights /= weights.sum()
        high = int(np.argmax(centroids))
        low = int(np.argmin(centroids))
        spread = float(centroids[high] - centroids[low])
        centroid_spread_uncertainty = sqrt(
            centroid_uncertainties[high] ** 2 + centroid_uncertainties[low] ** 2
        )
        widest = int(np.argmax(sigmas))
        narrowest = int(np.argmin(sigmas))
        width_spread = float(sigmas[widest] - sigmas[narrowest])
        width_spread_uncertainty = sqrt(
            sigma_uncertainties[widest] ** 2 + sigma_uncertainties[narrowest] ** 2
        )
        median_sigma = float(np.median(sigmas))
        threshold = min(0.5 * accumulated.calibration_A1, 0.5 * median_sigma)
        mean_centroid = float(weights @ centroids)
        base_variance = float(weights @ (sigmas**2))
        mixed_variance = float(
            weights @ (sigmas**2 + (centroids - mean_centroid) ** 2)
        )
        base_sigma = sqrt(max(base_variance, 0.0))
        mixed_sigma = sqrt(max(mixed_variance, 0.0))
        broadening = mixed_sigma / base_sigma - 1.0 if base_sigma > 0 else float("nan")
        summaries.append(
            {
                "component": component.name,
                "available_file_count": len(usable),
                "centroid_spread_keV": spread,
                "centroid_spread_native_channels": spread / accumulated.calibration_A1,
                "centroid_spread_fraction_of_median_sigma": (
                    spread / median_sigma if median_sigma > 0 else float("nan")
                ),
                "centroid_spread_combined_conditional_sigma": (
                    spread / centroid_spread_uncertainty
                    if centroid_spread_uncertainty > 0
                    else float("nan")
                ),
                "centroid_threshold_keV": threshold,
                "centroid_threshold_exceeded": spread > threshold,
                "maximum_pair_high_file_id": usable[high]["file_id"],
                "maximum_pair_low_file_id": usable[low]["file_id"],
                "base_mixture_sigma_keV": base_sigma,
                "centroid_mixed_sigma_keV": mixed_sigma,
                "mixture_sigma_fractional_increase": broadening,
                "mixture_broadening_exceeds_5_percent": broadening > 0.05,
                "width_spread_fraction_of_median": (
                    width_spread / median_sigma
                    if median_sigma > 0
                    else float("nan")
                ),
                "width_spread_combined_conditional_sigma": (
                    width_spread / width_spread_uncertainty
                    if width_spread_uncertainty > 0
                    else float("nan")
                ),
                "maximum_width_file_id": usable[widest]["file_id"],
                "minimum_width_file_id": usable[narrowest]["file_id"],
            }
        )
    coherent_pairs: dict[tuple[int, int], int] = {}
    for summary in summaries:
        if not summary["centroid_threshold_exceeded"]:
            continue
        high = summary["maximum_pair_high_file_id"]
        low = summary["maximum_pair_low_file_id"]
        if high != "" and low != "":
            pair = (int(high), int(low))
            coherent_pairs[pair] = coherent_pairs.get(pair, 0) + 1
    condition_centroid = max(coherent_pairs.values(), default=0) >= 2
    condition_mixture = any(
        bool(summary["mixture_broadening_exceeds_5_percent"])
        for summary in summaries
    )
    condition_area = any(
        bool(record["movement_exceeds_canonical_conditional_1sigma"])
        for record in per_window_lookup.values()
    )
    invalid = condition_centroid and condition_mixture and condition_area
    diagnostics = {
        "coherent_same_sign_multi_anchor_centroid_threshold_exceeded": condition_centroid,
        "coherent_high_minus_low_file_pair_anchor_counts": {
            f"{high}-{low}": count
            for (high, low), count in sorted(coherent_pairs.items())
        },
        "any_anchor_mixture_broadening_exceeds_5_percent": condition_mixture,
        "any_primary_ratio_per_window_movement_exceeds_1sigma": condition_area,
        "all_three_invalidation_conditions_met": invalid,
        "sum_first_accumulation_valid_for_primary_fit": not invalid,
        "high_energy_limitation": workflow["drift_diagnostic"][
            "high_energy_limitation"
        ],
        "decision": (
            "stop; evidence invalidates accumulated primary fit"
            if invalid
            else "continue; predeclared three-part invalidation rule not met"
        ),
    }
    return file_rows, summaries, diagnostics


def _line_bound(result: JointPeakFitResult, component: str) -> bool:
    return f"line.{component}.rate_counts_per_s" in result.active_bounds


def _variant_ratio_lookup(
    results: Mapping[str, JointPeakFitResult],
) -> dict[str, dict[str, tuple[float, float]]]:
    output: dict[str, dict[str, tuple[float, float]]] = {}
    for variant, result in results.items():
        ratios = _ratio_result(result)
        output[variant] = {
            name: (
                float(ratios.values[index]),
                sqrt(max(float(ratios.covariance[index, index]), 0.0)),
            )
            for index, name in enumerate(ratios.labels)
        }
    return output


def _row_map(
    table_config: Mapping[str, Any],
    historical: Mapping[str, Any],
    canonical: JointPeakFitResult,
    canonical_spec: JointPeakSpec,
    results: Mapping[str, JointPeakFitResult],
    per_window: Mapping[str, Mapping[str, Any]],
    residual_rows: Sequence[Mapping[str, Any]],
    phase2_rows: Sequence[Mapping[str, Any]],
    drift_diagnostics: Mapping[str, Any],
    live_time_s: float,
) -> tuple[list[dict[str, Any]], bool]:
    thresholds = table_config["sum_first_local_workflow"]["row_thresholds"]
    quantitative_z = float(thresholds["supported_conditional_z_minimum"])
    zero_compatible_z = float(
        thresholds["zero_compatible_conditional_z_maximum"]
    )
    residual_threshold = float(thresholds["material_absolute_poisson_residual"])
    by_name, by_energy = _component_metadata(table_config)
    primary_by_energy = {
        round(_paper_energy(by_name[component.name]), 6): component
        for component in canonical_spec.components
    }
    component_by_name = {
        component.name: component for component in canonical_spec.components
    }
    ratios = _ratio_result(canonical)
    ratio_lookup = {
        name: (
            float(ratios.values[index]),
            sqrt(max(float(ratios.covariance[index, index]), 0.0)),
        )
        for index, name in enumerate(ratios.labels)
    }
    variant_ratios = _variant_ratio_lookup(results)
    residual_lookup = {str(row["window"]): row for row in residual_rows}
    phase2_lookup = {
        round(float(row["paper_row_energy_keV"]), 6): row
        for row in phase2_rows
        if row.get("paper_row_energy_keV", "") not in (None, "")
    }

    def flags(component_name: str) -> dict[str, Any]:
        component = component_by_name[component_name]
        local = per_window[component_name]
        index = canonical.line_names.index(component_name)
        rate = float(canonical.line_rates_counts_per_s[index])
        rate_sigma = sqrt(
            max(float(canonical.line_rate_covariance[index, index]), 0.0)
        )
        ratio, ratio_sigma = ratio_lookup[component_name]
        affine_ratio = variant_ratios["no_tail_affine"][component_name][0]
        tail_ratio = variant_ratios["common_left_tail_quadratic"][component_name][0]
        background_shift = abs(ratio - affine_ratio)
        tail_shift = abs(ratio - tail_ratio)
        residual = residual_lookup[component.window]
        return {
            "rate": rate,
            "rate_sigma": rate_sigma,
            "z": rate / rate_sigma if rate_sigma > 0 else float("nan"),
            "bound": _line_bound(canonical, component_name),
            "ratio": ratio,
            "ratio_sigma": ratio_sigma,
            "affine_shift": background_shift,
            "tail_shift": tail_shift,
            "background_sensitive": background_shift > ratio_sigma,
            "local_identified": bool(
                local["fit_success"]
                and local["fisher_covariance_valid"]
                and not str(local["active_bounds"])
            ),
            "per_window_sensitive": bool(
                local[
                    "movement_exceeds_canonical_conditional_1sigma"
                ]
            ),
            "residual_gt4": int(residual["bins_with_absolute_residual_gt_4"]),
            "residual_material": (
                float(residual["largest_absolute_poisson_residual"])
                > residual_threshold
            ),
            "window": component.window,
        }

    reference_flags = flags(REFERENCE)
    reference_supported = bool(
        canonical.success
        and canonical.fisher_covariance_valid
        and not reference_flags["bound"]
        and reference_flags["z"] >= quantitative_z
        and reference_flags["local_identified"]
        and not reference_flags["per_window_sensitive"]
        and not reference_flags["background_sensitive"]
        and not reference_flags["residual_material"]
        and not drift_diagnostics[
            "all_three_invalidation_conditions_met"
        ]
    )
    hard_status = {
        478.0: (
            STATUS_BROAD,
            "B-10 reaction feature is broad; no released normalized non-Gaussian template",
        ),
        609.3: (
            STATUS_UNRELIABLE,
            "609.321-keV line is visible, but the neighboring 595.85-keV intrinsic-Ge response lacks a normalized template",
        ),
        725.0: (
            STATUS_BLEND,
            "Cd-113 725.298, Ac-228 726.863, and Bi-212 727.330 are locally unresolved",
        ),
        1399.6: (
            STATUS_BLEND,
            "1399.638--1401.515-keV target/neighbor yields are non-identifiable",
        ),
        5433.1: (
            STATUS_UNAVAILABLE,
            "target and 5429.3-keV nuisance are boundary-active in the sparse local window",
        ),
        5824.6: (
            STATUS_UNAVAILABLE,
            "sparse-window quadratic-background target yield is boundary-active",
        ),
        7367.9: (
            STATUS_UNAVAILABLE,
            "target is zero-compatible; neighboring structure fails the recurrence rule and remains unidentified",
        ),
        7916.3: (
            STATUS_UNAVAILABLE,
            "target and 7908.4-keV nuisance are boundary-active in the sparse local window",
        ),
    }
    rows: list[dict[str, Any]] = []
    for historical_row in historical["exact_legacy_ratios"]:
        paper = float(historical_row["energy_keV"])
        key = round(paper, 6)
        item = by_energy[key]
        component = primary_by_energy.get(key)
        recommended_ratio: float | str = ""
        recommended_sigma: float | str = ""
        diagnostic_ratio: float | str = ""
        diagnostic_sigma: float | str = ""
        fitted_counts: float | str = ""
        fitted_counts_sigma: float | str = ""
        rate: float | str = ""
        rate_sigma: float | str = ""
        interval_kind = "unavailable"
        model_background_shift: float | str = ""
        model_tail_shift: float | str = ""
        per_window_shift: float | str = ""
        local_residual_gt4: int | str = ""
        if component is None:
            status, reason = hard_status[paper]
            component_name = str(item.get("name", ""))
        else:
            component_name = component.name
            current = flags(component_name)
            diagnostic_ratio = current["ratio"]
            diagnostic_sigma = current["ratio_sigma"]
            rate = current["rate"]
            rate_sigma = current["rate_sigma"]
            fitted_counts = current["rate"] * live_time_s
            fitted_counts_sigma = current["rate_sigma"] * live_time_s
            model_background_shift = current["affine_shift"]
            model_tail_shift = current["tail_shift"]
            per_window_shift = per_window[component_name]["absolute_ratio_movement"]
            local_residual_gt4 = current["residual_gt4"]
            if paper in hard_status:
                status, reason = hard_status[paper]
            elif not reference_supported:
                status = STATUS_UNRELIABLE
                reason = "558.456-keV reference did not pass the quantitative rule"
            elif current["bound"]:
                status = STATUS_UNAVAILABLE
                reason = "fitted target yield is boundary-active"
            elif not isfinite(current["z"]) or current["z"] < zero_compatible_z:
                status = STATUS_UNAVAILABLE
                reason = "conditional fitted-yield interval includes zero; no symmetric ratio reported"
            elif current["z"] < quantitative_z:
                status = STATUS_UNRELIABLE
                reason = "visible line is below the declared 3-sigma quantitative threshold"
            elif not current["local_identified"]:
                status = STATUS_UNRELIABLE
                reason = "independent-window sensitivity is not full-rank and interior"
            elif current["per_window_sensitive"]:
                status = STATUS_UNRELIABLE
                reason = "per-window free-width ratio moves by more than one canonical conditional sigma"
            elif current["background_sensitive"]:
                status = STATUS_UNRELIABLE
                reason = "affine-versus-quadratic local-background ratio shift exceeds one conditional sigma"
            elif current["residual_material"]:
                status = STATUS_UNRELIABLE
                reason = "local window retains at least one greater-than-4-sigma Poisson residual"
            else:
                status = STATUS_QUANTITATIVE
                reason = "interior identified yield stable under declared local sensitivities"
            if status == STATUS_QUANTITATIVE:
                recommended_ratio = current["ratio"]
                recommended_sigma = current["ratio_sigma"]
                interval_kind = (
                    "exact_self_ratio"
                    if component_name == REFERENCE
                    else "conditional_fisher_symmetric"
                )
        phase2 = phase2_lookup.get(key, {})
        rows.append(
            {
                "paper_energy_keV": paper,
                "authoritative_energy_keV": item.get("energy_keV", ""),
                "component": component_name,
                "identity": item.get("identity", item.get("name", "")),
                "published_legacy_density_ratio": (
                    "" if historical_row.get("paper_display") == "--" else historical_row["ratio"]
                ),
                "published_historical_uncertainty": (
                    ""
                    if historical_row.get("paper_display") == "--"
                    else historical_row["historical_uncertainty"]
                ),
                "exact_historical_replay_ratio": historical_row["ratio"],
                "exact_historical_replay_uncertainty": historical_row[
                    "historical_uncertainty"
                ],
                "repaired_historical_same_fit_full_line_ratio": historical_row[
                    "repaired_full_line_ratio"
                ],
                "repaired_historical_same_fit_conditional_uncertainty": historical_row[
                    "repaired_full_line_conditional_uncertainty"
                ],
                "prior_phase2_conditional_ratio": phase2.get(
                    "aggregate_ratio_to_558_5_keV", ""
                ),
                "prior_phase2_conditional_uncertainty": phase2.get(
                    "aggregate_ratio_fisher_uncertainty", ""
                ),
                "sum_first_fitted_full_line_detector_counts": fitted_counts,
                "sum_first_fitted_counts_conditional_uncertainty": fitted_counts_sigma,
                "sum_first_full_line_rate_counts_per_s": rate,
                "sum_first_rate_conditional_uncertainty_counts_per_s": rate_sigma,
                "sum_first_diagnostic_ratio": diagnostic_ratio,
                "sum_first_diagnostic_ratio_conditional_uncertainty": diagnostic_sigma,
                "recommended_relative_detected_count": recommended_ratio,
                "recommended_conditional_uncertainty": recommended_sigma,
                "interval_kind": interval_kind,
                "affine_background_absolute_ratio_shift": model_background_shift,
                "unsupported_tail_absolute_ratio_shift": model_tail_shift,
                "per_window_free_width_absolute_ratio_shift": per_window_shift,
                "local_bins_with_absolute_residual_gt_4": local_residual_gt4,
                "row_status": status,
                "status_reason": reason,
                "reference_supported_quantitative": reference_supported,
                "uncertainty_semantics": (
                    "reported uncertainty is local conditional Fisher covariance; "
                    "model shifts are separate and no global deviance multiplier is applied"
                ),
                "scientific_scope": (
                    "relative detected peak counts only; no efficiency correction, "
                    "unfolding, source rate, activity, flux, abundance, or response validation"
                ),
            }
        )
    if len(rows) != 35:
        raise RuntimeError("Table 3 sum-first map is not exactly 35 rows")
    return rows, reference_supported


def analyze_table3_sum_first(
    spectra: Sequence[PublicSpectrum],
    table_config: Mapping[str, Any],
    historical: Mapping[str, Any],
    calibration: CalibrationConstraint,
    resolution: LinearResolution,
    *,
    phase2_rows: Sequence[Mapping[str, Any]] = (),
    minimum_expected_counts: float = 5.0,
) -> Table3SumFirstAnalysis:
    """Run the declared corrected sum-first workflow and return generated rows."""

    inputs = tuple(spectra)
    accumulated = accumulate_spectra_exact(
        inputs, name="Cycle493_RD_low_gain_exact_sum"
    )
    accumulation = _assert_accumulation(
        inputs, accumulated, table_config, historical
    )
    results, specs, resolutions, _, channel_ranges = _fit_variants(
        accumulated, table_config, historical, calibration, resolution
    )
    canonical_name = "no_tail_quadratic"
    canonical = results[canonical_name]
    canonical_spec = specs[canonical_name]
    canonical_resolution = resolutions[canonical_name]
    live_time = accumulated.live_time
    row_thresholds = table_config["sum_first_local_workflow"]["row_thresholds"]

    per_window_rows, per_window_lookup = _per_window_sensitivity(
        accumulated,
        canonical,
        canonical_spec,
        calibration,
        canonical_resolution,
        float(row_thresholds["material_ratio_shift_conditional_sigma"]),
    )
    drift_rows, drift_summary, drift = _drift_diagnostics(
        inputs,
        accumulated,
        canonical,
        canonical_spec,
        calibration,
        canonical_resolution,
        table_config["sum_first_local_workflow"],
        per_window_lookup,
    )
    if drift["all_three_invalidation_conditions_met"]:
        drift["primary_reporting_action"] = (
            "no quantitative accumulated ratios; retain diagnostic rows only"
        )
    else:
        drift["primary_reporting_action"] = (
            "accumulation remains valid; assess each row locally"
        )
    residual_rows = window_residual_diagnostics(
        canonical,
        (accumulated,),
        canonical_spec,
        calibration,
        canonical_resolution,
        minimum_expected_counts,
    )
    cluster_rows = _cluster_diagnostics(
        inputs,
        accumulated,
        table_config,
        historical,
        canonical,
        calibration,
        canonical_resolution,
    )

    candidate_rows, reference_supported = _row_map(
        table_config,
        historical,
        canonical,
        canonical_spec,
        results,
        per_window_lookup,
        residual_rows,
        phase2_rows,
        drift,
        live_time,
    )

    ratio_result = _ratio_result(canonical)
    count_covariance = canonical.line_rate_covariance * live_time**2
    model_rows = _model_comparison_rows(results, specs, table_config)
    tail = results["common_left_tail_quadratic"]
    tail_bound = any(name.startswith("shape.") for name in tail.active_bounds)
    products: dict[str, list[dict[str, Any]]] = {
        "table3_sum_first_candidate.csv": candidate_rows,
        "table3_sum_first_model_comparison.csv": model_rows,
        "table3_sum_first_per_window_sensitivity.csv": per_window_rows,
        "table3_sum_first_window_diagnostics.csv": residual_rows,
        "table3_sum_first_weak_branch_diagnostics.csv": cluster_rows,
        "table3_sum_first_per_file_drift.csv": drift_rows,
        "table3_sum_first_drift_summary.csv": drift_summary,
        "table3_sum_first_full_parameter_covariance.csv": (
            _parameter_covariance_rows(canonical)
        ),
        "table3_sum_first_full_line_count_covariance.csv": _covariance_rows(
            canonical.line_names, count_covariance, "detector counts^2"
        ),
        "table3_sum_first_ratio_covariance.csv": _covariance_rows(
            ratio_result.labels, ratio_result.covariance, "dimensionless^2"
        ),
    }
    diagnostics = {
        "result_semantics": table_config["sum_first_local_workflow"][
            "result_semantics"
        ],
        "accumulation_runtime_assertion": accumulation,
        "historical_window_channel_ranges_1_based_inclusive": {
            name: list(bounds) for name, bounds in sorted(channel_ranges.items())
        },
        "canonical_model": canonical_name,
        "canonical_poisson_deviance": canonical.poisson_deviance,
        "canonical_native_bin_count": canonical.observed_counts.size,
        "canonical_parameter_count": len(canonical.parameter_names),
        "canonical_descriptive_degrees_of_freedom": canonical.degrees_of_freedom,
        "canonical_fisher_covariance_valid": canonical.fisher_covariance_valid,
        "canonical_active_bounds": list(canonical.active_bounds),
        "calibration_nuisance": table_config["sum_first_local_workflow"][
            "calibration_nuisance"
        ],
        "fitted_calibration_offset_keV": canonical.parameter(
            "calibration.offset_keV"
        ),
        "fitted_calibration_fractional_gain_stretch": canonical.parameter(
            "calibration.fractional_gain_stretch"
        ),
        "tail_locally_supported": not tail_bound,
        "tail_reporting_action": (
            "sensitivity only; fitted shape nuisance is boundary-active"
            if tail_bound
            else "requires morphology review; not selected by p-value"
        ),
        "background_reporting_action": (
            "quadratic retained uniformly from the historical local structure; "
            "affine row shifts reported separately"
        ),
        "reference_component": REFERENCE,
        "reference_supported_quantitative": reference_supported,
        "reference_self_ratio": 1.0,
        "reference_self_ratio_variance": 0.0,
        "drift": drift,
        "uncertainty_scope": (
            "canonical covariance is full fitted conditional covariance; model "
            "and per-window shifts remain separate; no global deviance multiplier "
            "or exact-coverage claim"
        ),
        "global_fit_action": (
            "global deviance is descriptive only; row classifications control reporting"
        ),
        "weak_branch_action": (
            "no cluster is called explained without a constrained held-line prediction; "
            "free candidate amplitudes are flexible background"
        ),
    }
    return Table3SumFirstAnalysis(
        products,
        diagnostics,
        accumulated,
        canonical,
        canonical_spec,
        calibration,
        canonical_resolution,
    )
