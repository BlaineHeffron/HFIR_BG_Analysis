"""Local native-channel detector-count ratios for descriptive paper Table 8.

This module composes :mod:`src.public_data.peak_likelihood`.  It does not
unfold the measured spectrum, infer an incident flux, or compare against a
simulation response.  Five independent physical domains replace the rejected
table-wide absolute model; covariance is complete within each domain and zero
between disjoint independently fitted domains by construction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from math import isfinite, sqrt
from typing import Any, Mapping, Sequence

import numpy as np

from src.public_data.browser import PublicSpectrum
from src.public_data.peak_likelihood import (
    CalibrationConstraint,
    FitWindow,
    JointPeakFitResult,
    JointPeakSpec,
    LineComponent,
    LinearResolution,
    ProfileInterval,
    RatioDefinition,
    RatioResult,
    fit_joint_peak_model,
    profile_ratio_interval,
    peak_shape_bin_probabilities,
    ratio_values_and_covariance,
    ratios_from_fit,
    calibrated_channel_energy,
    resolution_sigma_and_derivatives,
)
from src.public_data.peak_residuals import (
    residual_bin_diagnostics,
    window_residual_diagnostics,
)


CANONICAL_MODEL = "common_left_tail_quadratic"
STATUS_LABELS = {
    "Q": "supported quantitative",
    "V": "quantitative but explicitly model-sensitive",
    "R": "visible component but ratio unreliable",
    "X": "unresolved blend",
    "B": "boundary/upper limit",
    "U": "unsupported/unavailable",
}
ROLE_PAIRS = (
    ("fep/sep", "fep", "sep"),
    ("fep/dep", "fep", "dep"),
    ("sep/dep", "sep", "dep"),
)


@dataclass(frozen=True)
class Table8LocalFitBinJob:
    """One canonical group fit to export as generated native-bin evidence."""

    filename: str
    result: JointPeakFitResult
    spec: JointPeakSpec
    calibration: CalibrationConstraint
    resolution: LinearResolution


@dataclass(frozen=True)
class Table8LocalAnalysis:
    """Generated products, serializable diagnostics, and fit-bin export jobs."""

    products: Mapping[str, list[dict[str, Any]]]
    diagnostics: Mapping[str, Any]
    fit_bin_jobs: tuple[Table8LocalFitBinJob, ...]


@dataclass(frozen=True)
class _GroupFits:
    name: str
    parents: tuple[str, ...]
    canonical_result: JointPeakFitResult
    canonical_spec: JointPeakSpec
    canonical_resolution: LinearResolution
    variants: Mapping[str, JointPeakFitResult]
    variant_specs: Mapping[str, JointPeakSpec]
    variant_resolutions: Mapping[str, LinearResolution]
    role_results: Mapping[str, JointPeakFitResult]
    role_specs: Mapping[str, JointPeakSpec]
    role_resolutions: Mapping[str, LinearResolution]
    basin_stability_rows: tuple[Mapping[str, Any], ...]


def _array_sha256(values: np.ndarray, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
    return sha256(array.tobytes()).hexdigest()


def table8_ratio_definitions(
    components: Sequence[LineComponent], parents: Sequence[str]
) -> tuple[RatioDefinition, ...]:
    """Return the frozen three ratios per parent in authoritative energy order."""

    by_parent: dict[str, dict[str, str]] = {}
    for component in components:
        if component.parent in parents and component.role in {"fep", "sep", "dep"}:
            by_parent.setdefault(component.parent, {})[component.role] = component.name
    definitions: list[RatioDefinition] = []
    for parent in sorted(parents, key=float):
        roles = by_parent.get(parent, {})
        if set(roles) != {"fep", "sep", "dep"}:
            raise RuntimeError(f"Table 8 parent {parent} lacks exactly FEP/SEP/DEP")
        for label, numerator, denominator in ROLE_PAIRS:
            definitions.append(
                RatioDefinition(
                    f"{parent}:{label}", roles[numerator], roles[denominator]
                )
            )
    return tuple(definitions)


def ratio_stability_summary(
    canonical_value: float,
    canonical_sd: float,
    variant_values: Mapping[str, float],
) -> dict[str, Any]:
    """Apply the frozen one-SD-or-five-percent movement rule."""

    threshold = max(canonical_sd, 0.05 * abs(canonical_value))
    movements = {
        name: float(value - canonical_value)
        for name, value in variant_values.items()
        if isfinite(float(value))
    }
    if movements:
        driver, signed = max(movements.items(), key=lambda item: abs(item[1]))
        maximum = abs(signed)
    else:
        driver, signed, maximum = "", float("nan"), float("nan")
    sigma_units = maximum / canonical_sd if canonical_sd > 0 else float("inf")
    relative = maximum / abs(canonical_value) if canonical_value else float("inf")
    passes = bool(isfinite(maximum) and maximum <= threshold)
    floor_only = bool(
        passes
        and maximum > canonical_sd
        and 0.05 * abs(canonical_value) > canonical_sd
    )
    return {
        "threshold": threshold,
        "maximum": maximum,
        "signed": signed,
        "driver": driver,
        "sigma_units": sigma_units,
        "relative": relative,
        "passes": passes,
        "five_percent_floor_only": floor_only,
    }


def assemble_independent_ratio_covariance(
    label_blocks: Sequence[Sequence[str]],
    covariance_blocks: Sequence[np.ndarray],
) -> tuple[tuple[str, ...], np.ndarray]:
    """Preserve within-group covariance and set independent cross-group terms zero."""

    if len(label_blocks) != len(covariance_blocks):
        raise ValueError("ratio label/covariance block counts differ")
    labels = tuple(label for block in label_blocks for label in block)
    covariance = np.zeros((len(labels), len(labels)), dtype=np.float64)
    cursor = 0
    for block_labels, block_values in zip(label_blocks, covariance_blocks):
        block = np.asarray(block_values, dtype=np.float64)
        size = len(block_labels)
        if block.shape != (size, size):
            raise ValueError("ratio covariance block does not match its labels")
        if not np.allclose(block, block.T, rtol=1e-10, atol=1e-14):
            raise ValueError("ratio covariance block must be symmetric")
        stop = cursor + size
        covariance[cursor:stop, cursor:stop] = block
        cursor = stop
    if len(set(labels)) != len(labels):
        raise ValueError("ratio labels must be unique across independent groups")
    return labels, covariance


def classify_table8_ratio(
    *,
    canonical_available: bool,
    profile_kind: str,
    profile_excludes_zero: bool,
    numerator_z: float,
    denominator_z: float,
    numerator_bound: bool,
    denominator_bound: bool,
    covariance_valid: bool,
    unresolved: bool,
    target_core_residual: bool,
    model_sensitive: bool,
    mandatory_model_sensitive: bool,
    tail_status_differs: bool,
    weak_z_threshold: float = 3.0,
) -> tuple[str, str]:
    """Map frozen local evidence to one of the six predeclared statuses."""

    if not canonical_available or denominator_bound:
        return "U", "canonical fit/denominator does not support a finite ratio"
    if numerator_bound:
        if profile_kind == "upper_limit":
            return "B", "numerator is on its nonnegative bound; valid upper profile"
        return "U", "boundary numerator lacks a valid upper profile"
    weak = min(numerator_z, denominator_z) < weak_z_threshold
    if weak and not profile_excludes_zero:
        if profile_kind == "upper_limit":
            return "B", "weak numerator has a valid one-sided upper profile"
        return "U", "weak target yield lacks a two-sided nonzero profile"
    if unresolved:
        return "X", "declared target/neighbor or target/target blend is unresolved"
    if not covariance_valid:
        return "R", "target is visible but the complete conditional covariance failed"
    if target_core_residual:
        return "R", "absolute Pearson residual >=4 remains inside a target core"
    if model_sensitive or mandatory_model_sensitive or tail_status_differs:
        reasons = []
        if model_sensitive:
            reasons.append("declared model/role movement exceeds threshold")
        if mandatory_model_sensitive:
            reasons.append("Al-27/Ge-70 DEP ambiguity")
        if tail_status_differs:
            reasons.append("tail/no-tail status differs")
        return "V", "; ".join(reasons)
    return "Q", "identifiable interior yields; stable declared variants; clean target core"


def _assert_input(
    spectrum: PublicSpectrum, historical: Mapping[str, Any]
) -> dict[str, Any]:
    expected = historical["input"]
    count_hash = _array_sha256(spectrum.counts, "<i8")
    energy_hash = _array_sha256(spectrum.energy_keV, "<f8")
    exact = {
        "file_id": spectrum.file_id,
        "file_name": spectrum.file_name,
        "run_id": spectrum.run_id,
        "run_name": spectrum.run_name,
        "live_time_s": spectrum.live_time,
        "calibration_A0_keV": spectrum.calibration_A0,
        "calibration_A1_keV_per_channel": spectrum.calibration_A1,
        "channel_count": spectrum.counts.size,
        "detector_counts": int(np.sum(spectrum.counts, dtype=np.int64)),
        "counts_sha256_int64_little_endian": count_hash,
        "energy_grid_sha256_float64_little_endian": energy_hash,
    }
    for key, value in exact.items():
        if value != expected[key]:
            raise RuntimeError(f"frozen Table 8 input changed: {key}")
    return exact


def _native_windows(
    spectrum: PublicSpectrum,
    table_config: Mapping[str, Any],
    background_model: str,
) -> tuple[dict[str, FitWindow], dict[str, tuple[int, int]]]:
    configured = {item["name"]: item for item in table_config["windows"]}
    windows: dict[str, FitWindow] = {}
    ranges: dict[str, tuple[int, int]] = {}
    for record in table_config["local_workflow"]["native_windows"]:
        name = str(record["name"])
        if name not in configured:
            raise RuntimeError(f"local Table 8 window {name} is not phase-2 declared")
        first = int(record["first_channel_1_based"])
        last = int(record["last_channel_1_based"])
        if first < 1 or last < first or last > spectrum.counts.size:
            raise RuntimeError(f"invalid native channel range for {name}")
        low = spectrum.calibration_A0 + spectrum.calibration_A1 * (first - 0.5)
        high = spectrum.calibration_A0 + spectrum.calibration_A1 * (last + 0.5)
        selected = np.flatnonzero(
            (spectrum.energy_keV >= low) & (spectrum.energy_keV < high)
        )
        expected = np.arange(first - 1, last, dtype=np.int64)
        if not np.array_equal(selected, expected):
            raise RuntimeError(f"native-bin identity changed for {name}")
        phase2 = configured[name]
        phase2_selected = np.flatnonzero(
            (spectrum.energy_keV >= float(phase2["low_keV"]))
            & (spectrum.energy_keV < float(phase2["high_keV"]))
        )
        if not np.array_equal(phase2_selected, expected):
            raise RuntimeError(f"frozen local bins differ from phase-2 bins for {name}")
        windows[name] = FitWindow(name, low, high, background_model)
        ranges[name] = (first, last)
    return windows, ranges


def _candidate_records(table_config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item["name"]): item
        for item in table_config["component_audit"]["candidates"]
    }


def _target_records(table_config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = {str(item["name"]): item for item in table_config["components"]}
    parents = tuple(str(value) for value in table_config["target_parents_keV"])
    offsets = {"fep": 0.0, "sep": 511.0, "dep": 1022.0}
    for item in records.values():
        parent = str(item["parent"])
        role = str(item["role"])
        if parent not in parents or role not in offsets:
            raise RuntimeError("unexpected Table 8 target component identity")
        exact = float(parent) - offsets[role]
        if not np.isclose(
            float(item["energy_keV"]), exact, rtol=0.0, atol=1e-9
        ):
            raise RuntimeError(
                f"Table 8 component {item['name']} violates exact escape arithmetic"
            )
    if len(records) != 3 * len(parents):
        raise RuntimeError("Table 8 target set is not exactly eight triplets")
    return records


def _component(item: Mapping[str, Any], *, contaminant: bool = False) -> LineComponent:
    return LineComponent(
        str(item["name"]),
        float(item["energy_keV"]),
        str(item["window"]),
        "contaminant" if contaminant else str(item["role"]),
        str(item.get("parent", "")),
        "shared",
    )


def _group_spec(
    spectrum: PublicSpectrum,
    table_config: Mapping[str, Any],
    group: Mapping[str, Any],
    *,
    background_model: str,
    neighbors: Sequence[str],
    variant: str,
) -> JointPeakSpec:
    windows, _ = _native_windows(spectrum, table_config, background_model)
    group_windows = tuple(str(value) for value in group["windows"])
    group_parents = set(str(value) for value in group["parents_keV"])
    targets = [
        _component(item)
        for item in table_config["components"]
        if str(item["parent"]) in group_parents
        and str(item["window"]) in group_windows
    ]
    candidates = _candidate_records(table_config)
    additions = [_component(candidates[name], contaminant=True) for name in neighbors]
    components = tuple(targets + additions)
    if any(component.window not in group_windows for component in components):
        raise RuntimeError(f"component escaped declared group {group['name']}")
    definitions = table8_ratio_definitions(components, tuple(group_parents))
    if len(definitions) != 3 * len(group_parents):
        raise RuntimeError("local Table 8 group lost a target ratio")
    return JointPeakSpec(
        f"paper-table-8-local:{group['name']}:{variant}",
        tuple(windows[name] for name in group_windows),
        components,
    )


def _resolution_for_variant(
    base: LinearResolution, variant: Mapping[str, Any]
) -> LinearResolution:
    tail_model = str(variant["tail_model"])
    return replace(
        base,
        tail_model=tail_model,
        tail_fraction=base.tail_fraction if tail_model == "constant" else 0.0,
        per_run_scale_sigma=0.0,
    )


def _constant_role_resolution(
    base: LinearResolution,
    canonical: JointPeakFitResult,
    components: Sequence[LineComponent],
) -> LinearResolution:
    energy = float(np.mean([component.energy_keV for component in components]))
    sigma = resolution_sigma_and_derivatives(
        base,
        energy,
        canonical.parameter("resolution.intercept_keV"),
        canonical.parameter("resolution.linear_sigma_slope_keV_per_keV"),
    )[0]
    return replace(
        base,
        intercept_keV=sigma,
        intercept_bounds_keV=(0.05, 20.0),
        slope=0.0,
        form="constant",
        tail_model="none",
        tail_fraction=0.0,
        per_run_scale_sigma=0.0,
    )


def _basin_stability_rows(
    spectrum: PublicSpectrum,
    group: Mapping[str, Any],
    workflow: Mapping[str, Any],
    spec: JointPeakSpec,
    calibration: CalibrationConstraint,
    resolution: LinearResolution,
    canonical: JointPeakFitResult,
    canonical_initial: Mapping[str, float],
) -> tuple[Mapping[str, Any], ...]:
    """Fail closed unless enumerated 11386-keV restarts recover one optimum."""

    check = workflow["canonical_basin_stability_check"]
    if str(group["name"]) != str(check["group"]):
        return ()
    if check["initial_source"] != "canonical_initial_from_frozen_phase2":
        raise RuntimeError("Table 8 basin check initialization source changed")
    acceptance = check["acceptance"]
    definitions = table8_ratio_definitions(spec.components, group["parents_keV"])
    canonical_ratios = ratios_from_fit(canonical, definitions)
    rows: list[Mapping[str, Any]] = []
    for perturbation in check["perturbations"]:
        factor = float(perturbation["multiplicative_factor"])
        initial = {
            name: float(value) * factor
            for name, value in canonical_initial.items()
        }
        result = fit_joint_peak_model(
            (spectrum,), spec, calibration, resolution, initial=initial
        )
        success_full_rank = bool(
            result.success
            and result.fisher_covariance_valid
            and result.fisher_rank == len(result.parameter_names)
        )
        same_parameter_identity = bool(
            result.parameter_names == canonical.parameter_names
            and result.line_names == canonical.line_names
        )
        if success_full_rank and same_parameter_identity:
            ratios = ratios_from_fit(result, definitions)
            maximum_ratio_difference = float(
                np.max(np.abs(ratios.values - canonical_ratios.values), initial=0.0)
            )
            maximum_parameter_difference = float(
                np.max(
                    np.abs(result.parameter_values - canonical.parameter_values),
                    initial=0.0,
                )
            )
            maximum_expected_difference = float(
                np.max(
                    np.abs(result.expected_counts - canonical.expected_counts),
                    initial=0.0,
                )
            )
        else:
            maximum_ratio_difference = float("inf")
            maximum_parameter_difference = float("inf")
            maximum_expected_difference = float("inf")
        penalized_nll_difference = float(
            result.penalized_nll - canonical.penalized_nll
        )
        deviance_difference = float(
            result.poisson_deviance - canonical.poisson_deviance
        )
        identical_active_bounds = result.active_bounds == canonical.active_bounds
        passed = bool(
            same_parameter_identity
            and abs(penalized_nll_difference)
            <= float(acceptance["maximum_absolute_penalized_nll_difference"])
            and abs(deviance_difference)
            <= float(acceptance["maximum_absolute_poisson_deviance_difference"])
            and maximum_parameter_difference
            <= float(acceptance["maximum_absolute_parameter_difference"])
            and maximum_ratio_difference
            <= float(acceptance["maximum_absolute_ratio_difference"])
            and (
                identical_active_bounds
                or not bool(acceptance["require_identical_active_bounds"])
            )
            and (
                success_full_rank
                or not bool(
                    acceptance["require_success_full_rank_covariance"]
                )
            )
        )
        rows.append(
            {
                "group": str(group["name"]),
                "restart": str(perturbation["name"]),
                "multiplicative_factor": factor,
                "initial_source": str(check["initial_source"]),
                "copied_initial_values_json": json.dumps(
                    canonical_initial, sort_keys=True, separators=(",", ":")
                ),
                "perturbed_initial_values_json": json.dumps(
                    initial, sort_keys=True, separators=(",", ":")
                ),
                "success_full_rank_covariance": success_full_rank,
                "same_parameter_and_line_identity": same_parameter_identity,
                "identical_active_bounds": identical_active_bounds,
                "active_bounds_json": json.dumps(result.active_bounds),
                "penalized_nll": result.penalized_nll,
                "signed_penalized_nll_difference": penalized_nll_difference,
                "poisson_deviance": result.poisson_deviance,
                "signed_poisson_deviance_difference": deviance_difference,
                "maximum_absolute_parameter_difference": maximum_parameter_difference,
                "maximum_absolute_expected_bin_count_difference": maximum_expected_difference,
                "maximum_absolute_ratio_difference": maximum_ratio_difference,
                "acceptance_json": json.dumps(
                    acceptance, sort_keys=True, separators=(",", ":")
                ),
                "passed": passed,
                "semantics": str(check["role"]),
            }
        )
    if not rows or not all(bool(row["passed"]) for row in rows):
        raise RuntimeError(
            "Table 8 11386-keV canonical optimum failed deterministic "
            "basin-stability acceptance"
        )
    return tuple(rows)


def _fit_group(
    spectrum: PublicSpectrum,
    table_config: Mapping[str, Any],
    group: Mapping[str, Any],
    calibration: CalibrationConstraint,
    base_resolution: LinearResolution,
) -> _GroupFits:
    workflow = table_config["local_workflow"]
    model_definitions = {
        str(item["name"]): item for item in workflow["model_variants"]
    }
    canonical_definition = model_definitions[CANONICAL_MODEL]
    canonical_neighbors = tuple(str(value) for value in group["canonical_neighbors"])
    canonical_spec = _group_spec(
        spectrum,
        table_config,
        group,
        background_model=str(canonical_definition["background_model"]),
        neighbors=canonical_neighbors,
        variant=CANONICAL_MODEL,
    )
    canonical_resolution = _resolution_for_variant(
        base_resolution, canonical_definition
    )
    canonical_initial = {
        str(name): float(value)
        for name, value in workflow[
            "canonical_initial_from_frozen_phase2"
        ].items()
        if name != "role"
    }
    canonical = fit_joint_peak_model(
        (spectrum,),
        canonical_spec,
        calibration,
        canonical_resolution,
        initial=canonical_initial,
    )
    if not canonical.success:
        raise RuntimeError(
            f"Table 8 local canonical group {group['name']} failed: {canonical.message}"
        )
    basin_stability_rows = _basin_stability_rows(
        spectrum,
        group,
        workflow,
        canonical_spec,
        calibration,
        canonical_resolution,
        canonical,
        canonical_initial,
    )
    variants: dict[str, JointPeakFitResult] = {CANONICAL_MODEL: canonical}
    specs: dict[str, JointPeakSpec] = {CANONICAL_MODEL: canonical_spec}
    resolutions: dict[str, LinearResolution] = {
        CANONICAL_MODEL: canonical_resolution
    }
    for name, definition in model_definitions.items():
        if name == CANONICAL_MODEL:
            continue
        spec = _group_spec(
            spectrum,
            table_config,
            group,
            background_model=str(definition["background_model"]),
            neighbors=canonical_neighbors,
            variant=name,
        )
        resolution = _resolution_for_variant(base_resolution, definition)
        result = fit_joint_peak_model(
            (spectrum,),
            spec,
            calibration,
            resolution,
            warm_start=canonical,
            warm_start_source=f"Table 8 local {group['name']} canonical",
        )
        variants[name] = result
        specs[name] = spec
        resolutions[name] = resolution
    for definition in workflow["neighbor_variants"]:
        if str(group["name"]) not in definition["groups"]:
            continue
        name = str(definition["name"])
        neighbors = (
            ()
            if bool(definition["replace_canonical_neighbors"])
            else canonical_neighbors
        ) + tuple(str(value) for value in definition["neighbors"])
        spec = _group_spec(
            spectrum,
            table_config,
            group,
            background_model="quadratic",
            neighbors=neighbors,
            variant=name,
        )
        result = fit_joint_peak_model(
            (spectrum,),
            spec,
            calibration,
            canonical_resolution,
            warm_start=canonical,
            warm_start_source=f"Table 8 local {group['name']} canonical",
            allow_guarded_basin_restart=name == "steel_catalog_sensitivity",
            guarded_basin_restart_source=(
                "declared Table 8 steel-catalog sensitivity"
                if name == "steel_catalog_sensitivity"
                else ""
            ),
        )
        variants[name] = result
        specs[name] = spec
        resolutions[name] = canonical_resolution

    role_results: dict[str, JointPeakFitResult] = {}
    role_specs: dict[str, JointPeakSpec] = {}
    role_resolutions: dict[str, LinearResolution] = {}
    for window in canonical_spec.windows:
        components = tuple(
            component
            for component in canonical_spec.components
            if component.window == window.name
        )
        role = next(
            component.role
            for component in components
            if component.parent in group["parents_keV"]
        )
        local_window = replace(window, background_model="quadratic")
        local_spec = JointPeakSpec(
            f"{canonical_spec.name}:role-local:{role}",
            (local_window,),
            components,
        )
        target_components = tuple(
            component
            for component in components
            if component.parent in group["parents_keV"]
        )
        local_resolution = _constant_role_resolution(
            canonical_resolution, canonical, target_components
        )
        warm_values = canonical.parameter_values.copy()
        warm_values[canonical.parameter_names.index("resolution.intercept_keV")] = (
            local_resolution.intercept_keV
        )
        role_warm = replace(canonical, parameter_values=warm_values)
        result = fit_joint_peak_model(
            (spectrum,),
            local_spec,
            calibration,
            local_resolution,
            warm_start=role_warm,
            warm_start_source=(
                f"Table 8 local {group['name']} assembled canonical with "
                "effective-width initialization"
            ),
        )
        role_results[role] = result
        role_specs[role] = local_spec
        role_resolutions[role] = local_resolution
    return _GroupFits(
        str(group["name"]),
        tuple(str(value) for value in group["parents_keV"]),
        canonical,
        canonical_spec,
        canonical_resolution,
        variants,
        specs,
        resolutions,
        role_results,
        role_specs,
        role_resolutions,
        basin_stability_rows,
    )


def _line_correlation(result: JointPeakFitResult, left: str, right: str) -> float:
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


def _ratio_lookup(
    result: JointPeakFitResult, spec: JointPeakSpec, parents: Sequence[str]
) -> tuple[RatioResult, dict[str, tuple[float, float]]]:
    ratios = ratios_from_fit(result, table8_ratio_definitions(spec.components, parents))
    return ratios, {
        label: (
            float(ratios.values[index]),
            sqrt(max(float(ratios.covariance[index, index]), 0.0)),
        )
        for index, label in enumerate(ratios.labels)
    }


def _safe_ratio_lookup(
    result: JointPeakFitResult, spec: JointPeakSpec, parents: Sequence[str]
) -> dict[str, tuple[float, float]]:
    """Return diagnostic variant ratios; preserve unavailable zero denominators."""

    lookup: dict[str, tuple[float, float]] = {}
    for definition in table8_ratio_definitions(spec.components, parents):
        try:
            ratio = ratios_from_fit(result, (definition,))
        except ValueError:
            lookup[definition.name] = (float("nan"), float("nan"))
            continue
        lookup[definition.name] = (
            float(ratio.values[0]),
            sqrt(max(float(ratio.covariance[0, 0]), 0.0)),
        )
    return lookup


def _role_ratio_result(group: _GroupFits) -> RatioResult:
    target_components = tuple(
        component
        for component in group.canonical_spec.components
        if component.parent in group.parents
    )
    names = tuple(component.name for component in target_components)
    rates = np.full(len(names), float("nan"), dtype=np.float64)
    covariance = np.zeros((len(names), len(names)), dtype=np.float64)
    index = {name: position for position, name in enumerate(names)}
    for result in group.role_results.values():
        if not result.success or not result.fisher_covariance_valid:
            raise RuntimeError(
                f"Table 8 role-local covariance failed for group {group.name}"
            )
        selected = [name for name in result.line_names if name in index]
        for left in selected:
            i = index[left]
            source_i = result.line_names.index(left)
            rates[i] = result.line_rates_counts_per_s[source_i]
            for right in selected:
                j = index[right]
                source_j = result.line_names.index(right)
                covariance[i, j] = result.line_rate_covariance[source_i, source_j]
    if not np.isfinite(rates).all() or not np.isfinite(covariance).all():
        raise RuntimeError(
            f"Table 8 role-local ratio assembly incomplete for group {group.name}"
        )
    return ratio_values_and_covariance(
        names,
        rates,
        covariance,
        table8_ratio_definitions(target_components, group.parents),
    )


def _role_centroid_shifts(
    spectrum: PublicSpectrum,
    canonical: JointPeakFitResult,
    local: JointPeakFitResult,
    components: Sequence[LineComponent],
) -> tuple[float, float]:
    mean_energy = float(np.mean([component.energy_keV for component in components]))

    def correction(result: JointPeakFitResult) -> float:
        return result.parameter("calibration.offset_keV") + result.parameter(
            "calibration.fractional_gain_stretch"
        ) * (mean_energy - spectrum.calibration_A0)

    def centroid(component: LineComponent, result: JointPeakFitResult) -> float:
        offset = result.parameter("calibration.offset_keV")
        stretch = result.parameter("calibration.fractional_gain_stretch")
        return (
            component.energy_keV
            - offset
            + stretch * spectrum.calibration_A0
        ) / (1.0 + stretch)

    changes = [
        centroid(component, local) - centroid(component, canonical)
        for component in components
    ]
    return correction(local) - correction(canonical), float(np.mean(changes))


def _role_centroid_score(
    spectrum: PublicSpectrum,
    result: JointPeakFitResult,
    spec: JointPeakSpec,
    calibration: CalibrationConstraint,
    resolution: LinearResolution,
    role: str,
) -> tuple[float, float, float]:
    """One-step shared-role centroid score with fitted nuisances held fixed.

    The fit itself keeps one physical calibration and line shape.  This score
    measures remaining antisymmetric target-core morphology without adding a
    selectable shift parameter to the canonical model.
    """

    offset = result.parameter("calibration.offset_keV")
    stretch = result.parameter("calibration.fractional_gain_stretch")
    intercept = result.parameter("resolution.intercept_keV")
    slope = result.parameter("resolution.linear_sigma_slope_keV_per_keV")
    tail_fraction = (
        result.parameter("shape.low_energy_tail_fraction")
        if "shape.low_energy_tail_fraction" in result.parameter_names
        else 0.0
    )
    tail_scale = (
        result.parameter("shape.low_energy_tail_scale_in_sigma")
        if "shape.low_energy_tail_scale_in_sigma" in result.parameter_names
        else 1.0
    )
    derivative = np.zeros_like(result.expected_counts, dtype=np.float64)
    step_keV = 1e-3
    for component in spec.components:
        if component.role != role or component.parent not in {
            item.parent
            for item in spec.components
            if item.role in {"fep", "sep", "dep"}
            and item.role != "contaminant"
        }:
            continue
        mask = np.asarray(result.observation_window_names) == component.window
        channels = result.observation_channel_indices[mask].astype(np.float64)
        low = calibrated_channel_energy(
            spectrum, calibration, channels + 0.5, offset, stretch
        )
        high = calibrated_channel_energy(
            spectrum, calibration, channels + 1.5, offset, stretch
        )
        sigma = resolution_sigma_and_derivatives(
            resolution, component.energy_keV, intercept, slope
        )[0]
        upper = peak_shape_bin_probabilities(
            low,
            high,
            component.energy_keV + step_keV,
            sigma,
            tail_fraction,
            tail_scale,
        )
        lower = peak_shape_bin_probabilities(
            low,
            high,
            component.energy_keV - step_keV,
            sigma,
            tail_fraction,
            tail_scale,
        )
        derivative[mask] += (
            result.line_rate(component.name)
            * spectrum.live_time
            * (upper - lower)
            / (2.0 * step_keV)
        )
    information = float(
        np.sum(derivative**2 / np.maximum(result.expected_counts, 1e-12))
    )
    score = float(
        np.sum(
            (result.observed_counts / np.maximum(result.expected_counts, 1e-12) - 1.0)
            * derivative
        )
    )
    if information <= 0:
        return float("nan"), float("nan"), float("nan")
    shift = score / information
    standard_deviation = 1.0 / sqrt(information)
    return shift, standard_deviation, shift / standard_deviation


def _target_core_maxima(
    spectrum: PublicSpectrum,
    result: JointPeakFitResult,
    spec: JointPeakSpec,
    calibration: CalibrationConstraint,
    resolution: LinearResolution,
    half_width_sigma: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = residual_bin_diagnostics(
        (spectrum,), result, spec, calibration, resolution
    )
    intercept = result.parameter("resolution.intercept_keV")
    slope = result.parameter("resolution.linear_sigma_slope_keV_per_keV")
    maxima: dict[str, float] = {}
    for component in spec.components:
        sigma = resolution_sigma_and_derivatives(
            resolution, component.energy_keV, intercept, slope
        )[0]
        selected = [
            abs(float(row["poisson_residual"]))
            for row in rows
            if row["window"] == component.window
            and abs(
                float(row["fitted_calibrated_center_energy_keV"])
                - component.energy_keV
            )
            <= half_width_sigma * sigma
        ]
        maxima[component.name] = max(selected, default=0.0)
    return maxima, rows


def _parameter_covariance_rows(
    group: str, result: JointPeakFitResult
) -> list[dict[str, Any]]:
    return [
        {
            "group": group,
            "row": row_name,
            "column": column_name,
            "covariance": float(result.covariance[row, column]),
            "semantics": "complete conditional covariance within this independent group fit",
        }
        for row, row_name in enumerate(result.parameter_names)
        for column, column_name in enumerate(result.parameter_names)
    ]


def _line_covariance_rows(
    group: str, result: JointPeakFitResult, live_time: float
) -> list[dict[str, Any]]:
    covariance = result.line_rate_covariance * live_time**2
    return [
        {
            "group": group,
            "row": row_name,
            "column": column_name,
            "covariance_detector_counts_squared": float(covariance[row, column]),
        }
        for row, row_name in enumerate(result.line_names)
        for column, column_name in enumerate(result.line_names)
    ]


def _model_rows(group: _GroupFits) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    canonical = group.canonical_result
    canonical_channels = canonical.observation_channel_indices
    for name, result in group.variants.items():
        resolution = group.variant_resolutions[name]
        rows.append(
            {
                "group": group.name,
                "variant": name,
                "canonical": name == CANONICAL_MODEL,
                "comparison_scope": "same declared group and identical native bins only; no cross-domain ranking",
                "background_model": group.variant_specs[name].windows[0].background_model,
                "tail_model": resolution.tail_model,
                "components_json": json.dumps(result.line_names, separators=(",", ":")),
                "success": result.success,
                "same_native_bins_as_canonical": bool(
                    np.array_equal(result.observation_channel_indices, canonical_channels)
                    and result.observation_window_names
                    == canonical.observation_window_names
                ),
                "poisson_deviance": result.poisson_deviance,
                "delta_deviance_from_group_canonical": (
                    result.poisson_deviance - canonical.poisson_deviance
                ),
                "native_bin_count": result.observed_counts.size,
                "parameter_count": len(result.parameter_names),
                "fisher_rank": result.fisher_rank,
                "fisher_condition": result.fisher_condition,
                "fisher_covariance_valid": result.fisher_covariance_valid,
                "active_bounds_json": json.dumps(result.active_bounds),
                "calibration_offset_keV": result.parameter("calibration.offset_keV"),
                "calibration_fractional_gain_stretch": result.parameter(
                    "calibration.fractional_gain_stretch"
                ),
                "resolution_intercept_keV": result.parameter(
                    "resolution.intercept_keV"
                ),
                "resolution_slope": (
                    result.parameter("resolution.linear_sigma_slope_keV_per_keV")
                    if "resolution.linear_sigma_slope_keV_per_keV"
                    in result.parameter_names
                    else ""
                ),
                "tail_fraction": (
                    result.parameter("shape.low_energy_tail_fraction")
                    if "shape.low_energy_tail_fraction" in result.parameter_names
                    else ""
                ),
                "tail_scale_in_sigma": (
                    result.parameter("shape.low_energy_tail_scale_in_sigma")
                    if "shape.low_energy_tail_scale_in_sigma"
                    in result.parameter_names
                    else ""
                ),
            }
        )
    return rows


def _historical_lookup(
    historical: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    rows = {
        f"{item['parent']}:{item['ratio']}": item
        for item in historical["ratio_rows"]
    }
    if len(rows) != 24:
        raise RuntimeError("historical Table 8 map must contain exactly 24 ratios")
    return rows


def _variant_base_status(
    *,
    result: JointPeakFitResult,
    definition: RatioDefinition,
    core_maxima: Mapping[str, float],
    z_threshold: float,
    residual_threshold: float,
) -> str:
    if not result.success or not result.fisher_covariance_valid:
        return "U"
    positions = {name: index for index, name in enumerate(result.line_names)}
    numerator = positions[definition.numerator]
    denominator = positions[definition.denominator]
    rates = result.line_rates_counts_per_s
    sd = np.sqrt(np.maximum(np.diag(result.line_rate_covariance), 0.0))
    active = set(result.active_bounds)
    if (
        f"line.{definition.denominator}.rate_counts_per_s" in active
        or rates[denominator] <= 0
    ):
        return "U"
    if f"line.{definition.numerator}.rate_counts_per_s" in active:
        return "B"
    z = min(
        rates[numerator] / sd[numerator] if sd[numerator] > 0 else float("inf"),
        rates[denominator] / sd[denominator]
        if sd[denominator] > 0
        else float("inf"),
    )
    if z < z_threshold:
        return "B"
    if max(core_maxima[definition.numerator], core_maxima[definition.denominator]) >= residual_threshold:
        return "R"
    return "Q"


def analyze_table8_local(
    spectrum: PublicSpectrum,
    table_config: Mapping[str, Any],
    historical: Mapping[str, Any],
    calibration: CalibrationConstraint,
    resolution: LinearResolution,
    *,
    minimum_expected_counts: float = 5.0,
    profile_base_nll_consistency_tolerance: float = 0.005,
) -> Table8LocalAnalysis:
    """Run the frozen five-domain Table 8 workflow."""

    input_assertion = _assert_input(spectrum, historical)
    _target_records(table_config)
    workflow = table_config["local_workflow"]
    local_calibration = replace(
        calibration, curvature_mean_keV=None, curvature_sigma_keV=None
    )
    groups = tuple(
        _fit_group(
            spectrum, table_config, group, local_calibration, resolution
        )
        for group in workflow["grouping"]
    )
    historical_rows = _historical_lookup(historical)
    thresholds = workflow["identifiability"]
    z_threshold = float(thresholds["minimum_conditional_yield_z"])
    correlation_threshold = float(thresholds["target_pair_unresolved_correlation"])
    nuisance_correlation_threshold = float(
        thresholds["target_nuisance_unresolved_correlation"]
    )
    residual_threshold = float(thresholds["target_core_poisson_residual_absolute"])
    half_width_sigma = float(thresholds["target_core_half_width_sigma"])
    confidence_level = float(thresholds["profile_confidence_level"])

    candidate_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    centroid_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    parameter_covariance_rows: list[dict[str, Any]] = []
    line_covariance_rows: list[dict[str, Any]] = []
    line_count_rows: list[dict[str, Any]] = []
    fit_bin_jobs: list[Table8LocalFitBinJob] = []
    all_ratio_labels: list[str] = []
    ratio_label_blocks: list[tuple[str, ...]] = []
    ratio_covariance_blocks: list[np.ndarray] = []
    ratio_jacobian_rows: list[dict[str, Any]] = []
    group_diagnostics: list[dict[str, Any]] = []
    basin_stability_rows: list[dict[str, Any]] = []

    for group in groups:
        basin_stability_rows.extend(dict(row) for row in group.basin_stability_rows)
        model_rows.extend(_model_rows(group))
        canonical = group.canonical_result
        canonical_ratios, canonical_lookup = _ratio_lookup(
            canonical, group.canonical_spec, group.parents
        )
        all_ratio_labels.extend(canonical_ratios.labels)
        ratio_label_blocks.append(canonical_ratios.labels)
        ratio_covariance_blocks.append(canonical_ratios.covariance)
        ratio_jacobian_rows.extend(
            {
                "group": group.name,
                "ratio": ratio_name,
                "line": line_name,
                "derivative_seconds_per_count": float(
                    canonical_ratios.jacobian[ratio_index, line_index]
                ),
                "semantics": "full ratio Jacobian with respect to fitted line rates in counts/s",
            }
            for ratio_index, ratio_name in enumerate(canonical_ratios.labels)
            for line_index, line_name in enumerate(canonical.line_names)
        )
        role_ratios = _role_ratio_result(group)
        role_lookup = {
            label: (
                float(role_ratios.values[index]),
                sqrt(max(float(role_ratios.covariance[index, index]), 0.0)),
            )
            for index, label in enumerate(role_ratios.labels)
        }
        canonical_core, _ = _target_core_maxima(
            spectrum,
            canonical,
            group.canonical_spec,
            local_calibration,
            group.canonical_resolution,
            half_width_sigma,
        )
        variant_lookups: dict[str, dict[str, tuple[float, float]]] = {}
        variant_core: dict[str, dict[str, float]] = {}
        eligible_variants: set[str] = set()
        for name, result in group.variants.items():
            lookup = _safe_ratio_lookup(
                result, group.variant_specs[name], group.parents
            )
            variant_lookups[name] = lookup
            core, _ = _target_core_maxima(
                spectrum,
                result,
                group.variant_specs[name],
                local_calibration,
                group.variant_resolutions[name],
                half_width_sigma,
            )
            variant_core[name] = core
            if result.success and result.fisher_covariance_valid:
                eligible_variants.add(name)

        residuals = window_residual_diagnostics(
            canonical,
            (spectrum,),
            group.canonical_spec,
            local_calibration,
            group.canonical_resolution,
            minimum_expected_counts,
        )
        for row in residuals:
            window_rows.append({"group": group.name, **row})

        for role, result in group.role_results.items():
            spec = group.role_specs[role]
            target_components = tuple(
                component
                for component in spec.components
                if component.parent in group.parents
            )
            correction_shift, released_grid_shift = _role_centroid_shifts(
                spectrum, canonical, result, target_components
            )
            score_shift, score_sd, score_z = _role_centroid_score(
                spectrum,
                canonical,
                group.canonical_spec,
                local_calibration,
                group.canonical_resolution,
                role,
            )
            centroid_rows.append(
                {
                    "group": group.name,
                    "role": role,
                    "target_components_json": json.dumps(
                        [component.name for component in target_components],
                        separators=(",", ":"),
                    ),
                    (
                        "role_local_minus_group_canonical_calibration_"
                        "correction_at_mean_target_keV"
                    ): correction_shift,
                    (
                        "role_local_minus_group_canonical_released_grid_"
                        "centroid_keV"
                    ): released_grid_shift,
                    "canonical_fixed_nuisance_one_step_centroid_score_shift_keV": score_shift,
                    "canonical_fixed_nuisance_centroid_score_sd_keV": score_sd,
                    "canonical_fixed_nuisance_centroid_score_z": score_z,
                    "role_local_calibration_offset_keV": result.parameter(
                        "calibration.offset_keV"
                    ),
                    "role_local_fractional_gain_stretch": result.parameter(
                        "calibration.fractional_gain_stretch"
                    ),
                    "effective_sigma_keV": result.parameter(
                        "resolution.intercept_keV"
                    ),
                    "success": result.success,
                    "fisher_rank": result.fisher_rank,
                    "fisher_dimension": len(result.parameter_names),
                    "fisher_condition": result.fisher_condition,
                    "active_bounds_json": json.dumps(result.active_bounds),
                    "semantics": (
                        "role-local calibration-correction and released-grid "
                        "centroid differences are opposite-coordinate no-tail/"
                        "free-width sensitivities; one-step score measures "
                        "canonical antisymmetric morphology with fitted "
                        "nuisances fixed; none is fitted or absorbed by "
                        "canonical calibration"
                    ),
                }
            )

        definitions = table8_ratio_definitions(
            group.canonical_spec.components, group.parents
        )
        definitions_by_name = {definition.name: definition for definition in definitions}
        line_positions = {
            name: index for index, name in enumerate(canonical.line_names)
        }
        line_sd = np.sqrt(
            np.maximum(np.diag(canonical.line_rate_covariance), 0.0)
        )
        active = set(canonical.active_bounds)

        unresolved_by_component: dict[str, list[str]] = {}
        target_components = tuple(
            component
            for component in group.canonical_spec.components
            if component.parent in group.parents
        )
        neighbors = tuple(
            component
            for component in group.canonical_spec.components
            if component.parent not in group.parents
        )
        for target in target_components:
            for neighbor in neighbors:
                if target.window != neighbor.window:
                    continue
                correlation = _line_correlation(
                    canonical, target.name, neighbor.name
                )
                correlation_rows.append(
                    {
                        "group": group.name,
                        "variant": CANONICAL_MODEL,
                        "left": target.name,
                        "right": neighbor.name,
                        "relationship": "target-nuisance",
                        "correlation": correlation,
                        "absolute_correlation_threshold": nuisance_correlation_threshold,
                    }
                )
                if abs(correlation) >= nuisance_correlation_threshold:
                    unresolved_by_component.setdefault(target.name, []).append(
                        f"{neighbor.name}:{correlation:.6g}"
                    )
        if group.name == "cluster_7631_7724":
            for role in ("dep", "sep", "fep"):
                left = f"t8_7631_180_{role}"
                right = f"t8_7645_580_{role}"
                correlation = _line_correlation(canonical, left, right)
                correlation_rows.append(
                    {
                        "group": group.name,
                        "variant": CANONICAL_MODEL,
                        "left": left,
                        "right": right,
                        "relationship": "14.400-keV target-target pair",
                        "correlation": correlation,
                        "absolute_correlation_threshold": correlation_threshold,
                    }
                )
                if abs(correlation) >= correlation_threshold:
                    reason = f"{left}/{right}:{correlation:.6g}"
                    unresolved_by_component.setdefault(left, []).append(reason)
                    unresolved_by_component.setdefault(right, []).append(reason)

        for name in ("al27_6711_alternative", "ge70_6707_alternative"):
            if name not in group.variants:
                continue
            result = group.variants[name]
            if not result.success or not result.fisher_covariance_valid:
                continue
            target = "t8_7724_034_dep"
            neighbor = (
                "al27_6710_7_fep"
                if name == "al27_6711_alternative"
                else "ge70_6707_45_fep"
            )
            correlation = _line_correlation(result, target, neighbor)
            correlation_rows.append(
                {
                    "group": group.name,
                    "variant": name,
                    "left": target,
                    "right": neighbor,
                    "relationship": "equal-status target-nuisance sensitivity",
                    "correlation": correlation,
                    "absolute_correlation_threshold": nuisance_correlation_threshold,
                }
            )
            if abs(correlation) >= nuisance_correlation_threshold:
                unresolved_by_component.setdefault(target, []).append(
                    f"{name}:{neighbor}:{correlation:.6g}"
                )

        for definition in definitions:
            key = definition.name
            value, fisher_sd = canonical_lookup[key]
            variant_values = {
                name: lookup[key][0]
                for name, lookup in variant_lookups.items()
                if name != CANONICAL_MODEL and name in eligible_variants
            }
            variant_values["role_local_free_width_no_tail"] = role_lookup[key][0]
            stability = ratio_stability_summary(value, fisher_sd, variant_values)
            range_values = {CANONICAL_MODEL: value, **variant_values}
            range_min_driver = min(range_values, key=range_values.get)
            range_max_driver = max(range_values, key=range_values.get)
            numerator_index = line_positions[definition.numerator]
            denominator_index = line_positions[definition.denominator]
            numerator_z = (
                canonical.line_rates_counts_per_s[numerator_index]
                / line_sd[numerator_index]
                if line_sd[numerator_index] > 0
                else float("inf")
            )
            denominator_z = (
                canonical.line_rates_counts_per_s[denominator_index]
                / line_sd[denominator_index]
                if line_sd[denominator_index] > 0
                else float("inf")
            )
            numerator_bound = (
                f"line.{definition.numerator}.rate_counts_per_s" in active
            )
            denominator_bound = (
                f"line.{definition.denominator}.rate_counts_per_s" in active
            )
            profile: ProfileInterval | None = None
            if (
                min(numerator_z, denominator_z) < z_threshold
                or numerator_bound
                or denominator_bound
            ):
                profile = profile_ratio_interval(
                    (spectrum,),
                    group.canonical_spec,
                    local_calibration,
                    group.canonical_resolution,
                    canonical,
                    definition,
                    confidence_level=confidence_level,
                    base_nll_consistency_tolerance=(
                        profile_base_nll_consistency_tolerance
                    ),
                )
            if profile is None:
                profile_rows.append(
                    {
                        "ratio": key,
                        "triggered": False,
                        "kind": "regular_interior_fisher",
                        "estimate": value,
                        "confidence_level": confidence_level,
                        "lower": "",
                        "upper": "",
                        "message": "both target yields interior at >=3 conditional Fisher sigma",
                        "diagnostics_json": "{}",
                    }
                )
                profile_kind = "regular_interior_fisher"
                profile_excludes_zero = True
                interval_lower: float | str = value - 1.96 * fisher_sd
                interval_upper: float | str = value + 1.96 * fisher_sd
            else:
                profile_record = asdict(profile)
                profile_rows.append(
                    {
                        "ratio": profile.ratio,
                        "triggered": True,
                        "kind": profile.kind,
                        "estimate": profile.estimate,
                        "confidence_level": profile.confidence_level,
                        "lower": profile.lower,
                        "upper": profile.upper,
                        "message": profile.message,
                        "diagnostics_json": json.dumps(
                            {
                                key: value
                                for key, value in profile_record.items()
                                if key
                                not in {
                                    "ratio",
                                    "kind",
                                    "estimate",
                                    "confidence_level",
                                    "lower",
                                    "upper",
                                    "message",
                                }
                            },
                            separators=(",", ":"),
                        ),
                    }
                )
                profile_kind = profile.kind
                profile_excludes_zero = bool(
                    profile.kind == "two_sided"
                    and isfinite(profile.lower)
                    and profile.lower > 0
                )
                interval_lower = profile.lower
                interval_upper = profile.upper

            no_tail_name = "no_tail_quadratic"
            no_tail_value, no_tail_sd = variant_lookups[no_tail_name][key]
            no_tail_definition = definitions_by_name[key]
            canonical_base_status = _variant_base_status(
                result=canonical,
                definition=definition,
                core_maxima=canonical_core,
                z_threshold=z_threshold,
                residual_threshold=residual_threshold,
            )
            no_tail_base_status = _variant_base_status(
                result=group.variants[no_tail_name],
                definition=no_tail_definition,
                core_maxima=variant_core[no_tail_name],
                z_threshold=z_threshold,
                residual_threshold=residual_threshold,
            )
            tail_status_differs = canonical_base_status != no_tail_base_status
            unresolved_reasons = list(
                unresolved_by_component.get(definition.numerator, ())
            ) + list(unresolved_by_component.get(definition.denominator, ()))
            parent, label = key.split(":", 1)
            mandatory_model_sensitive = parent == "7724.034" and "dep" in label
            equal_status_reallocation = max(
                (
                    abs(variant_lookups[name][key][0] - value)
                    for name in (
                        "al27_6711_alternative",
                        "ge70_6707_alternative",
                    )
                    if name in variant_lookups and name in eligible_variants
                ),
                default=0.0,
            )
            if mandatory_model_sensitive and equal_status_reallocation > stability["threshold"]:
                unresolved_reasons.append(
                    "equal-status Al-27/Ge-70 reallocation exceeds stability threshold"
                )
            target_core = max(
                canonical_core[definition.numerator],
                canonical_core[definition.denominator],
            )
            status, reason = classify_table8_ratio(
                canonical_available=canonical.success,
                profile_kind=profile_kind,
                profile_excludes_zero=profile_excludes_zero,
                numerator_z=numerator_z,
                denominator_z=denominator_z,
                numerator_bound=numerator_bound,
                denominator_bound=denominator_bound,
                covariance_valid=canonical.fisher_covariance_valid,
                unresolved=bool(unresolved_reasons),
                target_core_residual=target_core >= residual_threshold,
                model_sensitive=not stability["passes"],
                mandatory_model_sensitive=mandatory_model_sensitive,
                tail_status_differs=tail_status_differs,
                weak_z_threshold=z_threshold,
            )
            historical_row = historical_rows[key]
            recommendation_value: float | str = value if status in {"Q", "V"} else ""
            recommendation_sd: float | str = fisher_sd if status in {"Q", "V"} else ""
            recommendation_upper: float | str = (
                interval_upper if status == "B" else ""
            )
            floor_annotation = (
                f"{stability['driver']}: {stability['maximum']:.6g} = "
                f"{stability['sigma_units']:.3g} SD = "
                f"{100.0 * stability['relative']:.3g}%"
                if stability["five_percent_floor_only"]
                else ""
            )
            role_local_signed_movement = role_lookup[key][0] - value
            role_local_absolute_movement = abs(role_local_signed_movement)
            role_local_movement_in_sd = (
                role_local_absolute_movement / fisher_sd
                if fisher_sd > 0
                else float("inf")
            )
            role_local_relative_movement = (
                role_local_absolute_movement / abs(value)
                if value
                else float("inf")
            )
            active_bound_annotation = ""
            if canonical.active_bounds:
                active_bound_annotation = (
                    "active canonical nuisance bounds: "
                    + ", ".join(canonical.active_bounds)
                    + "; role-local free-width/no-tail movement "
                    + f"{role_local_absolute_movement:.6g} = "
                    + f"{role_local_movement_in_sd:.3g} SD = "
                    + f"{100.0 * role_local_relative_movement:.3g}%, "
                    + (
                        "within frozen stability threshold"
                        if role_local_absolute_movement <= stability["threshold"]
                        else "above frozen stability threshold"
                    )
                )
            legacy_9719 = (
                "The surviving independent-Gaussian check put FEP/DEP and "
                "SEP/DEP 16.9% and 21.6% above the paper; N_window/(7 sigma) "
                "density semantics plus stateful maximum-search/grouping make "
                "that legacy pathway non-authoritative. Local full-line "
                "semantics resolve FEP/DEP but leave the SEP/DEP difference "
                "as an unexplained legacy-provenance residual."
                if parent == "9718.790"
                else ""
            )
            legacy_9719_resolution = ""
            if parent == "9718.790" and label == "fep/dep":
                legacy_9719_resolution = (
                    f"resolved for this ratio: local full-line {value:.6g} differs "
                    f"from paper {historical_row['paper'][0]:.6g} by "
                    f"{abs(value-historical_row['paper'][0])/fisher_sd:.3g} local Fisher SD"
                )
            elif parent == "9718.790" and label == "sep/dep":
                legacy_9719_resolution = (
                    f"not fully resolved: local full-line {value:.6g} remains "
                    f"{abs(value-historical_row['paper'][0])/fisher_sd:.3g} local "
                    f"Fisher SD ({100*abs(value/historical_row['paper'][0]-1):.3g}%) "
                    f"above paper {historical_row['paper'][0]:.6g}; absent the "
                    "paper-generating fit/ROOT products, the remainder is an "
                    "unexplained legacy-provenance residual"
                )
            elif parent == "9718.790":
                legacy_9719_resolution = (
                    "not one of the audited 17-22% DEP-denominator discrepancy pair"
                )
            candidate_rows.append(
                {
                    "parent_energy_keV": parent,
                    "paper_label": next(
                        item["paper_label"]
                        for item in historical["authoritative_parents"]
                        if item["energy_keV"] == parent
                    ),
                    "ratio": label,
                    "paper_ratio": historical_row["paper"][0],
                    "paper_uncertainty": historical_row["paper"][1],
                    "current_ordered_full_line_ratio": historical_row[
                        "ordered_full_line"
                    ][0],
                    "current_ordered_full_line_covariance_sd": historical_row[
                        "ordered_full_line"
                    ][1],
                    "phase2_ratio": historical_row["phase2"][0],
                    "phase2_fisher_sd": historical_row["phase2"][1],
                    "local_ratio": value,
                    "local_conditional_fisher_sd": fisher_sd,
                    "interval_kind": profile_kind,
                    "interval_confidence_level": confidence_level,
                    "interval_lower": interval_lower,
                    "interval_upper": interval_upper,
                    "numerator_component": definition.numerator,
                    "denominator_component": definition.denominator,
                    "numerator_conditional_z": numerator_z,
                    "denominator_conditional_z": denominator_z,
                    "numerator_active_bound": numerator_bound,
                    "denominator_active_bound": denominator_bound,
                    "target_core_max_absolute_poisson_residual": target_core,
                    "no_tail_ratio": no_tail_value,
                    "no_tail_conditional_fisher_sd": no_tail_sd,
                    "no_tail_absolute_movement": abs(no_tail_value - value),
                    "canonical_base_status": canonical_base_status,
                    "no_tail_base_status": no_tail_base_status,
                    "tail_status_differs": tail_status_differs,
                    "role_local_ratio": role_lookup[key][0],
                    "role_local_conditional_fisher_sd": role_lookup[key][1],
                    "role_local_signed_movement": role_local_signed_movement,
                    "role_local_absolute_movement": role_local_absolute_movement,
                    "role_local_movement_in_canonical_fisher_sd": role_local_movement_in_sd,
                    "role_local_relative_movement": role_local_relative_movement,
                    "canonical_active_bounds_json": json.dumps(
                        canonical.active_bounds
                    ),
                    "canonical_active_bound_annotation": active_bound_annotation,
                    "maximum_declared_absolute_movement": stability["maximum"],
                    "maximum_declared_signed_movement": stability["signed"],
                    "maximum_declared_movement_driver": stability["driver"],
                    "maximum_declared_movement_in_fisher_sd": stability[
                        "sigma_units"
                    ],
                    "maximum_declared_relative_movement": stability["relative"],
                    "declared_variant_min_ratio": range_values[range_min_driver],
                    "declared_variant_min_driver": range_min_driver,
                    "declared_variant_max_ratio": range_values[range_max_driver],
                    "declared_variant_max_driver": range_max_driver,
                    "stability_threshold": stability["threshold"],
                    "stability_passes": stability["passes"],
                    "five_percent_floor_only": stability[
                        "five_percent_floor_only"
                    ],
                    "five_percent_floor_annotation": floor_annotation,
                    "equal_status_al_ge_absolute_reallocation": equal_status_reallocation,
                    "unresolved_reasons_json": json.dumps(unresolved_reasons),
                    "status": status,
                    "status_label": STATUS_LABELS[status],
                    "status_reason": reason,
                    "recommended_ratio": recommendation_value,
                    "recommended_conditional_fisher_sd": recommendation_sd,
                    "recommended_upper_limit": recommendation_upper,
                    "legacy_9718_discrepancy_explanation": legacy_9719,
                    "legacy_9718_discrepancy_resolution": legacy_9719_resolution,
                    "result_semantics": workflow["result_semantics"],
                }
            )
            role_rows.append(
                {
                    "group": group.name,
                    "ratio": key,
                    "canonical_ratio": value,
                    "canonical_fisher_sd": fisher_sd,
                    "role_local_ratio": role_lookup[key][0],
                    "role_local_fisher_sd": role_lookup[key][1],
                    "signed_movement": role_lookup[key][0] - value,
                    "absolute_movement": abs(role_lookup[key][0] - value),
                    "movement_in_canonical_fisher_sd": (
                        abs(role_lookup[key][0] - value) / fisher_sd
                        if fisher_sd > 0
                        else float("inf")
                    ),
                    "semantics": "independent role-window free-width/no-tail diagnostic; cross-role covariance exactly zero",
                }
            )

        parameter_covariance_rows.extend(
            _parameter_covariance_rows(group.name, canonical)
        )
        component_lookup = {
            component.name: component for component in group.canonical_spec.components
        }
        canonical_line_sd = np.sqrt(
            np.maximum(np.diag(canonical.line_rate_covariance), 0.0)
        )
        line_count_rows.extend(
            {
                "group": group.name,
                "component": name,
                "parent_energy_keV": component_lookup[name].parent,
                "role": component_lookup[name].role,
                "line_energy_keV": component_lookup[name].energy_keV,
                "full_normalized_line_rate_counts_per_s": float(
                    canonical.line_rates_counts_per_s[index]
                ),
                "full_normalized_line_rate_conditional_fisher_sd_counts_per_s": float(
                    canonical_line_sd[index]
                ),
                "full_normalized_line_counts": float(
                    canonical.line_rates_counts_per_s[index] * spectrum.live_time
                ),
                "full_normalized_line_count_conditional_fisher_sd": float(
                    canonical_line_sd[index] * spectrum.live_time
                ),
                "conditional_z": (
                    float(canonical.line_rates_counts_per_s[index] / canonical_line_sd[index])
                    if canonical_line_sd[index] > 0
                    else float("inf")
                ),
                "yield_active_bound": (
                    f"line.{name}.rate_counts_per_s" in canonical.active_bounds
                ),
                "normalization": "unit-integral fitted Gaussian-plus-declared-tail component over the complete real line",
            }
            for index, name in enumerate(canonical.line_names)
        )
        line_covariance_rows.extend(
            _line_covariance_rows(group.name, canonical, spectrum.live_time)
        )
        fit_bin_jobs.append(
            Table8LocalFitBinJob(
                f"table8_local_{group.name}_fit_bins.csv.gz",
                canonical,
                group.canonical_spec,
                local_calibration,
                group.canonical_resolution,
            )
        )
        group_diagnostics.append(
            {
                "group": group.name,
                "parents_keV": list(group.parents),
                "poisson_deviance": canonical.poisson_deviance,
                "native_bin_count": canonical.observed_counts.size,
                "parameter_count": len(canonical.parameter_names),
                "descriptive_degrees_of_freedom": canonical.degrees_of_freedom,
                "fisher_rank": canonical.fisher_rank,
                "fisher_condition": canonical.fisher_condition,
                "fisher_covariance_valid": canonical.fisher_covariance_valid,
                "active_bounds": list(canonical.active_bounds),
                "optimizer_message": canonical.message,
            }
        )

    assembled_labels, ratio_covariance = assemble_independent_ratio_covariance(
        ratio_label_blocks, ratio_covariance_blocks
    )
    if tuple(all_ratio_labels) != assembled_labels:
        raise RuntimeError("Table 8 ratio ordering changed during covariance assembly")
    ratio_covariance_rows = [
        {
            "row": row_name,
            "column": column_name,
            "covariance": float(ratio_covariance[row, column]),
            "semantics": (
                "complete J Sigma J-transpose within a group; exactly zero between disjoint independently fitted groups"
            ),
        }
        for row, row_name in enumerate(all_ratio_labels)
        for column, column_name in enumerate(all_ratio_labels)
    ]
    status_counts = {
        code: sum(row["status"] == code for row in candidate_rows)
        for code in STATUS_LABELS
    }
    role_local_means = {
        role: float(
            np.mean(
                [
                    row[
                        "role_local_minus_group_canonical_released_grid_centroid_keV"
                    ]
                    for row in centroid_rows
                    if row["role"] == role
                ]
            )
        )
        for role in ("fep", "sep", "dep")
    }
    role_correction_means = {
        role: float(
            np.mean(
                [
                    row[
                        "role_local_minus_group_canonical_calibration_correction_at_mean_target_keV"
                    ]
                    for row in centroid_rows
                    if row["role"] == role
                ]
            )
        )
        for role in ("fep", "sep", "dep")
    }
    role_score_means = {
        role: float(
            np.mean(
                [
                    row[
                        "canonical_fixed_nuisance_one_step_centroid_score_shift_keV"
                    ]
                    for row in centroid_rows
                    if row["role"] == role
                ]
            )
        )
        for role in ("fep", "sep", "dep")
    }
    products = {
        "table8_local_candidate.csv": candidate_rows,
        "table8_local_model_comparison.csv": model_rows,
        "table8_local_role_sensitivity.csv": role_rows,
        "table8_local_role_centroid_shifts.csv": centroid_rows,
        "table8_local_window_diagnostics.csv": window_rows,
        "table8_local_profile_intervals.csv": profile_rows,
        "table8_local_line_correlations.csv": correlation_rows,
        "table8_local_full_parameter_covariance.csv": parameter_covariance_rows,
        "table8_local_full_line_count_covariance.csv": line_covariance_rows,
        "table8_local_full_line_counts.csv": line_count_rows,
        "table8_local_ratio_covariance.csv": ratio_covariance_rows,
        "table8_local_ratio_jacobian.csv": ratio_jacobian_rows,
        "table8_local_basin_stability.csv": basin_stability_rows,
    }
    diagnostics = {
        "result_semantics": workflow["result_semantics"],
        "input_runtime_assertion": input_assertion,
        "historical_record": "config/table8_historical_reconstruction.json",
        "canonical_model": CANONICAL_MODEL,
        "group_count": len(groups),
        "ratio_count": len(candidate_rows),
        "group_diagnostics": group_diagnostics,
        "status_counts": status_counts,
        "role_local_minus_group_canonical_calibration_correction_mean_keV": role_correction_means,
        "role_local_minus_group_canonical_released_grid_centroid_mean_keV": role_local_means,
        "canonical_fixed_nuisance_one_step_centroid_score_shift_mean_keV": role_score_means,
        "profile_trigger_count": sum(row["triggered"] for row in profile_rows),
        "five_percent_floor_only_count": sum(
            row["five_percent_floor_only"] for row in candidate_rows
        ),
        "canonical_basin_stability": {
            "restart_count": len(basin_stability_rows),
            "all_passed": bool(
                basin_stability_rows
                and all(row["passed"] for row in basin_stability_rows)
            ),
            "group": workflow["canonical_basin_stability_check"]["group"],
        },
        "bootstrap": "not run; fewer than 200 replicas cannot change a local reporting action",
        "cross_domain_covariance": "exactly zero for disjoint Poisson bins and independently fitted per-group calibration/shape nuisances",
        "simulation_compared": False,
        "manuscript_replacement_automatic": False,
    }
    return Table8LocalAnalysis(
        products, diagnostics, tuple(fit_bin_jobs)
    )
