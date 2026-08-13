"""Integrity checks for maintained phase-2 configuration contracts."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.reanalyze_paper_peak_statistics import (
    _assert_table8_canonical_fit_identity,
    _nonstandard_penalized_information_criteria,
    _table8_guarded_restart_settings,
    _write_manifest,
)
from src.public_data.peak_likelihood import (
    FitWindow,
    JointPeakSpec,
    LineComponent,
    LinearResolution,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "paper_peak_statistics.json"
CONFIG = json.loads(CONFIG_PATH.read_text())
HISTORICAL = json.loads(
    (ROOT / "config" / "table3_historical_reconstruction.json").read_text()
)


def test_penalized_information_criteria_keep_nonstandard_labels():
    fields = _nonstandard_penalized_information_criteria(10.0, 3, 100)
    assert fields["nonstandard_penalized_objective_aic"] == "26"
    assert fields["nonstandard_penalized_objective_bic"] == format(
        20.0 + 3.0 * np.log(100), ".12g"
    )
    assert "nonstandard" in fields["information_criterion_semantics"]
    assert "not used for model promotion" in fields[
        "information_criterion_semantics"
    ]
    assert "akaike_information_criterion" not in fields
    assert "bayesian_information_criterion" not in fields


def test_table8_guarded_restart_rule_is_explicit_and_validated():
    variants = CONFIG["table8"]["component_audit"]["variants"]
    with_restart = [
        variant for variant in variants if "guarded_basin_restart" in variant
    ]
    assert [variant["name"] for variant in with_restart] == [
        "steel_catalog_sensitivity"
    ]
    rule = with_restart[0]["guarded_basin_restart"]
    assert _table8_guarded_restart_settings(
        "renamed_steel_variant", {"renamed_steel_variant": rule}
    ) == (True, rule["initial_source"])
    with pytest.raises(ValueError, match="both allowed=true"):
        _table8_guarded_restart_settings(
            "broken", {"broken": {"allowed": True}}
        )


def test_table8_downstream_model_identity_fails_on_tail_mismatch():
    spec = JointPeakSpec(
        "identity-check",
        (FitWindow("window", 90.0, 110.0, "quadratic"),),
        (LineComponent("line", 100.0, "window"),),
    )
    no_tail = LinearResolution(
        1.0, 1.0e-4, form="sqrt", tail_model="none"
    )
    result = SimpleNamespace(
        line_names=("line",),
        parameter_names=(
            "resolution.variance_slope_keV",
            "background.window.middle_counts_per_s_per_keV",
        ),
    )
    _assert_table8_canonical_fit_identity(result, spec, no_tail)
    with pytest.raises(RuntimeError, match="tail-model identity"):
        _assert_table8_canonical_fit_identity(
            result, spec, replace(no_tail, tail_model="constant")
        )


def test_manifest_preserves_exploratory_scope_and_reporting(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        repo_root=ROOT,
        config=CONFIG,
        config_path=CONFIG_PATH,
        database_sha256="manufactured-database-hash",
        input_records={},
        diagnostics_by_table={},
        output_hashes={},
        bootstrap_replicates=1,
        code_revision={"commit": "manufactured", "working_tree_dirty": False},
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["reporting_configuration"] == CONFIG["reporting"]
    assert manifest["configuration"]["schema_version"] == 11
    assert manifest["result_semantics"] == CONFIG["result_semantics"]
    assert CONFIG["result_semantics"].startswith("phase-2 new exploratory")
    assert CONFIG["reporting"]["candidate_status"].startswith("exploratory")
    assert manifest["scientific_scope"][
        "candidate_numbers_approved_for_manuscript"
    ] is False


def test_historical_table3_record_freezes_inputs_bug_and_caption_conflict():
    assert HISTORICAL["record_kind"] == "paper-exact historical reconstruction"
    assert HISTORICAL["caption_input_conflict"] == (
        "The released historical script selects Cycle493_RD_low_gain. The "
        "public bundle contains no Cycle498 baseline spectrum matching the "
        "caption. Exact numerical replay therefore does not validate the "
        "caption's Cycle498 claim."
    )
    assert HISTORICAL["historical_revision"]["commit"] == (
        "cdabda2dcafce176bad62ea3ca19a9f085b2986b"
    )
    assert HISTORICAL["bug_provenance"]["introduced_commit"] == (
        "5cb9e0afd362e7d3703456531594b0e1424555e3"
    )
    files = HISTORICAL["selection_and_order"]["files"]
    assert [item["file_id"] for item in files] == [1716, 1765, 334, 1676]
    assert all(len(item["spectrum_sha256"]) == 64 for item in files)
    assert HISTORICAL["selection_and_order"]["total_live_time_s"] == 260013.89
    assert HISTORICAL["selection_and_order"]["summed_detector_counts"] == 5845047
    assert HISTORICAL["selection_and_order"]["maximum_summed_channel_count"] == 156917
    assert HISTORICAL["selection_and_order"][
        "summed_counts_sha256_int64_little_endian"
    ] == "ca030c709a3ece435df510a6e1e62b4ffe866e43fc9c181f948b50e6ae2a6d24"
    windows = HISTORICAL["historical_local_windows"]
    assert len(windows) == 34
    assert sum(
        item["last_channel_1_based"] - item["first_channel_1_based"] + 1
        for item in windows
    ) == 515
    by_energy = {
        item["energy_keV"]: item for item in HISTORICAL["exact_legacy_ratios"]
    }
    assert len(by_energy) == 35
    assert round(by_energy[238.6]["ratio"], 3) == 0.346
    assert round(by_energy[7367.9]["ratio"], 3) == 0.003
    assert by_energy[478.0]["paper_display"] == "--"
    assert by_energy[558.5]["repaired_full_line_ratio"] == 1.0
    assert by_energy[558.5]["repaired_full_line_conditional_uncertainty"] == 0.0


def test_table3_sum_first_model_and_drift_decisions_are_frozen():
    workflow = CONFIG["table3"]["sum_first_local_workflow"]
    assert [item["name"] for item in workflow["model_variants"]] == [
        "no_tail_affine",
        "no_tail_quadratic",
        "common_left_tail_quadratic",
    ]
    assert workflow["drift_diagnostic"]["anchors_keV"] == [
        351.932,
        558.456,
        1293.640,
        1764.491,
        2614.511,
    ]
    assert workflow["drift_diagnostic"][
        "mixture_broadening_threshold_fraction"
    ] == 0.05
    assert "25 keV" in workflow["weak_branch_rule"]
    assert workflow["row_thresholds"] == {
        "supported_conditional_z_minimum": 3.0,
        "zero_compatible_conditional_z_maximum": 1.96,
        "material_absolute_poisson_residual": 4.0,
        "material_ratio_shift_conditional_sigma": 1.0,
    }
    isolated = {
        item["name"]: item["energy_keV"]
        for item in workflow["isolated_candidate_components"]
    }
    assert isolated == {
        "rd_609_3": 609.321,
        "rd_707_4": 707.419,
        "rd_1364_3": 1364.339,
        "rd_1377_7": 1377.669,
    }


def test_table8_candidates_are_declared_inside_windows_and_not_auto_promoted():
    table = CONFIG["table8"]
    windows = {
        item["name"]: (item["low_keV"], item["high_keV"])
        for item in table["windows"]
    }
    candidates = table["component_audit"]["candidates"]
    assert len({item["name"] for item in candidates}) == len(candidates)
    assert table["component_audit"]["authoritative_nuclear_source"][
        "url"
    ].startswith("https://www.nndc.bnl.gov/")
    for candidate in candidates:
        low, high = windows[candidate["window"]]
        assert low < candidate["energy_keV"] < high
        assert candidate["source_plausibility"]
        assert candidate["classification"]

    variants = {
        item["name"]: set(item["candidate_components"])
        for item in table["component_audit"]["variants"]
    }
    canonical = variants[table["canonical_component_variant"]]
    assert table["canonical_model_variant"] == table["canonical_component_variant"]
    calibration = table["calibration_constraint"]
    low = min(window["low_keV"] for window in table["windows"])
    high = max(window["high_keV"] for window in table["windows"])
    assert calibration["curvature_mean_keV"] == 0.0
    assert calibration["curvature_sigma_keV"] == 0.893146
    assert calibration["curvature_bounds_keV"] == [-2.679438, 2.679438]
    assert calibration["curvature_pivot_keV"] == (low + high) / 2.0
    assert calibration["curvature_scale_keV"] == (high - low) / 2.0
    target_components = [
        item for item in table["components"] if item["role"] != "contaminant"
    ]
    for parent_text in table["target_parents_keV"]:
        parent = float(parent_text)
        by_role = {
            item["role"]: item["energy_keV"]
            for item in target_components
            if float(item["parent"]) == parent
        }
        assert by_role == pytest.approx(
            {"fep": parent, "sep": parent - 511.0, "dep": parent - 1022.0}
        )
    assert canonical == {
        "fe54_6268_9_fep",
        "cu63_6616_0_dep",
        "cu63_7127_0_sep",
        "cu63_7638_0_fep",
    }
    ambiguous = {
        item["name"]: item
        for item in candidates
        if item["name"] in {"al27_6710_7_fep", "ge70_6707_45_fep"}
    }
    assert all(
        item["classification"]
        == "plausible_ambiguous_sensitivity_not_canonical"
        for item in ambiguous.values()
    )
    assert canonical.isdisjoint(ambiguous)
    al_components = variants["al27_6711_alternative"]
    ge_components = variants["target_plus_fe_cu_ge70"]
    assert al_components - ge_components == {"al27_6710_7_fep"}
    assert ge_components - al_components == {"ge70_6707_45_fep"}


def test_table8_phase2_window_change_is_common_and_not_directly_comparable():
    change = CONFIG["table8"]["phase2_window_change"]
    assert change["window"] == "t8_6809_sep"
    assert change["phase1_low_keV"] == 6272.61
    assert change["phase2_low_keV"] == 6255.0
    assert "every phase-2 Table 8 variant" in change["reason"]
    assert "not direct likelihood comparisons" in change["comparison_warning"]


def test_table3_reference_audit_keeps_unsupported_candidates_unfitted():
    audit = CONFIG["table3"]["reference_window_component_audit"]
    candidates = {
        item["name"]: item
        for item in audit["candidates"]
    }
    expected = {
        "co59_555_972",
        "in115_556_845",
        "w186_557_16",
        "ge74_595_85_inelastic",
        "annihilation_511_local_continuum",
        "unidentified_escape_into_540_578_roi",
        "unidentified_true_coincidence_sum_in_roi",
    }
    assert expected <= candidates.keys()
    assert all(
        candidates[name]["fit_treatment"].startswith(
            ("not fitted", "not promoted")
        )
        for name in expected
    )
    assert audit["canonical_variant"] == "capgam_energy_558_456"
    assert audit["canonical_component"]["authoritative_energy_keV"] == 558.456


def test_table3_residual_repair_excludes_nonidentifiable_windows():
    table = CONFIG["table3"]
    components = {item["name"]: item for item in table["components"]}
    windows = {item["name"] for item in table["windows"]}
    excluded = {item["name"]: item for item in table["excluded_components"]}
    accepted = table["residual_model_repair"]["accepted_components"]
    assert [item["name"] for item in accepted] == [
        "ac228_338_320_nuisance"
    ]
    assert components["ac228_338_320_nuisance"]["role"] == "contaminant"
    assert components["ac228_338_320_nuisance"]["window"] == "rd_352"
    assert {"rd_609", "rd_707_725", "rd_1364_1400"}.isdisjoint(windows)
    unavailable = {
        "rd_609_3",
        "rd_707_4",
        "rd_725_0",
        "rd_1364_3",
        "rd_1377_7",
        "rd_1399_6",
    }
    assert unavailable.issubset(excluded)
    assert all(
        excluded[name]["interval_kind"].startswith("unavailable_")
        for name in unavailable
    )


def test_table3_model_uses_authoritative_energies_with_paper_labels():
    components = {
        item["name"]: item for item in CONFIG["table3"]["components"]
    }
    expected = {
        "rd_238_6": (238.632, 238.6),
        "rd_242_0": (241.995, 242.0),
        "rd_295_2": (295.224, 295.2),
        "rd_351_9": (351.932, 351.9),
        "rd_558_5_reference": (558.456, 558.5),
        "rd_651_3": (651.256, 651.3),
        "rd_768_4": (768.36, 768.4),
        "rd_805_9": (805.887, 805.9),
        "rd_1120_3": (1120.294, 1120.3),
        "rd_1209_7": (1209.713, 1209.7),
        "rd_1238_1": (1238.122, 1238.1),
        "rd_1281_0": (1280.976, 1281.0),
        "rd_1293_6": (1293.64, 1293.6),
        "rd_1764_5": (1764.491, 1764.5),
        "rd_2204_2": (2204.1, 2204.2),
        "rd_2223_0": (2223.245, 2223.0),
        "rd_2455_8": (2456.0, 2455.8),
        "rd_2614_533": (2614.511, 2614.533),
    }
    assert {
        name: (components[name]["energy_keV"], components[name]["paper_energy_keV"])
        for name in expected
    } == expected
