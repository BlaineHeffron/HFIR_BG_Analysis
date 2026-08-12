"""Integrity checks for frozen phase-2 component and historical records."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.public_data import peak_likelihood as likelihood
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


def test_optimizer_stationarity_gate_is_frozen():
    config = json.loads(
        (ROOT / "config" / "paper_peak_statistics.json").read_text()
    )
    assert config["reporting"][
        "optimizer_scaled_projected_gradient_tolerance"
    ] == 0.001
    assert config["reporting"]["optimizer_scoring_max_iterations"] == 8
    assert config["reporting"][
        "optimizer_polish_max_scaled_displacement_inf"
    ] == 10.0
    assert config["reporting"][
        "optimizer_polish_max_stable_nll_decrease"
    ] == 1.0
    assert "stable per-bin Poisson NLL" in config["likelihood"]["optimization"]


def test_profile_inner_solver_gates_are_frozen():
    config = json.loads(
        (ROOT / "config" / "paper_peak_statistics.json").read_text()
    )
    reporting = config["reporting"]
    assert config["schema_version"] == 8
    assert reporting["profile_base_nll_consistency_tolerance"] == 0.005
    assert reporting["profile_scaled_kkt_tolerance"] == 0.001
    assert reporting[
        "profile_linear_constraint_relative_tolerance"
    ] == 1e-8
    assert reporting["profile_slsqp_max_chains"] == 3
    assert reporting["profile_slsqp_max_iterations_per_chain"] == 3000
    assert reporting["profile_slsqp_ftol"] == 1e-12
    assert reporting["profile_scoring_max_iterations_per_chain"] == 8
    assert reporting[
        "profile_stable_nll_difference_identity_tolerance"
    ] == 1e-9
    assert reporting["profile_base_nll_consistency_tolerance"] == (
        likelihood._PROFILE_BASE_NLL_CONSISTENCY_TOLERANCE
    )
    assert reporting["profile_scaled_kkt_tolerance"] == (
        likelihood._PROFILE_STATIONARITY_TOLERANCE
    )
    assert reporting[
        "profile_linear_constraint_relative_tolerance"
    ] == likelihood._PROFILE_LINEAR_CONSTRAINT_RELATIVE_TOLERANCE
    assert reporting["profile_slsqp_max_chains"] == (
        likelihood._PROFILE_SLSQP_MAX_CHAINS
    )
    assert reporting["profile_slsqp_max_iterations_per_chain"] == (
        likelihood._PROFILE_SLSQP_OPTIONS["maxiter"]
    )
    assert reporting["profile_slsqp_ftol"] == (
        likelihood._PROFILE_SLSQP_OPTIONS["ftol"]
    )
    assert reporting["profile_scoring_max_iterations_per_chain"] == (
        likelihood._PROFILE_SCORING_MAX_ITERATIONS
    )
    assert reporting[
        "profile_stable_nll_difference_identity_tolerance"
    ] == likelihood._NLL_DIFFERENCE_IDENTITY_TOLERANCE
    canonical = config["likelihood"]["canonical_intervals"]
    assert "stable-difference SLSQP" in canonical
    assert "exact quadratic-cone coordinates" in canonical
    assert "tight-profile-base consistency" in canonical


def test_fisher_covariance_coordinates_are_frozen():
    config = json.loads(
        (ROOT / "config" / "paper_peak_statistics.json").read_text()
    )
    convention = config["likelihood"]["fisher_covariance"]
    assert "scaled-parameter coordinates" in convention
    assert "transformed back to physical coordinates" in convention


def test_table3_heterogeneity_boundary_policy_is_frozen():
    config = json.loads(
        (ROOT / "config" / "paper_peak_statistics.json").read_text()
    )
    policy = config["table3"]["heterogeneity_boundary_policy"]
    assert "Rate GLS excludes" in policy
    assert "either numerator or denominator" in policy
    assert "fewer than two interior runs" in policy
    assert "unavailable, not negative" in policy


def test_nonstandard_penalized_information_criteria_are_frozen():
    config = json.loads(
        (ROOT / "config" / "paper_peak_statistics.json").read_text()
    )
    semantics = config["likelihood"]["information_criteria"]
    assert "nonstandard penalized-objective descriptive arithmetic" in semantics
    assert "not used for model promotion" in semantics
    fields = _nonstandard_penalized_information_criteria(10.0, 3, 100)
    assert fields["nonstandard_penalized_objective_aic"] == "26"
    assert fields["nonstandard_penalized_objective_bic"] == format(
        20.0 + 3.0 * np.log(100), ".12g"
    )
    assert "nonstandard" in fields["information_criterion_semantics"]
    assert "akaike_information_criterion" not in fields
    assert "bayesian_information_criterion" not in fields


def test_table8_guarded_restart_rule_comes_from_frozen_config():
    config = json.loads(
        (ROOT / "config" / "paper_peak_statistics.json").read_text()
    )
    variants = config["table8"]["component_audit"]["variants"]
    with_restart = [
        variant for variant in variants if "guarded_basin_restart" in variant
    ]
    assert [variant["name"] for variant in with_restart] == [
        "steel_catalog_sensitivity"
    ]
    rule = with_restart[0]["guarded_basin_restart"]
    assert rule["allowed"] is True
    assert "default-config warm chain SLSQP candidate" in rule["initial_source"]
    allowed, source = _table8_guarded_restart_settings(
        "renamed_steel_variant",
        {"renamed_steel_variant": rule},
    )
    assert allowed is True
    assert source == rule["initial_source"]
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


def test_bootstrap_pseudo_observation_convention_is_frozen():
    config = json.loads(
        (ROOT / "config" / "paper_peak_statistics.json").read_text()
    )
    reporting = config["reporting"]
    assert reporting[
        "bootstrap_gaussian_pseudo_observation_seed_domain"
    ] == likelihood._BOOTSTRAP_GAUSSIAN_PSEUDO_OBSERVATION_SEED_DOMAIN
    diagnostic = config["likelihood"]["coverage_diagnostic"]
    assert "every Gaussian constraint pseudo-observation" in diagnostic
    assert "N(fitted generating nuisances, declared covariance)" in diagnostic


def test_manifest_serializes_frozen_reporting_configuration(tmp_path):
    config_path = ROOT / "config" / "paper_peak_statistics.json"
    config = json.loads(config_path.read_text())
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        repo_root=ROOT,
        config=config,
        config_path=config_path,
        database_sha256="manufactured-database-hash",
        input_records={},
        diagnostics_by_table={},
        output_hashes={},
        bootstrap_replicates=1,
        code_revision={"commit": "manufactured", "working_tree_dirty": False},
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["reporting_configuration"] == config["reporting"]
    assert manifest["configuration"]["schema_version"] == 8
    assert manifest["scientific_scope"][
        "candidate_numbers_approved_for_manuscript"
    ] is False


def test_historical_table3_record_freezes_exact_inputs_and_bug_revision():
    record = json.loads(
        (ROOT / "config" / "table3_historical_reconstruction.json").read_text()
    )
    assert record["record_kind"] == "paper-exact historical reconstruction"
    assert record["historical_revision"]["commit"] == (
        "cdabda2dcafce176bad62ea3ca19a9f085b2986b"
    )
    assert record["bug_provenance"]["introduced_commit"] == (
        "5cb9e0afd362e7d3703456531594b0e1424555e3"
    )
    files = record["selection_and_order"]["files"]
    assert [item["file_id"] for item in files] == [1716, 1765, 334, 1676]
    assert all(len(item["spectrum_sha256"]) == 64 for item in files)
    assert record["selection_and_order"]["total_live_time_s"] == 260013.89
    ratios = record["exact_legacy_ratios"]
    assert len(ratios) == 35
    by_energy = {item["energy_keV"]: item for item in ratios}
    assert round(by_energy[238.6]["ratio"], 3) == 0.346
    assert round(by_energy[7367.9]["ratio"], 3) == 0.003
    assert by_energy[478.0]["paper_display"] == "--"


def test_table8_candidates_are_authoritative_declared_and_inside_windows():
    config = json.loads(
        (ROOT / "config" / "paper_peak_statistics.json").read_text()
    )
    table = config["table8"]
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
    assert table["canonical_model_variant"] == (
        f"{table['canonical_component_variant']}_quadratic_background"
    )
    assert {
        "fe54_6268_9_fep",
        "cu63_6616_0_dep",
        "cu63_7127_0_sep",
        "cu63_7638_0_fep",
    } == canonical
    ambiguous = {
        item["name"]: item
        for item in candidates
        if item["name"] in {"al27_6710_7_fep", "ge70_6707_45_fep"}
    }
    assert set(ambiguous) == {"al27_6710_7_fep", "ge70_6707_45_fep"}
    assert all(
        item["classification"]
        == "plausible_ambiguous_sensitivity_not_canonical"
        for item in ambiguous.values()
    )
    assert all(name not in canonical for name in ambiguous)
    al_components = variants["al27_6711_alternative"]
    ge_components = variants["target_plus_fe_cu_ge70"]
    assert al_components - ge_components == {"al27_6710_7_fep"}
    assert ge_components - al_components == {"ge70_6707_45_fep"}
    assert len(al_components) == len(ge_components)


def test_table8_phase2_window_widening_is_explicit_and_common_to_variants():
    config = json.loads(
        (ROOT / "config" / "paper_peak_statistics.json").read_text()
    )
    change = config["table8"]["phase2_window_change"]
    assert change["window"] == "t8_6809_sep"
    assert change["phase1_low_keV"] == 6272.61
    assert change["phase2_low_keV"] == 6255.0
    assert "every phase-2 Table 8 variant" in change["reason"]
    assert "not direct likelihood comparisons" in change["comparison_warning"]


def test_table3_reference_candidates_fail_closed_without_source_provenance():
    config = json.loads(
        (ROOT / "config" / "paper_peak_statistics.json").read_text()
    )
    audit = config["table3"]["reference_window_component_audit"]
    unsupported = {
        item["name"]: item
        for item in audit["candidates"]
        if "unsupported_source" in item["classification"]
    }
    assert set(unsupported) == {
        "co59_555_972",
        "in115_556_845",
        "w186_557_16",
    }
    assert all(item["fit_treatment"] == "not promoted or fitted" for item in unsupported.values())


def test_table3_reference_audit_records_detector_processes_fail_closed():
    config = json.loads(
        (ROOT / "config" / "paper_peak_statistics.json").read_text()
    )
    candidates = {
        item["name"]: item
        for item in config["table3"]["reference_window_component_audit"][
            "candidates"
        ]
    }
    expected = {
        "ge74_595_85_inelastic",
        "annihilation_511_local_continuum",
        "unidentified_escape_into_540_578_roi",
        "unidentified_true_coincidence_sum_in_roi",
    }
    assert expected <= candidates.keys()
    assert all(
        candidates[name]["fit_treatment"].startswith(("not fitted", "not promoted"))
        for name in expected
    )
