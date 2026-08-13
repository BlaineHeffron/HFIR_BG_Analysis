import json
from pathlib import Path

import numpy as np
import pytest

from src.public_data.peak_likelihood import LineComponent
from src.public_data.table8_local import (
    assemble_independent_ratio_covariance,
    classify_table8_ratio,
    ratio_stability_summary,
    table8_ratio_definitions,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "paper_peak_statistics.json").read_text())
HISTORICAL = json.loads(
    (ROOT / "config" / "table8_historical_reconstruction.json").read_text()
)


def test_table8_local_grouping_bins_and_escape_arithmetic_are_frozen():
    table = CONFIG["table8"]
    workflow = table["local_workflow"]
    groups = workflow["grouping"]
    assert [len(item["parents_keV"]) for item in groups] == [1, 4, 1, 1, 1]
    assert sum(len(item["windows"]) for item in groups) == 15
    assert len(workflow["native_windows"]) == 15
    assert sum(
        item["last_channel_1_based"] - item["first_channel_1_based"] + 1
        for item in workflow["native_windows"]
    ) == 1176
    assert HISTORICAL["input"]["counts_sha256_int64_little_endian"] == (
        "f53a0291840a1ed20cf1e302483a063a1618f4e273dab4ba62a1ca82d7155b50"
    )
    components = tuple(
        LineComponent(
            item["name"],
            item["energy_keV"],
            item["window"],
            item["role"],
            item["parent"],
        )
        for item in table["components"]
    )
    definitions = table8_ratio_definitions(components, table["target_parents_keV"])
    assert len(definitions) == 24
    offsets = {"fep": 0.0, "sep": 511.0, "dep": 1022.0}
    for item in table["components"]:
        assert item["energy_keV"] == pytest.approx(
            float(item["parent"]) - offsets[item["role"]], abs=1e-9
        )


def test_table8_local_covariance_keeps_multiplet_block_and_zeroes_cross_group():
    labels, covariance = assemble_independent_ratio_covariance(
        (("cluster:a", "cluster:b"), ("isolated:c",)),
        (
            np.asarray(((4.0, -1.25), (-1.25, 9.0))),
            np.asarray(((16.0,),)),
        ),
    )
    assert labels == ("cluster:a", "cluster:b", "isolated:c")
    assert covariance[:2, :2].tolist() == [[4.0, -1.25], [-1.25, 9.0]]
    assert np.count_nonzero(covariance[:2, 2:]) == 0
    assert np.count_nonzero(covariance[2:, :2]) == 0


def test_table8_11386_basin_check_is_deterministic_and_fail_closed():
    workflow = CONFIG["table8"]["local_workflow"]
    check = workflow["canonical_basin_stability_check"]
    assert check["group"] == "parent_11386_500"
    assert check["initial_source"] == "canonical_initial_from_frozen_phase2"
    assert [item["multiplicative_factor"] for item in check["perturbations"]] == [
        0.9,
        1.1,
    ]
    assert all("seed" not in item for item in check["perturbations"])
    assert check["acceptance"]["require_identical_active_bounds"] is True
    assert (
        check["acceptance"]["require_success_full_rank_covariance"] is True
    )


def test_table8_stability_floor_is_explicit_and_statuses_fail_closed():
    floor = ratio_stability_summary(1.0, 0.01, {"variant": 1.04})
    assert floor["passes"] is True
    assert floor["five_percent_floor_only"] is True
    assert floor["sigma_units"] == pytest.approx(4.0)
    movement = ratio_stability_summary(1.0, 0.01, {"variant": 1.06})
    assert movement["passes"] is False
    common = dict(
        canonical_available=True,
        profile_kind="regular_interior_fisher",
        profile_excludes_zero=True,
        numerator_z=10.0,
        denominator_z=10.0,
        numerator_bound=False,
        denominator_bound=False,
        covariance_valid=True,
        target_core_residual=False,
        model_sensitive=False,
        mandatory_model_sensitive=False,
        tail_status_differs=False,
    )
    assert classify_table8_ratio(unresolved=True, **common)[0] == "X"
    assert classify_table8_ratio(unresolved=False, **common)[0] == "Q"
    boundary = {**common, "numerator_bound": True, "profile_kind": "upper_limit"}
    assert classify_table8_ratio(unresolved=False, **boundary)[0] == "B"
