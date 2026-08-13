#!/usr/bin/env python3
"""Compare bounded source-run Ge depositions with public reactor-off spectra."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.public_data.cosmic_pilot import (  # noqa: E402
    area_convergence,
    binomial_clopper_pearson,
    check_sato_runtime,
    combine_identical_spectra,
    concatenate_batches,
    depositing_pid_summary,
    fixed_edges,
    load_config,
    load_simulation_batch,
    select_measured_spectra,
    semantic_replay_equal,
    sha256_file,
    shape_distances,
    sideband_diagnostic,
    unit_area_histogram,
    values_outside_masks,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "hpge_cosmic_pilot.json",
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="unaltered official v1.1.0 release archive for SHA verification",
    )
    parser.add_argument(
        "--simulation",
        action="append",
        default=[],
        metavar="LABEL=FILE.h5",
        help="validated source-run batch; repeat labels to aggregate batches",
    )
    parser.add_argument(
        "--replay",
        action="append",
        nargs=3,
        default=[],
        metavar=("LABEL", "FIRST.h5", "SECOND.h5"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _assignment(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"simulation must be LABEL=FILE.h5: {text!r}")
    label, value = text.split("=", 1)
    if not label or not value:
        raise ValueError(f"simulation must be LABEL=FILE.h5: {text!r}")
    return label, Path(value).expanduser().resolve()


def _mask_values(
    values: np.ndarray,
    lower: float,
    upper: float,
    masks: Sequence[Mapping[str, float]],
) -> np.ndarray:
    selected = values[(values >= lower) & (values < upper)]
    return values_outside_masks(selected, masks)


def _shape_rows(
    identity: str,
    kind: str,
    shape: np.ndarray,
    raw_total: float,
    edges: np.ndarray,
    minimum_entries: int = 0,
) -> list[dict[str, Any]]:
    error = np.sqrt(np.maximum(shape * (1.0 - shape), 0.0) / raw_total)
    return [
        {
            "identity": identity,
            "kind": kind,
            "bin_low_keV": float(edges[index]),
            "bin_high_keV": float(edges[index + 1]),
            "unit_area_fraction": float(shape[index]),
            "multinomial_stat_error": float(error[index]),
            "normalization_weight": float(raw_total),
            "raw_bin_weight": float(shape[index] * raw_total),
            "meets_minimum_entry_gate": bool(
                shape[index] * raw_total >= minimum_entries
            ),
        }
        for index in range(shape.size)
    ]


def _measured_shapes(
    spectra: Sequence[Any],
    combined: Any,
    edges: np.ndarray,
    masks: Sequence[Mapping[str, float]],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    shapes: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for spectrum in spectra:
        identity = f"measured_file:{spectrum.file_name}"
        shape, total = unit_area_histogram(
            spectrum.energy_keV, edges, masks, weights=spectrum.counts
        )
        shapes[identity] = shape
        rows.extend(_shape_rows(identity, "measured detector counts", shape, total, edges))
    shape, total = unit_area_histogram(
        combined.energy_keV, edges, masks, weights=combined.counts
    )
    shapes["measured_combined"] = shape
    rows.extend(
        _shape_rows("measured_combined", "measured detector counts", shape, total, edges)
    )
    return shapes, rows


def _low_masks(config: Mapping[str, Any]) -> list[dict[str, float]]:
    comparison = config["comparison"]
    table_path = Path(comparison["table3_config"])
    if not table_path.is_absolute():
        table_path = REPO_ROOT / table_path
    with table_path.open("r", encoding="utf-8") as handle:
        table = json.load(handle)
    lower, upper = comparison["low_window_keV"]
    masks = [dict(comparison["low_broad_mask_keV"])]
    half_width = float(comparison["low_table3_mask_half_width_keV"])
    masks.extend(
        {"center": float(center), "half_width": half_width}
        for center in table["peak_fit_order_keV"]
        if lower <= float(center) <= upper and float(center) != 478.0
    )
    return masks


def _comparison_eligibility(
    label: str,
    config: Mapping[str, Any],
    convergence: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, str]:
    policy = config["source_runs"][label]["measured_comparison"]
    if policy == "exploratory_shape":
        return True, policy
    if policy == "requires_area_convergence":
        matching = [
            result
            for key, result in convergence.items()
            if label in key.split("__vs__")
        ]
        if len(matching) != 1:
            return False, "area_diagnostic_unavailable"
        return matching[0].get("status") == "not_rejected", str(
            matching[0].get("status")
        )
    return False, policy


def _plot_main_shapes(
    output: Path,
    edges: np.ndarray,
    measured_shape: np.ndarray,
    simulation_shapes: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = 0.5 * (edges[:-1] + edges[1:])
    figure, axis = plt.subplots(figsize=(9.0, 5.2))
    axis.step(
        centers,
        measured_shape,
        where="mid",
        color="black",
        linewidth=1.7,
        label="Public reactor-off measured counts",
    )
    for label, shape in simulation_shapes.items():
        source = config["source_runs"][label]["source_component"]
        axis.step(centers, shape, where="mid", linewidth=1.2, label=source)
    if not simulation_shapes:
        axis.text(
            0.5,
            0.5,
            "No simulated source run met the predeclared statistics gate",
            transform=axis.transAxes,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.6"},
        )
    axis.set(
        xlabel="Energy / raw active-Ge deposition (keV)",
        ylabel="Unit-area bin fraction",
        title="HPGe cosmic pilot: shape-only boundary comparison",
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    archive = args.archive.expanduser().resolve()
    archive_sha = sha256_file(archive)
    expected_archive_sha = config["public_release"]["archive_sha256"]
    if archive_sha != expected_archive_sha:
        raise ValueError("official public-release archive SHA256 mismatch")

    bundle = args.bundle.expanduser().resolve()
    spectra = select_measured_spectra(config, bundle)
    combined = combine_identical_spectra(spectra, bundle / "spectra")
    comparison = config["comparison"]
    requested_lower, requested_upper = map(float, comparison["main_window_keV"])
    adc_edge = float(
        combined.energy_keV[-1] + combined.bin_width_keV[-1] / 2.0
    )
    upper = min(requested_upper, adc_edge)
    main_edges = fixed_edges(
        requested_lower, upper, float(comparison["main_bin_width_keV"])
    )
    main_masks = comparison["main_line_masks_keV"]
    measured_main, main_rows = _measured_shapes(
        spectra, combined, main_edges, main_masks
    )

    low_lower, low_upper = map(float, comparison["low_window_keV"])
    low_edges = fixed_edges(
        low_lower, min(low_upper, adc_edge), float(comparison["low_bin_width_keV"])
    )
    low_masks = _low_masks(config)
    measured_low, low_rows = _measured_shapes(spectra, combined, low_edges, low_masks)

    assignments: dict[str, list[Path]] = {}
    for raw in args.simulation:
        label, path = _assignment(raw)
        if label not in config["source_runs"]:
            raise ValueError(f"simulation label is not predeclared: {label}")
        assignments.setdefault(label, []).append(path)

    batch_manifest: list[dict[str, Any]] = []
    aggregates: dict[str, Any] = {}
    sato_closure: dict[str, list[dict[str, Any]]] = {}
    for label, paths in assignments.items():
        expected_pids = config["source_runs"][label]["expected_primary_pids"]
        batches = [load_simulation_batch(path, label, expected_pids) for path in paths]
        for batch in batches:
            entry = {
                "label": label,
                "path": str(batch.path),
                "file_sha256": sha256_file(batch.path),
                "semantic_sha256": batch.semantic_sha256,
                "processed_events": batch.nprim,
                "primary_records": int(batch.primary_pids.size),
                "deposited_events": batch.deposited_event_count,
                "zero_deposition_events": int(
                    batch.nprim - batch.deposited_event_count
                ),
                "events_above_adc_edge": int(
                    np.count_nonzero(batch.deposition_keV >= adc_edge)
                ),
                "events_in_main_window": int(
                    np.count_nonzero(
                        (batch.deposition_keV >= requested_lower)
                        & (batch.deposition_keV < upper)
                    )
                ),
                "events_in_main_window_after_masks": int(
                    _mask_values(
                        batch.deposition_keV,
                        requested_lower,
                        upper,
                        main_masks,
                    ).size
                ),
                "events_in_low_window": int(
                    np.count_nonzero(
                        (batch.deposition_keV >= low_lower)
                        & (batch.deposition_keV < low_upper)
                    )
                ),
                "maximum_deposition_keV": float(batch.deposition_keV.max()),
                "generator_runtime_s": batch.runtime_s,
            }
            batch_manifest.append(entry)
            if label.startswith("sato_"):
                xml_path = Path(f"{batch.path}.xml")
                closure = check_sato_runtime(batch, xml_path)
                closure["xml_path"] = str(xml_path)
                closure["xml_sha256"] = sha256_file(xml_path)
                sato_closure.setdefault(label, []).append(closure)
        aggregates[label] = concatenate_batches(label, batches)

    replay_results: list[dict[str, Any]] = []
    for label, first_raw, second_raw in args.replay:
        if label not in config["source_runs"]:
            raise ValueError(f"replay label is not predeclared: {label}")
        expected_pids = config["source_runs"][label]["expected_primary_pids"]
        first = load_simulation_batch(Path(first_raw), label, expected_pids)
        second = load_simulation_batch(Path(second_raw), label, expected_pids)
        passed = semantic_replay_equal(first, second)
        replay_results.append(
            {
                "label": label,
                "first": str(Path(first_raw).resolve()),
                "second": str(Path(second_raw).resolve()),
                "first_semantic_sha256": first.semantic_sha256,
                "second_semantic_sha256": second.semantic_sha256,
                "excluded_nondeterministic_fields": ["evt.ct (per-event CPU time)"],
                "status": "pass" if passed else "fail",
            }
        )
        if not passed:
            raise ValueError(f"deterministic replay failed for {label}")

    convergence_results: dict[str, dict[str, Any]] = {}
    settings = config["area_convergence"]
    for first_label, second_label in settings["pairs"]:
        if first_label not in aggregates or second_label not in aggregates:
            continue
        first_values = _mask_values(
            aggregates[first_label].deposition_keV,
            requested_lower,
            upper,
            main_masks,
        )
        second_values = _mask_values(
            aggregates[second_label].deposition_keV,
            requested_lower,
            upper,
            main_masks,
        )
        key = f"{first_label}__vs__{second_label}"
        convergence_results[key] = area_convergence(
            first_values, second_values, settings
        )

    simulation_main: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    pid_rows: list[dict[str, Any]] = []
    eligibility: dict[str, dict[str, Any]] = {}
    source_summaries: dict[str, dict[str, Any]] = {}
    for label, batch in aggregates.items():
        main_values = _mask_values(
            batch.deposition_keV, requested_lower, upper, main_masks
        )
        main_bin_counts, _ = np.histogram(main_values, bins=main_edges)
        deposited = batch.deposited_event_count
        source_summaries[label] = {
            "processed_events": batch.nprim,
            "deposited_events": deposited,
            "deposited_event_fraction": deposited / batch.nprim,
            "deposited_event_fraction_exact_95pct": binomial_clopper_pearson(
                deposited, batch.nprim
            ),
            "line_masked_main_window_events": int(main_values.size),
            "line_masked_main_window_event_fraction": main_values.size / batch.nprim,
            "line_masked_main_window_fraction_exact_95pct": binomial_clopper_pearson(
                int(main_values.size), batch.nprim
            ),
            "main_bins_with_at_least_25_entries": int(
                np.count_nonzero(
                    main_bin_counts
                    >= int(config["response_boundary"]["minimum_entries_per_interpreted_bin"])
                )
            ),
            "events_above_adc_edge": int(
                np.count_nonzero(batch.deposition_keV >= adc_edge)
            ),
            "maximum_deposition_keV": float(batch.deposition_keV.max()),
            "generator_runtime_s": batch.runtime_s,
            "aggregate_semantic_sha256": batch.semantic_sha256,
        }
        eligible, reason = _comparison_eligibility(label, config, convergence_results)
        eligibility[label] = {"eligible": eligible, "reason": reason}
        for row in depositing_pid_summary(batch):
            pid_rows.append({"source_run": label, **row})
        if not eligible:
            continue
        minimum_main = int(
            config["source_runs"][label].get(
                "minimum_main_events_for_comparison", 0
            )
        )
        if main_values.size < minimum_main:
            eligibility[label] = {
                "eligible": False,
                "reason": (
                    f"underpowered_at_cap: {main_values.size} < {minimum_main} "
                    "line-masked main-window events"
                ),
            }
            continue
        try:
            main_shape, main_total = unit_area_histogram(
                batch.deposition_keV, main_edges, main_masks
            )
            low_shape, low_total = unit_area_histogram(
                batch.deposition_keV, low_edges, low_masks
            )
        except ValueError as error:
            eligibility[label] = {
                "eligible": False,
                "reason": f"no_comparable_depositions: {error}",
            }
            continue
        simulation_main[label] = main_shape
        main_rows.extend(
            _shape_rows(
                label,
                "simulated source-run deposition",
                main_shape,
                main_total,
                main_edges,
                int(
                    config["response_boundary"][
                        "minimum_entries_per_interpreted_bin"
                    ]
                ),
            )
        )
        low_rows.extend(
            _shape_rows(
                label,
                "simulated source-run deposition",
                low_shape,
                low_total,
                low_edges,
                int(
                    config["response_boundary"][
                        "minimum_entries_per_interpreted_bin"
                    ]
                ),
            )
        )
        for measured_identity, measured_shape in measured_main.items():
            metric_rows.append(
                {
                    "window": "main_3000_to_adc_edge_line_masked",
                    "source_run": label,
                    "measured_identity": measured_identity,
                    **shape_distances(main_shape, measured_shape),
                }
            )
        for measured_identity, measured_shape in measured_low.items():
            metric_rows.append(
                {
                    "window": "low_500_to_3000_line_masked",
                    "source_run": label,
                    "measured_identity": measured_identity,
                    **shape_distances(low_shape, measured_shape),
                }
            )

    feature_centers = [float(mask["center"]) for mask in main_masks]
    feature_inputs = [
        (
            f"measured_file:{spectrum.file_name}",
            "measured detector counts",
            spectrum.energy_keV,
            spectrum.counts,
            "measured_mixture_diagnostic",
        )
        for spectrum in spectra
    ]
    feature_inputs.append(
        (
            "measured_combined",
            "measured detector counts",
            combined.energy_keV,
            combined.counts,
            "measured_mixture_diagnostic",
        )
    )
    feature_inputs.extend(
        (
            label,
            "simulated source-run deposition",
            batch.deposition_keV,
            None,
            (
                "eligible"
                if eligibility[label]["eligible"]
                else f"not_interpreted: {eligibility[label]['reason']}"
            ),
        )
        for label, batch in aggregates.items()
    )
    for identity, kind, values, weights, interpretation in feature_inputs:
        for center in feature_centers:
            feature_rows.append(
                {
                    "identity": identity,
                    "kind": kind,
                    "interpretation": interpretation,
                    **sideband_diagnostic(
                        values,
                        center,
                        float(comparison["named_feature_signal_half_width_keV"]),
                        comparison["named_feature_sidebands_keV"],
                        weights,
                    ),
                }
            )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    products = {
        "main_shapes": output_dir / "main_shapes.csv",
        "low_shapes": output_dir / "low_shapes.csv",
        "shape_metrics": output_dir / "shape_metrics.csv",
        "feature_diagnostics": output_dir / "feature_diagnostics.csv",
        "depositing_pid_summary": output_dir / "depositing_pid_summary.csv",
        "main_plot": output_dir / "main_shape_comparison.png",
    }
    pd.DataFrame(main_rows).to_csv(products["main_shapes"], index=False)
    pd.DataFrame(low_rows).to_csv(products["low_shapes"], index=False)
    pd.DataFrame(
        metric_rows,
        columns=(
            "window",
            "source_run",
            "measured_identity",
            "total_variation",
            "cosine_similarity",
        ),
    ).to_csv(products["shape_metrics"], index=False)
    pd.DataFrame(feature_rows).to_csv(products["feature_diagnostics"], index=False)
    pd.DataFrame(pid_rows).to_csv(products["depositing_pid_summary"], index=False)
    _plot_main_shapes(
        products["main_plot"],
        main_edges,
        measured_main["measured_combined"],
        simulation_main,
        config,
    )

    product_hashes = {name: sha256_file(path) for name, path in products.items()}
    manifest = {
        "schema_version": 1,
        "record_kind": config["record_kind"],
        "claim_boundary": config["claim_boundary"],
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "implementation_sha256": {
            "src/public_data/cosmic_pilot.py": sha256_file(
                REPO_ROOT / "src" / "public_data" / "cosmic_pilot.py"
            ),
            "scripts/compare_hpge_cosmic_pilot.py": sha256_file(
                Path(__file__).resolve()
            ),
        },
        "public_archive": str(archive),
        "public_archive_url": config["public_release"]["archive_url"],
        "public_archive_sha256": archive_sha,
        "public_database_sha256": config["public_release"]["database_sha256"],
        "measured": {
            "run_id": combined.run_id,
            "run_name": combined.run_name,
            "file_ids": list(combined.file_ids),
            "file_names": list(combined.file_names),
            "spectrum_sha256": list(combined.spectrum_sha256),
            "total_live_time_s": combined.live_time_s,
            "calibration_A0_keV": combined.calibration_A0,
            "calibration_A1_keV_per_channel": combined.calibration_A1,
            "adc_edge_keV": adc_edge,
        },
        "comparison": {
            "main_lower_keV": requested_lower,
            "main_upper_keV": upper,
            "normalization": comparison["normalization"],
            "raw_deposition_preserved_above_adc": True,
            "response_forward_model": False,
            "absolute_normalization": False,
            "fitted_component_mixture": False,
        },
        "simulation_batches": batch_manifest,
        "source_run_eligibility": eligibility,
        "source_run_summaries": source_summaries,
        "deterministic_replay": replay_results,
        "area_convergence": convergence_results,
        "sato_runtime_closure": sato_closure,
        "interpretation_guardrails": [
            "CRY/Sato are open-sky boundary conditions, not HFIR hall truth.",
            "Source-run labels are not event ancestry labels.",
            "Reactor-off data are a mixed background, not a pure cosmic target.",
            (
                "Similarity cannot identify a cosmic origin; incompatibility may "
                "falsify a source-run shape."
            ),
            "Sato thermal and nonthermal runs are not summed as an absolute prediction.",
        ],
        "products": {
            name: {"path": str(path), "sha256": product_hashes[name]}
            for name, path in products.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote bounded shape comparison to {output_dir}")
    print(f"Manifest SHA256: {sha256_file(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
