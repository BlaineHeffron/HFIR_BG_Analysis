#!/usr/bin/env python3
"""Generate the main public analysis products from downloaded data.

The unfolded spectra are the official arXiv ancillary results. The shield
comparison is rebuilt from the public calibrated spectra and SQLite metadata.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import sys
from contextlib import nullcontext, redirect_stdout
from copy import copy
from pathlib import Path

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_PAPER_DATA = REPO_ROOT / "data" / "arxiv_2607.05834" / "anc"
DEFAULT_ANALYSIS_ROOT = Path(os.environ.get("HFIRBG_ANALYSIS", REPO_ROOT / "analysis"))

UNFOLDED_LOCATIONS = (
    ("01", "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN", "MIF (Rx On)"),
    ("02", "MIF_BOX_AT_REACTOR_RXOFF", "MIF (Rx Off)"),
    ("03", "CYCLE461_DOWN_FACING_OVERNIGHT", "Shield Center"),
)

# These IDs and gain modes reproduce Figure 14. Historical database names for
# IDs 5-7 all contain "plusLead", but the campaign log/paper distinguish them
# as water-only, water+floor-lead, and the final water+floor-lead layouts.
SHIELD_CONFIGURATIONS = (
    (2, 7, "no_added_shield", "No added water or floor lead"),
    (5, 17, "seven_water_layers", "7 water layers"),
    (6, 17, "six_water_layers_floor_lead", "6 water layers + floor lead"),
    (7, 17, "seven_water_layers_floor_lead", "7 water layers + floor lead"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        choices=("all", "unfolded", "shields"),
        nargs="?",
        default="all",
        help="analysis product to generate (default: all)",
    )
    parser.add_argument(
        "--paper-data",
        type=Path,
        default=Path(os.environ.get("HFIRBG_PAPER_DATA", DEFAULT_PAPER_DATA)),
        help="directory containing the official arXiv ancillary files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ANALYSIS_ROOT / "public_analysis",
        help="output directory (default: $HFIRBG_ANALYSIS/public_analysis)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show per-file calibration messages while loading shield spectra",
    )
    return parser.parse_args()


def require_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"required file not found: {path}\n"
            "Run scripts/setup_analysis.sh to download the official paper results."
        )


def plot_unfolded_key_locations(paper_data: Path, output_dir: Path):
    """Plot the two response-model bounds for the three requested locations."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)
    manifest_rows = []
    colors = {"isotropic": "#1f77b4", "front": "#d62728"}

    for axis, (number, filename, alias) in zip(axes, UNFOLDED_LOCATIONS):
        for scenario in ("isotropic", "front"):
            source = paper_data / f"unfolded_spectrum_{scenario}_{number}_{filename}.csv"
            require_file(source)
            frame = pd.read_csv(source)
            required = {"Energy_keV", "Flux_Hz_per_mm2_per_keV"}
            if not required.issubset(frame.columns):
                raise ValueError(f"unexpected columns in {source}: {list(frame.columns)}")
            positive = frame["Flux_Hz_per_mm2_per_keV"] > 0
            axis.step(
                frame.loc[positive, "Energy_keV"],
                frame.loc[positive, "Flux_Hz_per_mm2_per_keV"],
                where="mid",
                linewidth=0.8,
                color=colors[scenario],
                label=scenario.capitalize(),
            )
            manifest_rows.append((number, alias, scenario, source.name))
        axis.set_yscale("log")
        axis.set_ylabel("Flux [Hz/mm²/keV]")
        axis.set_title(alias)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()

    axes[-1].set_xlabel("Energy [keV]")
    axes[-1].set_xlim(40, 12000)
    fig.suptitle("Unfolded gamma flux at key HFIR locations")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"unfolded_key_locations.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    with (output_dir / "unfolded_key_locations_manifest.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("measurement_number", "location", "response_scenario", "source_csv"))
        writer.writerows(manifest_rows)


def load_shield_spectra(verbose=False):
    """Load and combine reactor-on/off spectra used by the paper comparison."""
    if not os.environ.get("HFIRBGDATA"):
        raise RuntimeError("HFIRBGDATA is not set; source .env before running this analysis")

    from src.database.HFIRBG_DB import HFIRBG_DB
    from src.utilities.util import get_rd_data

    output_context = nullcontext() if verbose else redirect_stdout(io.StringIO())
    with output_context:
        with HFIRBG_DB() as database:
            reactor_on = get_rd_data(database, rxon_only=True)
            reactor_off = get_rd_data(database, rxoff_only=True)

    selected = []
    for shield_id, gain_id, slug, label in SHIELD_CONFIGURATIONS:
        try:
            on_spectrum = reactor_on[shield_id][gain_id]
            off_spectrum = reactor_off[shield_id][gain_id]
        except KeyError as error:
            raise RuntimeError(
                f"public database is missing shield {shield_id}, gain {gain_id} data"
            ) from error
        selected.append((slug, label, on_spectrum, off_spectrum))
    return selected


def plot_shield_comparison(output_dir: Path, verbose=False):
    """Reproduce the Figure 14 comparison and export its normalized values."""
    selected = load_shield_spectra(verbose=verbose)
    bin_edges = np.linspace(30.0, 60.0, 181)
    fig, axis = plt.subplots(figsize=(11, 7))
    export = {"Energy_keV": None}
    rate_rows = []

    for slug, label, on_spectrum, off_spectrum in selected:
        rebinned = copy(on_spectrum)
        rebinned.rebin(bin_edges)
        energy = rebinned.bin_midpoints
        rate = 1000.0 * rebinned.get_normalized_hist()
        error = 1000.0 * rebinned.get_normalized_err()
        export["Energy_keV"] = energy
        export[f"{slug}_Rate_mHz_per_kg_per_keV"] = rate
        export[f"{slug}_StatErr_mHz_per_kg_per_keV"] = error
        axis.errorbar(energy, rate, yerr=error, linewidth=1.0, elinewidth=0.5, label=label)

        on_rate, on_error = on_spectrum.integrate(30, 60, True)
        off_rate, off_error = off_spectrum.integrate(30, 60, True)
        rate_rows.append(
            {
                "configuration": label,
                "rx_on_mHz_per_kg_per_keV": 1000.0 * on_rate,
                "rx_on_stat_error_mHz_per_kg_per_keV": 1000.0 * on_error,
                "rx_off_mHz_per_kg_per_keV": 1000.0 * off_rate,
                "rx_off_stat_error_mHz_per_kg_per_keV": 1000.0 * off_error,
                "rx_only_mHz_per_kg_per_keV": 1000.0 * (on_rate - off_rate),
                "rx_only_stat_error_mHz_per_kg_per_keV": 1000.0
                * math.hypot(on_error, off_error),
            }
        )

    axis.set_xlim(30, 60)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Energy [keV]")
    axis.set_ylabel("Rate [mHz/kg/keV]")
    axis.set_title("Russian Doll shield configurations (Figure 14)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"shield_configuration_spectra.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(export).to_csv(output_dir / "shield_configuration_spectra.csv", index=False)
    pd.DataFrame(rate_rows).to_csv(output_dir / "shield_configuration_rates_30_60_keV.csv", index=False)


def main():
    args = parse_args()
    args.output_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    output_dir = args.output_dir.expanduser().resolve()
    paper_data = args.paper_data.expanduser().resolve()

    if args.target in ("all", "unfolded"):
        plot_unfolded_key_locations(paper_data, output_dir)
        print(f"Wrote unfolded key-location products to {output_dir}")
    if args.target in ("all", "shields"):
        plot_shield_comparison(output_dir, verbose=args.verbose)
        print(f"Wrote shield-comparison products to {output_dir}")


if __name__ == "__main__":
    main()
