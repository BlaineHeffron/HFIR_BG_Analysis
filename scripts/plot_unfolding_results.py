import sys
from os.path import dirname, realpath, join, basename

sys.path.insert(1, dirname(dirname(realpath(__file__))))

from argparse import ArgumentParser
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot
from ROOT import TFile


FNAMES = [
    "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN",
    "MIF_BOX_AT_REACTOR_RXOFF",
    "CYCLE461_DOWN_FACING_OVERNIGHT",
    "HB4_DOWN_OVERNIGHT_1",
    "EAST_FACE_18",
    "EAST_FACE_1",
]

ALIAS_MAP = {
    "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN": "MIF (Rx On)",
    "MIF_BOX_AT_REACTOR_RXOFF": "MIF (Rx Off)",
    "CYCLE461_DOWN_FACING_OVERNIGHT": "Shield Center",
    "HB4_DOWN_OVERNIGHT_1": "HB4",
    "EAST_FACE_18": "PROSPECT East 1",
    "EAST_FACE_1": "PROSPECT East 2",
}

MEASUREMENT_METADATA = [
    {
        "number": 1,
        "filename": "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN",
        "alias": "MIF (Rx On)",
        "z_pos": 109.38824265765834,
        "x_pos": 27.581965491243988,
        "angle": 46.5,
        "phi_deg": 76.75948008481281,
    },
    {
        "number": 2,
        "filename": "MIF_BOX_AT_REACTOR_RXOFF",
        "alias": "MIF (Rx Off)",
        "z_pos": 109.38824265765834,
        "x_pos": 27.581965491243988,
        "angle": 46.5,
        "phi_deg": 76.75948008481281,
    },
    {
        "number": 3,
        "filename": "CYCLE461_DOWN_FACING_OVERNIGHT",
        "alias": "Shield Center",
        "z_pos": 207.7,
        "x_pos": 46.5,
        "angle": 0.0,
        "phi_deg": 0.0,
    },
    {
        "number": 4,
        "filename": "HB4_DOWN_OVERNIGHT_1",
        "alias": "HB4",
        "z_pos": 307.7,
        "x_pos": 111.0,
        "angle": 0.0,
        "phi_deg": 0.0,
    },
    {
        "number": 5,
        "filename": "EAST_FACE_18",
        "alias": "PROSPECT East 1",
        "z_pos": 215.2,
        "x_pos": 72.0,
        "angle": 90.0,
        "phi_deg": 0.0,
    },
    {
        "number": 6,
        "filename": "EAST_FACE_1",
        "alias": "PROSPECT East 2",
        "z_pos": 214.3594748928162,
        "x_pos": 93.0,
        "angle": 63.5,
        "phi_deg": 0.0,
    },
]

SCENARIO_DEFAULTS = {
    "isotropic": {
        "label": "Isotropic",
        "results_dir": "unfold/sumita",
        "migration_matrix": "scripts/private/migration_matrix.root",
        "notes": "Uniform isotropic flux over the outer surface of the collimator-detector system.",
    },
    "front": {
        "label": "Front",
        "results_dir": "unfold/front",
        "migration_matrix": "scripts/private/migration_matrix_front.root",
        "notes": "Uniform directional flux incident on the detector/collimator front face.",
    },
}

PUBLIC_DATA_BUNDLE = "HFIRBG_public_data_v1.1.0"


def hist_to_arrays(hist):
    n_bins = hist.GetNbinsX()
    x = np.array([hist.GetBinCenter(i + 1) for i in range(n_bins)])
    y = np.array([hist.GetBinContent(i + 1) for i in range(n_bins)])
    edges = np.array([hist.GetBinLowEdge(i + 1) for i in range(n_bins + 1)])
    return x, y, edges


def export_spectrum_to_csv(x, y, filename, columns):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        for idx in range(len(x)):
            writer.writerow([f"{x[idx]:.2f}", f"{y[idx]:.6e}"])


def save_csv_rows(path, fieldnames, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_case(case_name, case_cfg, repo_root):
    case_results = []
    results_dir = case_cfg["results_dir"]
    if not os.path.isabs(results_dir):
        results_dir = join(repo_root, "analysis", results_dir) if not results_dir.startswith("analysis/") else join(repo_root, results_dir)

    for idx, fname in enumerate(FNAMES, start=1):
        filepath = join(results_dir, f"{fname}_unfold_results.root")
        if not os.path.exists(filepath):
            print(f"WARNING: Missing {case_name} result file: {filepath}")
            continue

        root_file = TFile(filepath, "READ")
        hist_unf = root_file.Get("UnfoldedEnergy")
        hist_meas = root_file.Get("Measured")

        if not hist_unf:
            print(f"WARNING: Could not retrieve UnfoldedEnergy from {filepath}")
            root_file.Close()
            continue

        x_unf, y_unf, edges_unf = hist_to_arrays(hist_unf)
        if hist_meas:
            x_meas, y_meas, _ = hist_to_arrays(hist_meas)
        else:
            x_meas, y_meas = None, None

        case_results.append(
            {
                "number": idx,
                "filename": fname,
                "alias": ALIAS_MAP.get(fname, fname),
                "case": case_name,
                "case_label": case_cfg["label"],
                "root_path": filepath,
                "x": x_unf,
                "y": y_unf,
                "edges": edges_unf,
                "x_meas": x_meas,
                "y_meas": y_meas,
            }
        )
        root_file.Close()

    return case_results


def write_measurement_metadata(outdir):
    metadata_path = join(outdir, "measurement_metadata.csv")
    save_csv_rows(
        metadata_path,
        ["number", "filename", "alias", "z_pos", "x_pos", "angle", "phi_deg"],
        MEASUREMENT_METADATA,
    )
    return metadata_path


def write_case_metadata(outdir, case_data, case_cfgs, repo_root):
    def summarize_matrix(matrix_path):
        if not os.path.exists(matrix_path):
            return {
                "simulated_energy_count": "",
                "supported_energy_min_keV": "",
                "supported_energy_max_keV": "",
            }
        root_file = uproot.open(matrix_path)
        energies = root_file["hEnergies"].values()
        mask = root_file["hSimulatedMask"].values() > 0
        supported = energies[mask]
        if len(supported) == 0:
            return {
                "simulated_energy_count": 0,
                "supported_energy_min_keV": "",
                "supported_energy_max_keV": "",
            }
        return {
            "simulated_energy_count": int(mask.sum()),
            "supported_energy_min_keV": float(supported.min()),
            "supported_energy_max_keV": float(supported.max()),
        }

    scenario_rows = []
    for case_name, specs in case_cfgs.items():
        matrix_path = specs["migration_matrix"]
        if not os.path.isabs(matrix_path):
            matrix_path = join(repo_root, matrix_path)
        matrix_summary = summarize_matrix(matrix_path)
        scenario_rows.append(
            {
                "scenario": case_name,
                "label": specs["label"],
                "public_bundle_matrix": (
                    f"{PUBLIC_DATA_BUNDLE}/migration_matrices/"
                    f"migration_matrix_{case_name}.npz"
                ),
                "simulated_energy_count": matrix_summary["simulated_energy_count"],
                "supported_energy_min_keV": matrix_summary["supported_energy_min_keV"],
                "supported_energy_max_keV": matrix_summary["supported_energy_max_keV"],
                "notes": specs["notes"],
            }
        )

    save_csv_rows(
        join(outdir, "unfolding_scenarios.csv"),
        [
            "scenario",
            "label",
            "public_bundle_matrix",
            "simulated_energy_count",
            "supported_energy_min_keV",
            "supported_energy_max_keV",
            "notes",
        ],
        scenario_rows,
    )

    measurement_case_rows = []
    for case_name, spectra in case_data.items():
        for spec in spectra:
            measurement_case_rows.append(
                {
                    "number": spec["number"],
                    "filename": spec["filename"],
                    "alias": spec["alias"],
                    "scenario": case_name,
                    "unfolded_csv": (
                        "unfolding_results/spectra/"
                        f"unfolded_spectrum_{case_name}_{spec['number']:02d}_"
                        f"{spec['filename']}.csv"
                    ),
                }
            )

    save_csv_rows(
        join(outdir, "measurement_case_files.csv"),
        ["number", "filename", "alias", "scenario", "unfolded_csv"],
        measurement_case_rows,
    )


def export_case_csvs(outdir, case_name, spectra, measured_reference):
    for spec in spectra:
        csv_filename = join(outdir, f"unfolded_spectrum_{case_name}_{spec['number']:02d}_{spec['filename']}.csv")
        export_spectrum_to_csv(
            spec["x"],
            spec["y"],
            csv_filename,
            columns=("Energy_keV", "Flux_Hz_per_mm2_per_keV"),
        )

    for spec in measured_reference:
        if spec["x_meas"] is None:
            continue
        meas_csv = join(outdir, f"measured_spectrum_{spec['number']:02d}_{spec['filename']}.csv")
        export_spectrum_to_csv(
            spec["x_meas"],
            spec["y_meas"],
            meas_csv,
            columns=("Energy_keV", "Rate_Hz_per_keV"),
        )


def save_figure(fig, outdir, stem):
    png = join(outdir, f"{stem}.png")
    pdf = join(outdir, f"{stem}.pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png}")
    print(f"Saved {pdf}")


def plot_case_overlay(outdir, case_name, spectra, title_suffix):
    if not spectra:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(spectra)))

    for idx, spec in enumerate(spectra):
        ax.step(
            spec["x"],
            spec["y"],
            where="mid",
            color=colors[idx],
            linewidth=1.5,
            alpha=0.85,
            label=spec["alias"],
        )

    ax.set_xlim(50, 11500)
    ax.set_yscale("log")
    ax.set_xlabel("Energy [keV]", fontsize=14)
    ax.set_ylabel("Gamma Flux [Hz/mm$^2$/keV]", fontsize=14)
    ax.set_title(f"Unfolded Gamma Spectra ({title_suffix})", fontsize=16)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    save_figure(fig, outdir, f"all_unfolded_spectra_{case_name}")


def plot_measured_vs_unfolded(outdir, case_name, spectra, title_suffix):
    if not spectra:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    color_meas = "#1f77b4"
    color_unf = "#d62728"

    for idx, spec in enumerate(spectra[:6]):
        row, col = divmod(idx, 3)
        ax1 = axes[row, col]
        if spec["y_meas"] is not None:
            ax1.step(
                spec["x_meas"],
                spec["y_meas"],
                where="mid",
                color=color_meas,
                linewidth=1.0,
                alpha=0.8,
                label="Measured",
            )
        ax1.set_yscale("log")
        ax1.set_ylabel("Measured [counts/keV]", fontsize=10, color=color_meas)
        ax1.tick_params(axis="y", labelcolor=color_meas, labelsize=9)

        ax2 = ax1.twinx()
        ax2.step(
            spec["x"],
            spec["y"],
            where="mid",
            color=color_unf,
            linewidth=1.0,
            alpha=0.8,
            label=title_suffix,
        )
        ax2.set_yscale("log")
        ax2.set_ylabel("Flux [Hz/mm$^2$/keV]", fontsize=10, color=color_unf)
        ax2.tick_params(axis="y", labelcolor=color_unf, labelsize=9)

        ax1.set_xlim(50, 11500)
        ax1.set_xlabel("Energy [keV]", fontsize=10)
        ax1.set_title(spec["alias"], fontsize=12)
        ax1.grid(True, alpha=0.3)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    plt.tight_layout()
    save_figure(fig, outdir, f"measured_vs_unfolded_comparison_{case_name}")


def plot_bounds(outdir, iso_spectra, front_spectra):
    if not iso_spectra or not front_spectra:
        return

    iso_map = {spec["filename"]: spec for spec in iso_spectra}
    front_map = {spec["filename"]: spec for spec in front_spectra}
    common = [name for name in FNAMES if name in iso_map and name in front_map]
    if not common:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, fname in enumerate(common):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        iso = iso_map[fname]
        front = front_map[fname]
        lower = np.minimum(iso["y"], front["y"])
        upper = np.maximum(iso["y"], front["y"])

        ax.fill_between(
            iso["x"],
            lower,
            upper,
            step="mid",
            color="#d9d9d9",
            alpha=0.85,
            label="Bracketed range",
        )
        ax.step(iso["x"], iso["y"], where="mid", color="#1f77b4", linewidth=1.1, label="Isotropic")
        ax.step(front["x"], front["y"], where="mid", color="#d62728", linewidth=1.1, label="Front")
        ax.set_xlim(50, 11500)
        ax.set_yscale("log")
        ax.set_xlabel("Energy [keV]", fontsize=10)
        ax.set_ylabel("Flux [Hz/mm$^2$/keV]", fontsize=10)
        ax.set_title(iso["alias"], fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    save_figure(fig, outdir, "unfolded_spectrum_bounds")


def plot_individual_unfolded(outdir, case_name, spectra, title_suffix):
    """Generate individual full-size plots for each measurement location."""
    if not spectra:
        return

    for spec in spectra:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.step(
            spec["x"],
            spec["y"],
            where="mid",
            color="#d62728",
            linewidth=1.5,
            alpha=0.85,
            label=title_suffix,
        )
        ax.set_xlim(50, 11500)
        ax.set_yscale("log")
        ax.set_xlabel("Energy [keV]", fontsize=14)
        ax.set_ylabel("Gamma Flux [Hz/mm$^2$/keV]", fontsize=14)
        ax.set_title(f"Unfolded Gamma Spectrum: {spec['alias']} ({title_suffix})", fontsize=16)
        ax.legend(loc="upper right", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
        plt.tight_layout()

        safe_alias = spec["alias"].replace(" ", "_").replace("(", "").replace(")", "")
        save_figure(fig, outdir, f"unfolded_{safe_alias}_{case_name}")


def copy_paper_figures(outdir, paper_dir):
    ensure_dir(paper_dir)
    copy_pairs = {
        "measurement_locations.pdf": "measurement_locations.pdf",
        "measured_vs_unfolded_comparison_isotropic.pdf": "measured_vs_unfolded_comparison.pdf",
        "all_unfolded_spectra_isotropic.pdf": "all_unfolded_spectra.pdf",
        "measured_vs_unfolded_comparison_front.pdf": "measured_vs_unfolded_comparison_front.pdf",
        "all_unfolded_spectra_front.pdf": "all_unfolded_spectra_front.pdf",
        "unfolded_spectrum_bounds.pdf": "unfolded_spectrum_bounds.pdf",
    }

    for src_name, dst_name in copy_pairs.items():
        src = join(outdir, src_name)
        dst = join(paper_dir, dst_name)
        if os.path.exists(src):
            with open(src, "rb") as src_handle, open(dst, "wb") as dst_handle:
                dst_handle.write(src_handle.read())
            print(f"Copied {src} -> {dst}")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: $HFIRBG_ANALYSIS/unfold/paper_files)",
    )
    parser.add_argument(
        "--paper-fig-dir",
        default=os.environ.get("HFIRBG_PAPER_FIG_DIR"),
        help="Optional destination for paper-ready PDFs (default: $HFIRBG_PAPER_FIG_DIR)",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["isotropic", "front"],
        help="Scenarios to process",
    )
    args = parser.parse_args()

    repo_root = dirname(realpath(dirname(__file__)))
    if args.outdir is None:
        analysis_root = os.environ.get("HFIRBG_ANALYSIS", join(repo_root, "analysis"))
        outdir = join(analysis_root, "unfold", "paper_files")
    else:
        outdir = args.outdir
    ensure_dir(outdir)

    requested_cases = {}
    for case_name in args.cases:
        if case_name not in SCENARIO_DEFAULTS:
            raise ValueError(f"Unknown case: {case_name}")
        requested_cases[case_name] = SCENARIO_DEFAULTS[case_name]

    write_measurement_metadata(outdir)

    case_data = {}
    for case_name, cfg in requested_cases.items():
        case_data[case_name] = load_case(case_name, cfg, repo_root)
        export_case_csvs(outdir, case_name, case_data[case_name], case_data[case_name])
        plot_case_overlay(outdir, case_name, case_data[case_name], cfg["label"])
        plot_measured_vs_unfolded(outdir, case_name, case_data[case_name], cfg["label"])
        plot_individual_unfolded(outdir, case_name, case_data[case_name], cfg["label"])

    measured_reference = None
    for case_name in ["isotropic", "front"]:
        if case_name in case_data and case_data[case_name]:
            measured_reference = case_data[case_name]
            break
    if measured_reference:
        export_case_csvs(outdir, "isotropic" if "isotropic" in case_data else measured_reference[0]["case"], [], measured_reference)

    write_case_metadata(outdir, case_data, requested_cases, repo_root)
    plot_bounds(outdir, case_data.get("isotropic", []), case_data.get("front", []))
    if args.paper_fig_dir:
        copy_paper_figures(outdir, args.paper_fig_dir)

    print(f"Processed cases: {', '.join(case_data.keys())}")
    print(f"Results saved to: {outdir}")


if __name__ == "__main__":
    main()
