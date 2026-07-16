#!/usr/bin/env python3
from argparse import ArgumentParser
import csv
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot


HIST_PATHS = ["GeEnergyPlugin/hGeEnergy", "miscAccum/hGeEnergy"]
RUNTIME_PATHS = ["accumulated/runtime", "miscAccum/runtime"]


def try_read_object(root_file, paths):
    for path in paths:
        try:
            return root_file[path]
        except Exception:
            continue
    return None


def read_runtime(root_file):
    runtime_obj = try_read_object(root_file, RUNTIME_PATHS)
    if runtime_obj is None:
        return 0.0
    for attr in ["fVal", "value"]:
        try:
            value = runtime_obj.member(attr) if hasattr(runtime_obj, "member") else getattr(runtime_obj, attr)
            if value is not None:
                return float(value)
        except Exception:
            continue
    return 0.0


def collect_files(directory, front=False):
    directory = Path(directory).expanduser()
    pattern = re.compile(r"^(\d+)_keV_dir_front_(\d+)_Phys\.root$" if front else r"^(\d+)_keV_(\d+)_Phys\.root$")
    files = {}
    for path in directory.iterdir():
        match = pattern.match(path.name)
        if match:
            files.setdefault(int(match.group(1)), []).append(path)
    return {energy: sorted(paths) for energy, paths in files.items()}


def summarize_energy(paths):
    combined_values = None
    combined_errors_sq = None
    runtime = 0.0

    for path in paths:
        with uproot.open(path) as root_file:
            hist = try_read_object(root_file, HIST_PATHS)
            if hist is None:
                continue
            values = hist.values()
            errors = hist.errors() if hasattr(hist, "errors") else np.sqrt(values)
            runtime += read_runtime(root_file)
            combined_values = values.copy() if combined_values is None else combined_values + values
            combined_errors_sq = errors ** 2 if combined_errors_sq is None else combined_errors_sq + errors ** 2

    if combined_values is None:
        return None

    combined_errors = np.sqrt(combined_errors_sq)
    mask = combined_values > 0
    rel_errors = np.divide(
        combined_errors,
        combined_values,
        out=np.full_like(combined_values, np.nan, dtype=float),
        where=mask,
    )

    return {
        "files": len(paths),
        "runtime_s": runtime,
        "sum_counts": float(combined_values.sum()),
        "nonzero_bins": int(mask.sum()),
        "median_rel_error": float(np.nanmedian(rel_errors[mask])) if np.any(mask) else np.nan,
        "p90_rel_error": float(np.nanquantile(rel_errors[mask], 0.9)) if np.any(mask) else np.nan,
        "mean_rel_error": float(np.nanmean(rel_errors[mask])) if np.any(mask) else np.nan,
    }


def save_csv(path, rows):
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path, rows):
    energies = np.array([row["energy_keV"] for row in rows], dtype=float)
    iso = np.array([row["iso_median_rel_error"] for row in rows], dtype=float)
    front = np.array([row["front_median_rel_error"] for row in rows], dtype=float)
    ratio = np.array([row["front_to_iso_median_error_ratio"] for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(energies, iso, marker="o", linewidth=1.2, label="Isotropic")
    axes[0].plot(energies, front, marker="o", linewidth=1.2, label="Front")
    axes[0].set_ylabel("Median Relative Error")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].plot(energies, ratio, marker="o", linewidth=1.2, color="#d62728")
    axes[1].set_xlabel("Simulated Energy [keV]")
    axes[1].set_ylabel("Front / Isotropic")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    pdf_path = str(Path(path).with_suffix(".pdf"))
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = ArgumentParser()
    parser.add_argument("--iso-dir", default="sim/det_response/2026_01_08")
    parser.add_argument("--front-dir", default="sim/det_response/2026-03-11")
    parser.add_argument("--out-csv", default="analysis/unfold/paper_files/migration_matrix_stats_comparison.csv")
    parser.add_argument("--out-plot", default="analysis/unfold/paper_files/migration_matrix_stats_comparison.png")
    args = parser.parse_args()

    iso_files = collect_files(args.iso_dir, front=False)
    front_files = collect_files(args.front_dir, front=True)
    common_energies = sorted(set(iso_files) & set(front_files))

    rows = []
    for energy in common_energies:
        iso = summarize_energy(iso_files[energy])
        front = summarize_energy(front_files[energy])
        if iso is None or front is None:
            continue
        rows.append(
            {
                "energy_keV": energy,
                "iso_files": iso["files"],
                "front_files": front["files"],
                "iso_runtime_s": iso["runtime_s"],
                "front_runtime_s": front["runtime_s"],
                "iso_sum_counts": iso["sum_counts"],
                "front_sum_counts": front["sum_counts"],
                "iso_nonzero_bins": iso["nonzero_bins"],
                "front_nonzero_bins": front["nonzero_bins"],
                "iso_median_rel_error": iso["median_rel_error"],
                "front_median_rel_error": front["median_rel_error"],
                "iso_p90_rel_error": iso["p90_rel_error"],
                "front_p90_rel_error": front["p90_rel_error"],
                "front_to_iso_median_error_ratio": front["median_rel_error"] / iso["median_rel_error"],
                "front_to_iso_runtime_ratio": front["runtime_s"] / iso["runtime_s"],
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    save_csv(out_csv, rows)
    save_plot(args.out_plot, rows)
    print(f"Saved {out_csv}")
    print(f"Saved {args.out_plot}")


if __name__ == "__main__":
    main()
