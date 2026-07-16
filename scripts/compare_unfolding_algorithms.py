#!/usr/bin/env python3
import csv
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

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

CASES = {
    "rl_isotropic": {
        "algorithm": "Richardson-Lucy",
        "response": "isotropic",
        "label": "RL + isotropic",
        "dir": "/home/blaine/projects/HFIR_BG_Analysis/analysis/unfold/sumita",
    },
    "rl_front": {
        "algorithm": "Richardson-Lucy",
        "response": "front",
        "label": "RL + front",
        "dir": "/home/blaine/projects/HFIR_BG_Analysis/analysis/unfold/front",
    },
    "poisson_pgd_isotropic": {
        "algorithm": "Poisson-PGD",
        "response": "isotropic",
        "label": "Poisson-PGD + isotropic",
        "dir": "/home/blaine/projects/HFIR_BG_Analysis/analysis/unfold/poisson_pgd_isotropic",
    },
    "poisson_pgd_front": {
        "algorithm": "Poisson-PGD",
        "response": "front",
        "label": "Poisson-PGD + front",
        "dir": "/home/blaine/projects/HFIR_BG_Analysis/analysis/unfold/poisson_pgd_front",
    },
}

OUTDIR = Path("/home/blaine/projects/HFIR_BG_Analysis/analysis/unfold/algorithm_comparison")


def load_hist(file_path, name):
    with uproot.open(file_path) as root_file:
        values, edges = root_file[name].to_numpy()
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, values.astype(float), edges.astype(float)


def load_result(case_key, fname):
    path = Path(CASES[case_key]["dir"]) / f"{fname}_unfold_results.root"
    if not path.exists():
        raise FileNotFoundError(path)
    x_unf, y_unf, edges_unf = load_hist(path, "UnfoldedEnergy")
    x_meas, y_meas, _ = load_hist(path, "Measured")
    x_ref, y_ref, _ = load_hist(path, "FoldedBack")
    return {
        "path": str(path),
        "x": x_unf,
        "y": y_unf,
        "edges": edges_unf,
        "x_meas": x_meas,
        "y_meas": y_meas,
        "x_ref": x_ref,
        "y_ref": y_ref,
    }


def integrate_band(x, y, edges, low, high):
    mask = (x >= low) & (x < high)
    widths = np.diff(edges)[mask]
    return float(np.sum(y[mask] * widths))


def calc_chi2(measured, refolded):
    denom = np.where(refolded > 0.0, refolded, 1.0)
    mask = refolded > 0.0
    if not np.any(mask):
        return float("nan")
    return float(np.sum(((measured[mask] - refolded[mask]) ** 2) / denom[mask]))


def low_band_ratio(a, b, low=50.0, high=2000.0):
    mask = (a["x"] >= low) & (a["x"] < high)
    if not np.any(mask):
        return float("nan")
    safe = np.maximum(b["y"][mask], 1e-30)
    ratio = a["y"][mask] / safe
    return float(np.median(ratio))


def low_band_log_distance(a, b, low=50.0, high=2000.0):
    mask = (a["x"] >= low) & (a["x"] < high)
    if not np.any(mask):
        return float("nan")
    safe_a = np.maximum(a["y"][mask], 1e-30)
    safe_b = np.maximum(b["y"][mask], 1e-30)
    return float(np.median(np.abs(np.log10(safe_a / safe_b))))


def summarize_case(result):
    low = integrate_band(result["x"], result["y"], result["edges"], 50.0, 2000.0)
    mid = integrate_band(result["x"], result["y"], result["edges"], 2000.0, 9000.0)
    high = integrate_band(result["x"], result["y"], result["edges"], 9000.0, 11500.0)
    total = integrate_band(result["x"], result["y"], result["edges"], 50.0, 11500.0)
    return {
        "low_flux": low,
        "mid_flux": mid,
        "high_flux": high,
        "total_flux": total,
        "low_fraction": low / total if total > 0.0 else float("nan"),
        "low_high_ratio": low / high if high > 0.0 else float("nan"),
        "refold_chi2": calc_chi2(result["y_meas"], result["y_ref"]),
    }


def save_figure(fig, stem):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    png = OUTDIR / f"{stem}.png"
    pdf = OUTDIR / f"{stem}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def plot_response_overlay(response, results):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    for idx, fname in enumerate(FNAMES):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        rl = results[("Richardson-Lucy", response, fname)]
        alt = results[("Poisson-PGD", response, fname)]
        ax.step(rl["x"], rl["y"], where="mid", linewidth=1.2, color="#1f77b4", label="RL")
        ax.step(alt["x"], alt["y"], where="mid", linewidth=1.2, color="#d62728", label="Poisson-PGD")
        ax.set_title(ALIAS_MAP[fname], fontsize=12)
        ax.set_xlim(50, 11500)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Unfolding algorithm comparison with {response} response matrix", fontsize=16)
    for ax in axes[:, 0]:
        ax.set_ylabel("Flux [Hz/mm$^2$/keV]")
    for ax in axes[-1, :]:
        ax.set_xlabel("Energy [keV]")
    plt.tight_layout()
    save_figure(fig, f"algorithm_overlay_{response}")


def plot_low_fraction(metrics):
    labels = [ALIAS_MAP[f] for f in FNAMES]
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(14, 7))
    series = [
        ("RL iso", [metrics[("Richardson-Lucy", "isotropic", f)]["low_fraction"] for f in FNAMES], "#1f77b4"),
        ("PGD iso", [metrics[("Poisson-PGD", "isotropic", f)]["low_fraction"] for f in FNAMES], "#6baed6"),
        ("RL front", [metrics[("Richardson-Lucy", "front", f)]["low_fraction"] for f in FNAMES], "#d62728"),
        ("PGD front", [metrics[("Poisson-PGD", "front", f)]["low_fraction"] for f in FNAMES], "#fb6a4a"),
    ]
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for (label, values, color), offset in zip(series, offsets):
        ax.bar(x + offset * width, values, width=width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Low-energy fraction (50-2000 keV / total)")
    ax.set_title("Low-energy content by unfolding algorithm and response model")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    save_figure(fig, "low_energy_fraction_comparison")


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    results = {}
    metrics = {}
    metric_rows = []
    for case_key, cfg in CASES.items():
        for fname in FNAMES:
            result = load_result(case_key, fname)
            results[(cfg["algorithm"], cfg["response"], fname)] = result
            summary = summarize_case(result)
            metrics[(cfg["algorithm"], cfg["response"], fname)] = summary
            metric_rows.append({
                "algorithm": cfg["algorithm"],
                "response": cfg["response"],
                "measurement": ALIAS_MAP[fname],
                "filename": fname,
                "low_flux_50_2000_keV": summary["low_flux"],
                "mid_flux_2000_9000_keV": summary["mid_flux"],
                "high_flux_9000_11500_keV": summary["high_flux"],
                "total_flux_50_11500_keV": summary["total_flux"],
                "low_fraction": summary["low_fraction"],
                "low_high_ratio": summary["low_high_ratio"],
                "refold_chi2": summary["refold_chi2"],
                "root_file": result["path"],
            })

    write_csv(
        OUTDIR / "algorithm_metrics.csv",
        [
            "algorithm", "response", "measurement", "filename", "low_flux_50_2000_keV",
            "mid_flux_2000_9000_keV", "high_flux_9000_11500_keV", "total_flux_50_11500_keV",
            "low_fraction", "low_high_ratio", "refold_chi2", "root_file",
        ],
        metric_rows,
    )

    comparison_rows = []
    for response in ["isotropic", "front"]:
        for fname in FNAMES:
            rl = results[("Richardson-Lucy", response, fname)]
            alt = results[("Poisson-PGD", response, fname)]
            rl_metrics = metrics[("Richardson-Lucy", response, fname)]
            alt_metrics = metrics[("Poisson-PGD", response, fname)]
            comparison_rows.append({
                "response": response,
                "measurement": ALIAS_MAP[fname],
                "filename": fname,
                "low_flux_ratio_alt_over_rl": alt_metrics["low_flux"] / rl_metrics["low_flux"] if rl_metrics["low_flux"] > 0.0 else float("nan"),
                "total_flux_ratio_alt_over_rl": alt_metrics["total_flux"] / rl_metrics["total_flux"] if rl_metrics["total_flux"] > 0.0 else float("nan"),
                "low_fraction_rl": rl_metrics["low_fraction"],
                "low_fraction_alt": alt_metrics["low_fraction"],
                "refold_chi2_rl": rl_metrics["refold_chi2"],
                "refold_chi2_alt": alt_metrics["refold_chi2"],
                "median_bin_ratio_alt_over_rl_50_2000": low_band_ratio(alt, rl),
                "median_abs_log10_bin_distance_50_2000": low_band_log_distance(alt, rl),
            })

    write_csv(
        OUTDIR / "algorithm_pairwise_comparison.csv",
        [
            "response", "measurement", "filename", "low_flux_ratio_alt_over_rl",
            "total_flux_ratio_alt_over_rl", "low_fraction_rl", "low_fraction_alt",
            "refold_chi2_rl", "refold_chi2_alt", "median_bin_ratio_alt_over_rl_50_2000",
            "median_abs_log10_bin_distance_50_2000",
        ],
        comparison_rows,
    )

    plot_response_overlay("isotropic", results)
    plot_response_overlay("front", results)
    plot_low_fraction(metrics)

    mif_iso = next(row for row in comparison_rows if row["response"] == "isotropic" and row["filename"] == "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN")
    mif_front = next(row for row in comparison_rows if row["response"] == "front" and row["filename"] == "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN")
    hb4_iso = next(row for row in comparison_rows if row["response"] == "isotropic" and row["filename"] == "HB4_DOWN_OVERNIGHT_1")

    isotropic_mif_still_suppressed = mif_iso["low_fraction_rl"] < 0.1 and mif_iso["low_fraction_alt"] < 0.1
    isotropic_hb4_shifted_low = hb4_iso["low_flux_ratio_alt_over_rl"] < 0.3
    front_mif_stable = abs(mif_front["low_fraction_alt"] - mif_front["low_fraction_rl"]) < 0.02
    front_restores_low_energy = mif_front["low_fraction_rl"] > mif_iso["low_fraction_rl"] and mif_front["low_fraction_alt"] > mif_iso["low_fraction_alt"]

    if isotropic_mif_still_suppressed and front_mif_stable and front_restores_low_energy:
        headline = "The low-energy dip is not unique to Richardson-Lucy: it remains present for the isotropic MIF unfold with the alternate penalized Poisson solver, while the front-face results stay stable across algorithms. That points to the isotropic response assumption as the main source of the artifact, with solver choice affecting the severity once the response model is mismatched."
    elif isotropic_hb4_shifted_low:
        headline = "Both the response model and the unfolding solver influence the low-energy result in the directional cases: the dip remains with the alternate penalized Poisson solver, but its depth changes enough that the current isotropic unfold should not be treated as algorithm-independent."
    else:
        headline = "The alternate unfold changes the low-energy content enough that the solver choice still matters, so the current dip cannot yet be assigned solely to the response matrix."

    report_path = OUTDIR / "unfolding_algorithm_comparison.md"
    with open(report_path, "w") as handle:
        handle.write("# Unfolding algorithm comparison\n\n")
        handle.write(headline + "\n\n")
        handle.write("## Data used\n\n")
        handle.write("- Spectra: the six paper unfold inputs (MIF reactor-on/off, Shield Center, HB4, PROSPECT East 1/2).\n")
        handle.write("- Response models: isotropic and front-face migration matrices already used for the paper.\n")
        handle.write("- Algorithms: existing Richardson-Lucy (RL) and a new non-negative penalized Poisson mirror-descent solver with first-difference Tikhonov regularization.\n\n")
        handle.write("## Key metrics\n\n")
        handle.write("| Response | Measurement | RL low frac | Alt low frac | Alt/RL low flux | RL chi2 | Alt chi2 | Median |log10(Alt/RL)| low band |\n")
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in comparison_rows:
            handle.write(
                f"| {row['response']} | {row['measurement']} | {row['low_fraction_rl']:.4f} | {row['low_fraction_alt']:.4f} | {row['low_flux_ratio_alt_over_rl']:.3f} | {row['refold_chi2_rl']:.2f} | {row['refold_chi2_alt']:.2f} | {row['median_abs_log10_bin_distance_50_2000']:.3f} |\n"
            )
        handle.write("\n## Interpretation\n\n")
        handle.write(f"- MIF reactor-on, isotropic matrix: Alt/RL low-flux ratio = {mif_iso['low_flux_ratio_alt_over_rl']:.3f}; low-band distance = {mif_iso['median_abs_log10_bin_distance_50_2000']:.3f}.\n")
        handle.write(f"- HB4, isotropic matrix: Alt/RL low-flux ratio = {hb4_iso['low_flux_ratio_alt_over_rl']:.3f}; low-band distance = {hb4_iso['median_abs_log10_bin_distance_50_2000']:.3f}.\n")
        handle.write(f"- MIF reactor-on, front matrix: RL low fraction = {mif_front['low_fraction_rl']:.4f}; Alt low fraction = {mif_front['low_fraction_alt']:.4f}.\n")
        handle.write("- If both algorithms agree within about a factor of two in the 50-2000 keV band while the front-face matrix restores the missing low-energy flux, that points to the migration matrix assumptions rather than an RL-specific pathology.\n")
        handle.write("- If the alternate penalized Poisson result diverges strongly from RL while preserving a comparable refold chi-squared, then solver bias remains a live concern.\n\n")
        handle.write("## Files\n\n")
        handle.write(f"- Metrics CSV: `{OUTDIR / 'algorithm_metrics.csv'}`\n")
        handle.write(f"- Pairwise CSV: `{OUTDIR / 'algorithm_pairwise_comparison.csv'}`\n")
        handle.write(f"- Isotropic overlay: `{OUTDIR / 'algorithm_overlay_isotropic.pdf'}`\n")
        handle.write(f"- Front overlay: `{OUTDIR / 'algorithm_overlay_front.pdf'}`\n")
        handle.write(f"- Low-energy bar chart: `{OUTDIR / 'low_energy_fraction_comparison.pdf'}`\n")

    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
