#!/usr/bin/env python3
"""Quantify Figure 7 floor-scan statistics and practical energy binning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.public_data.browser import (  # noqa: E402
    PublicSpectrum,
    load_spectrum,
    query_file_metadata,
    resolve_browser_paths,
)
from webapp.helpers import detector_face_position  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "analysis" / "floor_scan_statistics"
ENERGY_MIN_KEV = 50.0
ENERGY_MAX_KEV = 11400.0
ROUTINE_LIVE_MIN_S = 100.0
ROUTINE_LIVE_MAX_S = 400.0
BIN_WIDTHS_KEV = (1, 2, 5, 10, 20, 25, 50, 100, 200)
ENERGY_BANDS = (
    (50.0, 1000.0, 2.0),
    (1000.0, 3000.0, 10.0),
    (3000.0, 7000.0, 50.0),
    (7000.0, 11400.0, 200.0),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", help="SQLite database (default: env/public bundle)")
    parser.add_argument("--data-root", help="spectrum directory (default: env/public bundle)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def select_floor_scan_files(metadata: pd.DataFrame) -> pd.DataFrame:
    """Recover the original Figure 7 down-facing scan acquisitions.

    The public database also contains later monitoring runs at some of the same
    coordinates. Figure 7 came from the named ``position_scan_3`` through
    ``position_scan_8`` campaigns, so the run-name restriction is essential.
    The remaining predicates reproduce the legacy plotting selection.
    """

    maximum_energy = (
        metadata["calibration_A0"]
        + metadata["calibration_A1"] * 16384
        + metadata["calibration_A1"] / 2
    )
    mask = (
        metadata["run_name"].fillna("").str.match(r"^position_scan_[3-8]_")
        & metadata["shield_name"].eq("collimator30")
        & metadata["coordinate_angle"].eq(0)
        & metadata["coordinate_track"].eq(0)
        & ~metadata["run_description"].fillna("").str.contains(
            "lead", case=False, regex=False
        )
        & metadata["start_time"].ge(0)
        & maximum_energy.ge(ENERGY_MAX_KEV)
    )
    selected = metadata.loc[mask].copy()
    selected["routine_point"] = selected["live_time"].between(
        ROUTINE_LIVE_MIN_S, ROUTINE_LIVE_MAX_S, inclusive="both"
    )
    return selected.sort_values(["coordinate_Rz", "coordinate_Rx", "file_id"])


def fixed_width_histogram(
    spectrum: PublicSpectrum,
    energy_min_keV: float,
    energy_max_keV: float,
    bin_width_keV: float,
) -> pd.DataFrame:
    """Bin calibrated channel counts onto fixed energy edges."""

    if energy_max_keV <= energy_min_keV or bin_width_keV <= 0:
        raise ValueError("energy range and bin width must be positive")
    edges = np.arange(energy_min_keV, energy_max_keV, bin_width_keV)
    edges = np.append(edges, energy_max_keV)
    edges = np.unique(edges)
    counts = np.histogram(
        spectrum.energy_keV, bins=edges, weights=spectrum.counts
    )[0].astype(float)
    widths = np.diff(edges)
    return pd.DataFrame(
        {
            "energy_min_keV": edges[:-1],
            "energy_max_keV": edges[1:],
            "energy_keV": (edges[:-1] + edges[1:]) / 2,
            "bin_width_keV": widths,
            "counts": counts,
            "statistical_error_counts": np.sqrt(counts),
            "rate_counts_per_s": counts / spectrum.live_time,
            "rate_counts_per_s_per_keV": counts / spectrum.live_time / widths,
            "statistical_error_counts_per_s_per_keV": (
                np.sqrt(counts) / spectrum.live_time / widths
            ),
        }
    )


def representative_point(points: pd.DataFrame) -> pd.Series:
    """Select the point closest to robust medians of live time, counts, rate."""

    fields = ["live_time_s", "counts_50_11400_keV", "rate_50_11400_counts_per_s"]
    medians = points[fields].median()
    scales = points[fields].sub(medians).abs().median().replace(0, 1)
    distance = np.sqrt(points[fields].sub(medians).div(scales).pow(2).sum(axis=1))
    return points.loc[distance.idxmin()]


def point_metrics(selected: pd.DataFrame, spectra: dict[int, PublicSpectrum]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metadata in selected.itertuples(index=False):
        spectrum = spectra[int(metadata.file_id)]
        histogram = fixed_width_histogram(
            spectrum, ENERGY_MIN_KEV, ENERGY_MAX_KEV, ENERGY_MAX_KEV - ENERGY_MIN_KEV
        )
        counts = float(histogram["counts"].sum())
        map_z, map_x = detector_face_position(
            float(metadata.coordinate_Rx),
            float(metadata.coordinate_Rz),
            float(metadata.coordinate_Lx),
            float(metadata.coordinate_Lz),
            float(metadata.coordinate_angle),
        )
        rows.append(
            {
                "file_id": int(metadata.file_id),
                "file_name": str(metadata.file_name),
                "run_id": int(metadata.run_id),
                "run_name": str(metadata.run_name),
                "coordinate_id": int(metadata.coordinate_id),
                "detector_face_x_in": map_x,
                "detector_face_z_in": map_z,
                "live_time_s": float(metadata.live_time),
                "counts_50_11400_keV": counts,
                "statistical_error_counts": np.sqrt(counts),
                "rate_50_11400_counts_per_s": counts / float(metadata.live_time),
                "routine_point": bool(metadata.routine_point),
                "calibration_A0_keV": float(metadata.calibration_A0),
                "calibration_A1_keV_per_channel": float(metadata.calibration_A1),
            }
        )
    points = pd.DataFrame(rows)
    points["representative_point"] = False
    routine = points[points["routine_point"]]
    representative = representative_point(routine)
    points.loc[points["file_id"].eq(representative["file_id"]), "representative_point"] = True
    return points.sort_values("file_id").reset_index(drop=True)


def binning_summary(
    routine_points: pd.DataFrame,
    spectra: dict[int, PublicSpectrum],
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for low, high, recommended_width in ENERGY_BANDS:
        for width in BIN_WIDTHS_KEV:
            metrics: list[tuple[float, float, float, float]] = []
            for file_id in routine_points["file_id"].astype(int):
                counts = fixed_width_histogram(
                    spectra[file_id], low, high, float(width)
                )["counts"].to_numpy()
                metrics.append(
                    (
                        float(counts.mean()),
                        float(np.mean(counts > 0)),
                        float(np.mean(counts >= 10)),
                        float(np.mean(counts >= 25)),
                    )
                )
            values = np.asarray(metrics)
            median_mean = float(np.median(values[:, 0]))
            rows.append(
                {
                    "energy_min_keV": low,
                    "energy_max_keV": high,
                    "bin_width_keV": width,
                    "recommended_width": width == recommended_width,
                    "point_count": len(metrics),
                    "p25_mean_counts_per_bin": float(np.percentile(values[:, 0], 25)),
                    "median_mean_counts_per_bin": median_mean,
                    "p75_mean_counts_per_bin": float(np.percentile(values[:, 0], 75)),
                    "median_nonzero_bin_fraction": float(np.median(values[:, 1])),
                    "median_fraction_bins_ge_10_counts": float(np.median(values[:, 2])),
                    "median_fraction_bins_ge_25_counts": float(np.median(values[:, 3])),
                    "poisson_relative_error_at_median_mean": (
                        1 / np.sqrt(median_mean) if median_mean > 0 else np.inf
                    ),
                }
            )
    return pd.DataFrame(rows)


def representative_spectrum_export(
    point: pd.Series, spectrum: PublicSpectrum
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for low, high, width in ENERGY_BANDS:
        frame = fixed_width_histogram(spectrum, low, high, width)
        frame.insert(0, "file_id", int(point["file_id"]))
        frame.insert(1, "file_name", str(point["file_name"]))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def all_spectra_export(
    points: pd.DataFrame, spectra: dict[int, PublicSpectrum]
) -> pd.DataFrame:
    """Return every original scan acquisition with the suggested binning."""

    frames: list[pd.DataFrame] = []
    for point in points.itertuples(index=False):
        spectrum = spectra[int(point.file_id)]
        for low, high, width in ENERGY_BANDS:
            frame = fixed_width_histogram(spectrum, low, high, width)
            frame.insert(0, "file_id", int(point.file_id))
            frame.insert(1, "file_name", str(point.file_name))
            frame.insert(2, "run_id", int(point.run_id))
            frame.insert(3, "coordinate_id", int(point.coordinate_id))
            frame.insert(4, "detector_face_x_in", float(point.detector_face_x_in))
            frame.insert(5, "detector_face_z_in", float(point.detector_face_z_in))
            frame.insert(6, "live_time_s", float(point.live_time_s))
            frame.insert(7, "routine_point", bool(point.routine_point))
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_point_statistics(points: pd.DataFrame, output_dir: Path) -> None:
    routine = points[points["routine_point"]]
    representative = routine[routine["representative_point"]].iloc[0]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    scatter = axes[0].scatter(
        routine["detector_face_z_in"],
        routine["detector_face_x_in"],
        c=routine["rate_50_11400_counts_per_s"],
        s=35,
        cmap="viridis",
    )
    axes[0].scatter(
        representative["detector_face_z_in"], representative["detector_face_x_in"],
        marker="*", s=180, facecolor="white", edgecolor="black", label="representative"
    )
    axes[0].invert_yaxis()
    axes[0].set(xlabel="z [in]", ylabel="x [in]", title="Routine floor-scan points")
    axes[0].legend(loc="best")
    figure.colorbar(scatter, ax=axes[0], label="50–11400 keV count rate [s⁻¹]")

    axes[1].hist(routine["counts_50_11400_keV"], bins=20, color="#4472c4")
    axes[1].axvline(routine["counts_50_11400_keV"].median(), color="black", ls="--")
    axes[1].set(xlabel="Counts per acquisition", ylabel="Points", title="Per-point counts")

    axes[2].hist(routine["live_time_s"], bins=20, color="#70ad47")
    axes[2].axvline(routine["live_time_s"].median(), color="black", ls="--")
    axes[2].set(xlabel="Live time [s]", ylabel="Points", title="Per-point live time")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        figure.savefig(
            output_dir / f"floor_scan_point_statistics.{suffix}",
            dpi=180,
            metadata=metadata,
        )
    plt.close(figure)


def plot_representative_spectrum(
    point: pd.Series, spectrum: PublicSpectrum, output_dir: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=False)
    for axis, (low, high, width) in zip(axes.flat, ENERGY_BANDS):
        frame = fixed_width_histogram(spectrum, low, high, width)
        axis.step(
            frame["energy_keV"], frame["rate_counts_per_s_per_keV"], where="mid"
        )
        axis.set_yscale("log")
        axis.set_title(f"{low:g}–{high:g} keV, {width:g} keV bins")
        axis.set_xlabel("Energy [keV]")
        axis.set_ylabel("Counts s⁻¹ keV⁻¹")
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"Representative routine point: {point['file_name']} "
        f"({point['live_time_s']:.1f} s, {point['counts_50_11400_keV']:.0f} counts)"
    )
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        figure.savefig(
            output_dir / f"representative_point_adaptive_binning.{suffix}",
            dpi=180,
            metadata=metadata,
        )
    plt.close(figure)


def plot_quantile_spectra(
    routine: pd.DataFrame,
    spectra: dict[int, PublicSpectrum],
    output_dir: Path,
) -> None:
    ordered = routine.sort_values("counts_50_11400_keV").reset_index(drop=True)
    quantiles = (0.1, 0.25, 0.5, 0.75, 0.9)
    selections = [ordered.iloc[round(q * (len(ordered) - 1))] for q in quantiles]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    for axis, (low, high, width) in zip(axes.flat, ENERGY_BANDS):
        for quantile, point in zip(quantiles, selections):
            frame = fixed_width_histogram(
                spectra[int(point["file_id"])], low, high, width
            )
            axis.step(
                frame["energy_keV"], frame["rate_counts_per_s_per_keV"],
                where="mid", label=f"Q{quantile:g}: {point['file_name']}"
            )
        axis.set_yscale("log")
        axis.set_title(f"{low:g}–{high:g} keV, {width:g} keV bins")
        axis.set_xlabel("Energy [keV]")
        axis.set_ylabel("Counts s⁻¹ keV⁻¹")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    figure.suptitle("Routine floor-scan spectra across total-count quantiles")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        figure.savefig(
            output_dir / f"floor_scan_spectrum_quantiles.{suffix}",
            dpi=180,
            metadata=metadata,
        )
    plt.close(figure)


def write_summary(
    points: pd.DataFrame,
    binning: pd.DataFrame,
    representative: pd.Series,
    output_dir: Path,
) -> None:
    routine = points[points["routine_point"]]
    extended = points[~points["routine_point"]]
    recommendations = binning[binning["recommended_width"]]
    payload = {
        "selection": {
            "all_original_scan_acquisitions": len(points),
            "unique_coordinates": int(points["coordinate_id"].nunique()),
            "routine_acquisitions_100_to_400_s": len(routine),
            "routine_unique_coordinates": int(routine["coordinate_id"].nunique()),
            "extended_followup_acquisitions": len(extended),
        },
        "routine_point_statistics": {
            "median_live_time_s": float(routine["live_time_s"].median()),
            "median_counts_50_11400_keV": float(routine["counts_50_11400_keV"].median()),
            "median_rate_counts_per_s": float(
                routine["rate_50_11400_counts_per_s"].median()
            ),
            "count_range": [
                float(routine["counts_50_11400_keV"].min()),
                float(routine["counts_50_11400_keV"].max()),
            ],
        },
        "representative_point": representative.to_dict(),
        "recommended_binning": recommendations[
            [
                "energy_min_keV", "energy_max_keV", "bin_width_keV",
                "median_mean_counts_per_bin", "median_fraction_bins_ge_10_counts",
                "median_nonzero_bin_fraction",
            ]
        ].to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
    )
    try:
        relative_output = output_dir.relative_to(REPO_ROOT)
    except ValueError:
        command = (
            ".venv/bin/python scripts/analyze_floor_scan_statistics.py "
            f"--output-dir {output_dir}"
        )
    else:
        command = ".venv/bin/python scripts/analyze_floor_scan_statistics.py"
        if relative_output != DEFAULT_OUTPUT.relative_to(REPO_ROOT):
            command += f" --output-dir {relative_output}"
    lines = [
        "# Figure 7 floor-scan statistics", "",
        "Generated with:", "",
        "```bash", command, "```", "",
        f"- Original down-facing scan acquisitions: {len(points)} at {points['coordinate_id'].nunique()} coordinates.",
        f"- Routine 100–400 s acquisitions: {len(routine)} at {routine['coordinate_id'].nunique()} coordinates.",
        f"- Median routine live time: {routine['live_time_s'].median():.2f} s.",
        f"- Median routine 50–11400 keV counts: {routine['counts_50_11400_keV'].median():.0f}.",
        f"- Median routine rate: {routine['rate_50_11400_counts_per_s'].median():.2f} counts/s.",
        f"- Representative point: `{representative['file_name']}` (file {int(representative['file_id'])}), "
        f"{representative['live_time_s']:.2f} s and {representative['counts_50_11400_keV']:.0f} counts.",
        "- These spectra measure the downward-collimated component and should not be interpreted as an orientation-independent ambient flux.",
        "", "Suggested display binning for a typical point:", "",
        "| Energy range | Bin width | Median mean counts/bin | Median nonzero fraction |",
        "|---|---:|---:|---:|",
    ]
    for row in recommendations.itertuples(index=False):
        lines.append(
            f"| {row.energy_min_keV:g}–{row.energy_max_keV:g} keV | "
            f"{row.bin_width_keV:g} keV | {row.median_mean_counts_per_bin:.1f} | "
            f"{row.median_nonzero_bin_fraction:.2f} |"
        )
    lines += [
        "", "These widths are presentation defaults, not a reason to discard native channel counts.",
        "The 7–11.4 MeV region remains sparse even at 100–200 keV per bin; retain Poisson errors.",
        "[Download all individual acquisitions with these widths](floor_scan_spectra.csv.gz).",
        "The [point manifest](floor_scan_points.csv) and [full binning study](floor_scan_binning_summary.csv) are also available.",
        "", "## Review plots", "",
        "![Point statistics](floor_scan_point_statistics.png)", "",
        "![Representative spectrum](representative_point_adaptive_binning.png)", "",
        "![Spectrum quantiles](floor_scan_spectrum_quantiles.png)",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = resolve_browser_paths(db_path=args.db, data_root=args.data_root)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = select_floor_scan_files(query_file_metadata(paths.db_path))
    spectra = {
        int(file_id): load_spectrum(
            int(file_id), db_path=paths.db_path, data_root=paths.data_root
        )
        for file_id in selected["file_id"]
    }
    points = point_metrics(selected, spectra)
    routine = points[points["routine_point"]]
    representative = points[points["representative_point"]].iloc[0]
    summary = binning_summary(routine, spectra)
    representative_export = representative_spectrum_export(
        representative, spectra[int(representative["file_id"])]
    )
    spectra_export = all_spectra_export(points, spectra)

    points.to_csv(output_dir / "floor_scan_points.csv", index=False)
    summary.to_csv(output_dir / "floor_scan_binning_summary.csv", index=False)
    representative_export.to_csv(
        output_dir / "representative_point_spectrum.csv", index=False
    )
    spectra_export.to_csv(
        output_dir / "floor_scan_spectra.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    plot_point_statistics(points, output_dir)
    plot_representative_spectrum(
        representative, spectra[int(representative["file_id"])], output_dir
    )
    plot_quantile_spectra(routine, spectra, output_dir)
    write_summary(points, summary, representative, output_dir)

    print(f"Wrote Figure 7 floor-scan statistics to {output_dir}")
    print(
        f"Routine points: {len(routine)}; median live time "
        f"{routine['live_time_s'].median():.2f} s; median counts "
        f"{routine['counts_50_11400_keV'].median():.0f}"
    )
    print(
        f"Representative: {representative['file_name']} "
        f"(file {int(representative['file_id'])})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
