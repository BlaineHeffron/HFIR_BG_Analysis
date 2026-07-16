#!/usr/bin/env python3
"""
Plot collimator effectiveness using corrected isotropic and front-face simulations.

Outputs source-area-normalized photopeak rates (Hz/mm^2) vs gamma energy.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot


ISO_PATTERN = re.compile(r"^(\d+)_keV_(\d+)_Phys\.root$")
FRONT_PATTERN = re.compile(r"^(\d+)_keV_dir_front_(\d+)_Phys\.root$")

HIST_PATHS = ("GeEnergyPlugin/hGeEnergy", "miscAccum/hGeEnergy")
RUNTIME_PATHS = ("accumulated/runtime", "miscAccum/runtime")

# Source surface areas used by LeadCollimatorThrower for 1 Hz/mm^2 flux normalization.
# These were obtained from throws/runtime in reference production files.
DEFAULT_ISO_AREA_MM2 = 86969.96676382126
DEFAULT_FRONT_AREA_MM2 = 38707.56308487983


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--iso-dir",
        default="sim/det_response/2026_01_08",
        help="Directory with isotropic files (<E>_keV_<copy>_Phys.root).",
    )
    p.add_argument(
        "--front-dir",
        default="sim/det_response/2026-04-02/2026-04-02",
        help="Directory with front files (<E>_keV_dir_front_<copy>_Phys.root).",
    )
    p.add_argument(
        "--peak-width-fraction",
        type=float,
        default=0.0025,
        help="Photopeak integration half-width as fraction of peak energy (default: 0.25%%).",
    )
    p.add_argument(
        "--min-peak-half-width-kev",
        type=float,
        default=3.0,
        help="Minimum photopeak half-width in keV for low-energy stability.",
    )
    p.add_argument(
        "--iso-area-mm2",
        type=float,
        default=DEFAULT_ISO_AREA_MM2,
        help="Isotropic source surface area for normalization.",
    )
    p.add_argument(
        "--front-area-mm2",
        type=float,
        default=DEFAULT_FRONT_AREA_MM2,
        help="Front source surface area for normalization.",
    )
    p.add_argument(
        "--out-pdf",
        default="/home/blaine/projects/HFIRBG/paper/figures/pdf/collimator_effectiveness.pdf",
        help="Output PDF path.",
    )
    p.add_argument(
        "--out-png",
        default="analysis/unfold/paper_files/collimator_effectiveness_corrected.png",
        help="Output PNG path for quick checking.",
    )
    p.add_argument(
        "--out-csv",
        default="analysis/unfold/paper_files/collimator_effectiveness_corrected.csv",
        help="Output CSV with plotted values.",
    )
    p.add_argument(
        "--drop-last-point",
        action="store_true",
        default=True,
        help="Drop the highest-energy point before plotting/export to avoid edge artifacts.",
    )
    p.add_argument(
        "--keep-last-point",
        action="store_true",
        help="Keep the highest-energy point even if it looks like an edge artifact.",
    )
    return p.parse_args()


def try_read(file: uproot.ReadOnlyFile, paths: Tuple[str, ...]):
    for path in paths:
        try:
            return file[path]
        except Exception:
            continue
    return None


def read_runtime(file: uproot.ReadOnlyFile) -> float:
    obj = try_read(file, RUNTIME_PATHS)
    if obj is None:
        raise RuntimeError("Missing runtime object")
    for attr in ("fVal", "value"):
        try:
            val = obj.member(attr) if hasattr(obj, "member") else getattr(obj, attr)
            return float(val)
        except Exception:
            continue
    raise RuntimeError("Could not decode runtime value")


def read_hist(file: uproot.ReadOnlyFile) -> Tuple[np.ndarray, np.ndarray]:
    hist = try_read(file, HIST_PATHS)
    if hist is None:
        raise RuntimeError("Missing Ge energy histogram")
    return hist.values(), hist.axis().edges()


def collect_by_energy(directory: Path, pattern: re.Pattern) -> Dict[int, List[Path]]:
    grouped: Dict[int, List[Path]] = {}
    for f in directory.iterdir():
        m = pattern.match(f.name)
        if not m:
            continue
        e = int(m.group(1))
        grouped.setdefault(e, []).append(f)
    for e in grouped:
        grouped[e].sort(key=lambda p: int(pattern.match(p.name).group(2)))  # type: ignore[arg-type]
    return grouped


def integrate_photopeak(
    values: np.ndarray,
    edges_mev: np.ndarray,
    energy_kev: int,
    width_frac: float,
    min_half_width_kev: float,
) -> float:
    centers_kev = 1000.0 * (edges_mev[:-1] + edges_mev[1:]) / 2.0
    e = float(energy_kev)
    half_width = max(e * width_frac, min_half_width_kev)
    lo = e - half_width
    hi = e + half_width
    mask = (centers_kev >= lo) & (centers_kev <= hi)
    if not np.any(mask):
        idx = int(np.argmin(np.abs(centers_kev - e)))
        return float(values[idx])
    return float(np.sum(values[mask]))


def summarize_case(
    grouped_files: Dict[int, List[Path]],
    area_mm2: float,
    width_frac: float,
    min_half_width_kev: float,
) -> Dict[int, Tuple[float, float]]:
    """
    Returns mapping energy_keV -> (norm_rate_hz_per_mm2, norm_rate_err_hz_per_mm2).
    """
    out: Dict[int, Tuple[float, float]] = {}
    for energy_kev, files in grouped_files.items():
        peak_counts = 0.0
        runtime = 0.0
        for fp in files:
            with uproot.open(fp) as f:
                vals, edges = read_hist(f)
                peak_counts += integrate_photopeak(
                    vals, edges, energy_kev, width_frac, min_half_width_kev
                )
                runtime += read_runtime(f)
        if runtime <= 0:
            continue
        rate = peak_counts / runtime
        # Poisson counting uncertainty on integrated peak counts.
        rate_err = np.sqrt(max(peak_counts, 0.0)) / runtime
        out[energy_kev] = (rate / area_mm2, rate_err / area_mm2)
    return out


def write_csv(
    out_csv: Path,
    common: List[int],
    iso_data: Dict[int, Tuple[float, float]],
    front_data: Dict[int, Tuple[float, float]],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("energy_keV,isotropic_rate_hz_per_mm2,isotropic_err,front_rate_hz_per_mm2,front_err,front_over_iso\n")
        for e in common:
            iso, iso_err = iso_data[e]
            front, front_err = front_data[e]
            ratio = front / iso if iso > 0 else np.nan
            f.write(f"{e},{iso:.10e},{iso_err:.10e},{front:.10e},{front_err:.10e},{ratio:.10e}\n")


def plot(
    out_pdf: Path,
    out_png: Path,
    common: List[int],
    iso_data: Dict[int, Tuple[float, float]],
    front_data: Dict[int, Tuple[float, float]],
) -> None:
    x = np.array(common, dtype=float)
    y_iso = np.array([iso_data[e][0] for e in common], dtype=float)
    y_iso_err = np.array([iso_data[e][1] for e in common], dtype=float)
    y_front = np.array([front_data[e][0] for e in common], dtype=float)
    y_front_err = np.array([front_data[e][1] for e in common], dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    ax.errorbar(
        x,
        y_front,
        yerr=y_front_err,
        fmt="o",
        markersize=3.5,
        linewidth=1.0,
        color="tab:blue",
        label="front face",
    )
    ax.errorbar(
        x,
        y_iso,
        yerr=y_iso_err,
        fmt="o",
        markersize=3.5,
        linewidth=1.0,
        color="orange",
        label="isotropic",
    )
    ax.set_yscale("log")
    ax.set_xlabel("energy [keV]")
    ax.set_ylabel("rate [Hz/mm$^2$]")
    ax.set_xlim(max(0.0, x.min() - 60.0), x.max() * 1.02)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()

    iso_dir = Path(args.iso_dir).expanduser().resolve()
    front_dir = Path(args.front_dir).expanduser().resolve()
    if not iso_dir.exists():
        raise FileNotFoundError(f"Isotropic directory not found: {iso_dir}")
    if not front_dir.exists():
        raise FileNotFoundError(f"Front directory not found: {front_dir}")

    iso_files = collect_by_energy(iso_dir, ISO_PATTERN)
    front_files = collect_by_energy(front_dir, FRONT_PATTERN)

    common = sorted(set(iso_files) & set(front_files))
    if not common:
        raise RuntimeError("No common energies found between isotropic and front datasets.")

    iso_data = summarize_case(
        iso_files, args.iso_area_mm2, args.peak_width_fraction, args.min_peak_half_width_kev
    )
    front_data = summarize_case(
        front_files, args.front_area_mm2, args.peak_width_fraction, args.min_peak_half_width_kev
    )
    common = [e for e in common if e in iso_data and e in front_data]
    if not common:
        raise RuntimeError("No valid common energies after processing files.")
    if args.drop_last_point and not args.keep_last_point and len(common) > 1:
        common = common[:-1]

    out_pdf = Path(args.out_pdf).expanduser().resolve()
    out_png = Path(args.out_png).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()

    plot(out_pdf, out_png, common, iso_data, front_data)
    write_csv(out_csv, common, iso_data, front_data)

    ratios = np.array([front_data[e][0] / iso_data[e][0] for e in common if iso_data[e][0] > 0], dtype=float)
    print(f"Processed {len(common)} common energies from {common[0]} to {common[-1]} keV.")
    print(f"Front/isotropic median ratio: {np.nanmedian(ratios):.2f}")
    print(f"Output PDF: {out_pdf}")
    print(f"Output PNG: {out_png}")
    print(f"Output CSV: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
