#!/usr/bin/env python3
"""Band integrals: published CSV vs browser-input unfolds."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import uproot

REPO = Path(__file__).resolve().parents[1]
PUB = REPO / "analysis" / "unfold" / "paper_files"
if not (PUB / "unfolded_spectrum_isotropic_03_CYCLE461_DOWN_FACING_OVERNIGHT.csv").exists():
    PUB = Path("/home/blaine/projects/HFIRBG/paper/arxiv_submission/anc/spectra")

LOCS = [
    ("01", "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN", "MIF on"),
    ("02", "MIF_BOX_AT_REACTOR_RXOFF", "MIF off"),
    ("03", "CYCLE461_DOWN_FACING_OVERNIGHT", "Shield Center"),
    ("04", "HB4_DOWN_OVERNIGHT_1", "HB4"),
    ("05", "EAST_FACE_18", "East 18"),
    ("06", "EAST_FACE_1", "East 1"),
]
BANDS = [(200.0, 1000.0, "0.2-1"), (1000.0, 6000.0, "1-6"), (6000.0, 11500.0, "6-11.5"), (40.0, 12000.0, "40-12000")]


def band_sum(energy: np.ndarray, flux: np.ndarray, lo: float, hi: float) -> float:
    mask = (energy >= lo) & (energy < hi)
    return float(np.sum(flux[mask]))


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    e, f = [], []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            e.append(float(row["Energy_keV"]))
            f.append(float(row["Flux_Hz_per_mm2_per_keV"]))
    return np.asarray(e), np.asarray(f)


def load_root(path: Path) -> tuple[np.ndarray, np.ndarray]:
    hist = uproot.open(path)["UnfoldedEnergy"]
    edges = hist.axis().edges()
    return 0.5 * (edges[:-1] + edges[1:]), hist.values()


def main() -> None:
    print(
        "scenario location band published v121_browser v122_browser "
        "a_rel(v121/pub-1) b_rel(v122/v121-1) total_rel(v122/pub-1)"
    )
    for scenario, stem in (("isotropic", "iso"), ("front", "front")):
        for number, fname, label in LOCS:
            pub = PUB / f"unfolded_spectrum_{scenario}_{number}_{fname}.csv"
            r121 = (
                REPO
                / "analysis"
                / "unfold"
                / f"browser_v1.2.1_{stem}"
                / f"{fname}_unfold_results.root"
            )
            r122 = (
                REPO
                / "analysis"
                / "unfold"
                / f"browser_v1.2.2_{stem}"
                / f"{fname}_unfold_results.root"
            )
            if not pub.exists() or not r121.exists() or not r122.exists():
                print("MISSING", scenario, label, pub.exists(), r121.exists(), r122.exists())
                continue
            e0, f0 = load_csv(pub)
            e1, f1 = load_root(r121)
            e2, f2 = load_root(r122)
            for lo, hi, bname in BANDS:
                p = band_sum(e0, f0, lo, hi)
                a = band_sum(e1, f1, lo, hi)
                b = band_sum(e2, f2, lo, hi)
                a_rel = a / p - 1 if p else float("nan")
                b_rel = b / a - 1 if a else float("nan")
                t_rel = b / p - 1 if p else float("nan")
                print(
                    f"{scenario:10} {label:14} {bname:9} "
                    f"{p:.6g} {a:.6g} {b:.6g} "
                    f"{a_rel:+.4%} {b_rel:+.4%} {t_rel:+.4%}"
                )


if __name__ == "__main__":
    main()
