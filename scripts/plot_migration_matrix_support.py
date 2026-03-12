#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot


def load_matrix(path):
    root_file = uproot.open(path)
    energies = root_file["hEnergies"].values()
    mask = root_file["hSimulatedMask"].values() > 0
    matrix = root_file["MdetgenMC"].values()
    column_integral = matrix.sum(axis=0)
    return energies, mask, column_integral


def nearest_supported_distance(energies, mask):
    supported = energies[mask]
    if len(supported) == 0:
        return np.full_like(energies, np.nan, dtype=float)
    idx = np.searchsorted(supported, energies)
    left_idx = np.clip(idx - 1, 0, len(supported) - 1)
    right_idx = np.clip(idx, 0, len(supported) - 1)
    left = np.abs(energies - supported[left_idx])
    right = np.abs(energies - supported[right_idx])
    return np.minimum(left, right)


def main():
    parser = ArgumentParser()
    parser.add_argument("--iso", default="scripts/private/migration_matrix.root")
    parser.add_argument("--front", default="scripts/private/migration_matrix_front.root")
    parser.add_argument("--out", default="analysis/unfold/paper_files/migration_matrix_support_comparison.png")
    args = parser.parse_args()

    iso_e, iso_mask, iso_col = load_matrix(args.iso)
    front_e, front_mask, front_col = load_matrix(args.front)

    if len(iso_e) != len(front_e) or np.max(np.abs(iso_e - front_e)) > 0:
        raise RuntimeError("Energy axes do not match between migration matrices")

    energies = iso_e
    iso_dist = nearest_supported_distance(energies, iso_mask)
    front_dist = nearest_supported_distance(energies, front_mask)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].step(energies, iso_mask.astype(float), where="mid", label="Isotropic", linewidth=1.1)
    axes[0].step(energies, front_mask.astype(float), where="mid", label="Front", linewidth=1.1)
    axes[0].set_ylabel("Direct Sim")
    axes[0].set_ylim(-0.05, 1.1)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[0].set_title("Migration-matrix support comparison")

    axes[1].plot(energies, iso_dist, label="Isotropic", linewidth=1.2)
    axes[1].plot(energies, front_dist, label="Front", linewidth=1.2)
    axes[1].set_ylabel("Nearest Direct Sim\nDistance [keV]")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(energies, iso_col, label="Isotropic", linewidth=1.0)
    axes[2].plot(energies, front_col, label="Front", linewidth=1.0)
    axes[2].set_ylabel("Column Integral")
    axes[2].set_xlabel("Generated Energy [keV]")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    summary = [
        f"Isotropic direct support: {int(iso_mask.sum())} energies from {energies[iso_mask].min():.0f} to {energies[iso_mask].max():.0f} keV",
        f"Front direct support: {int(front_mask.sum())} energies from {energies[front_mask].min():.0f} to {energies[front_mask].max():.0f} keV",
        f"Front max distance to nearest direct simulation: {np.nanmax(front_dist):.0f} keV",
    ]
    txt = out.with_suffix(".txt")
    txt.write_text("\n".join(summary) + "\n")
    print(f"Saved {out}")
    print(f"Saved {out.with_suffix('.pdf')}")
    print(f"Saved {txt}")


if __name__ == "__main__":
    main()
