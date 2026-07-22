#!/usr/bin/env python3
"""Export ROOT response matrices to NumPy and package the public supplements.

The script creates two portable, compressed NumPy archives from the production
ROOT matrices.  It can then add them to both the versioned public-data tarball
and the paper's arXiv ancillary ZIP.  ROOT is needed only for this conversion;
readers need only NumPy to load the resulting ``.npz`` files.

Example:
    .venv/bin/python scripts/package_public_response_matrices.py \
        --paper-ancillary-dir /path/to/paper/arxiv_submission/anc \
        --paper-output /path/to/paper/paper_files.zip
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# The data directory is the v1.0.1 source bundle.  The release archive uses a
# new top-level name so an installation of v1.1.0 can coexist with v1.0.1.
SOURCE_DATA_BASENAME = "HFIRBG_public_data_v1.0.1"
RELEASE_DATA_BASENAME = "HFIRBG_public_data_v1.1.0"
DEFAULT_DATA_DIR = REPOSITORY_ROOT / "data" / SOURCE_DATA_BASENAME
DEFAULT_RELEASE_ARCHIVE = REPOSITORY_ROOT / "data" / f"{RELEASE_DATA_BASENAME}.tar.gz"
DEFAULT_ISOTROPIC_ROOT = REPOSITORY_ROOT / "scripts/private/migration_matrix.root"
DEFAULT_FRONT_ROOT = REPOSITORY_ROOT / "scripts/private/migration_matrix_front.root"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--release-archive", type=Path, default=DEFAULT_RELEASE_ARCHIVE)
    parser.add_argument(
        "--release-root-name",
        default=RELEASE_DATA_BASENAME,
        help="top-level directory name inside the public-data tarball",
    )
    parser.add_argument("--isotropic-root", type=Path, default=DEFAULT_ISOTROPIC_ROOT)
    parser.add_argument("--front-root", type=Path, default=DEFAULT_FRONT_ROOT)
    parser.add_argument(
        "--paper-ancillary-dir",
        type=Path,
        help="existing ancillary-file directory to add to paper_files.zip",
    )
    parser.add_argument(
        "--paper-output",
        type=Path,
        help="output paper_files.zip; required with --paper-ancillary-dir",
    )
    parser.add_argument(
        "--skip-archive-build",
        action="store_true",
        help="export the matrices but do not rebuild the public-data tarball",
    )
    return parser.parse_args()


def write_npy_member(archive: zipfile.ZipFile, name: str, array: np.ndarray) -> None:
    with archive.open(name, "w") as member:
        np.lib.format.write_array(member, array, version=(2, 0), allow_pickle=False)


def export_matrix(source: Path, destination: Path, scenario: str) -> None:
    """Export a production TH2F response matrix as a self-describing NPZ file."""
    import ROOT

    root_file = ROOT.TFile.Open(str(source), "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"cannot open ROOT matrix: {source}")
    try:
        histogram = root_file.Get("MdetgenMC")
        generated_histogram = root_file.Get("hEnergies")
        simulated_histogram = root_file.Get("hSimulatedMask")
        if not histogram or not generated_histogram or not simulated_histogram:
            raise RuntimeError(f"{source} is missing one or more required response objects")

        n_detected = histogram.GetNbinsX()
        n_generated = histogram.GetNbinsY()
        expected_size = (n_detected + 2) * (n_generated + 2)
        # TH2 stores x bins contiguously inside each y-bin row.  Drop ROOT's
        # under/overflow bins and transpose to the documented (detected, generated)
        # convention without changing any response coefficients.
        raw = np.frombuffer(histogram.GetArray(), dtype=np.float32, count=expected_size)
        root_layout = raw.reshape((n_generated + 2, n_detected + 2))
        matrix = np.ascontiguousarray(root_layout[1:-1, 1:-1].T)
        detected_energy = np.asarray(
            [histogram.GetXaxis().GetBinCenter(index) for index in range(1, n_detected + 1)],
            dtype=np.float64,
        )
        generated_energy = np.asarray(
            [generated_histogram.GetBinContent(index) for index in range(1, n_generated + 1)],
            dtype=np.float64,
        )
        simulated_mask = np.asarray(
            [simulated_histogram.GetBinContent(index) > 0 for index in range(1, n_generated + 1)],
            dtype=np.bool_,
        )
        metadata = {
            "format_version": 1,
            "scenario": scenario,
            "source_root_file": source.name,
            "matrix_array": "matrix",
            "matrix_dtype": "float32",
            "matrix_shape": [int(n_detected), int(n_generated)],
            "matrix_convention": "matrix[detected_energy_index, generated_energy_index]",
            "detected_energy_units": "keV",
            "generated_energy_units": "keV",
            "detected_energy_min_keV": float(detected_energy[0]),
            "detected_energy_max_keV": float(detected_energy[-1]),
            "generated_energy_min_keV": float(generated_energy[0]),
            "generated_energy_max_keV": float(generated_energy[-1]),
            "directly_simulated_generated_energy_count": int(simulated_mask.sum()),
            "normalization": "Unchanged from the production ROOT MdetgenMC histogram.",
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            with zipfile.ZipFile(
                temporary_path,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
                allowZip64=True,
            ) as archive:
                write_npy_member(archive, "matrix.npy", matrix)
                write_npy_member(archive, "detected_energy_keV.npy", detected_energy)
                write_npy_member(archive, "generated_energy_keV.npy", generated_energy)
                write_npy_member(archive, "simulated_generated_energy_mask.npy", simulated_mask)
                archive.writestr("metadata.json", json.dumps(metadata, indent=2, sort_keys=True) + "\n")
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)
    finally:
        root_file.Close()


def build_release_archive(data_dir: Path, destination: Path, release_root_name: str) -> None:
    if not (data_dir / "HFIRBG.db").is_file() or not (data_dir / "spectra").is_dir():
        raise RuntimeError(f"{data_dir} does not look like a public data bundle")
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with tarfile.open(temporary_path, "w:gz", compresslevel=6) as archive:
            archive.add(data_dir, arcname=release_root_name)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def build_paper_zip(ancillary_dir: Path, destination: Path, data_dir: Path) -> None:
    if not ancillary_dir.is_dir():
        raise RuntimeError(f"paper ancillary directory does not exist: {ancillary_dir}")
    with tempfile.TemporaryDirectory(prefix="hfirbg-paper-files-") as staging_name:
        staging = Path(staging_name) / "paper_files"
        shutil.copytree(ancillary_dir, staging)
        (staging / "UNFOLDING_RESULTS_README.md").rename(staging / "README.md")
        results_dir = staging / "unfolding_results"
        spectra_dir = results_dir / "spectra"
        figures_dir = results_dir / "figures"
        metadata_dir = results_dir / "metadata"
        for directory in (spectra_dir, figures_dir, metadata_dir):
            directory.mkdir(parents=True)
        for path in sorted(staging.iterdir()):
            if not path.is_file() or path.name == "README.md":
                continue
            if path.suffix.lower() in {".pdf", ".png"}:
                target_directory = figures_dir
            elif path.name.startswith(("measured_spectrum_", "unfolded_spectrum_")):
                target_directory = spectra_dir
            else:
                target_directory = metadata_dir
            path.rename(target_directory / path.name)

        data_output_dir = staging / "data"
        data_output_dir.mkdir()
        shutil.copy2(data_dir / "HFIRBG.db", data_output_dir / "HFIRBG.db")
        shutil.copytree(data_dir / "spectra", data_output_dir / "spectra")
        shutil.copytree(data_dir / "migration_matrices", staging / "response_matrices")
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            with zipfile.ZipFile(
                temporary_path,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
                allowZip64=True,
            ) as archive:
                for path in sorted(staging.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(staging.parent))
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if bool(args.paper_ancillary_dir) != bool(args.paper_output):
        raise SystemExit("--paper-ancillary-dir and --paper-output must be used together")
    matrix_dir = args.data_dir / "migration_matrices"
    targets = (
        (args.isotropic_root, matrix_dir / "migration_matrix_isotropic.npz", "isotropic"),
        (args.front_root, matrix_dir / "migration_matrix_front.npz", "front"),
    )
    for source, destination, scenario in targets:
        if not source.is_file():
            raise SystemExit(f"missing source matrix: {source}")
        print(f"Exporting {scenario} response matrix to {destination}")
        export_matrix(source, destination, scenario)
    if not args.skip_archive_build:
        print(f"Building public-data archive: {args.release_archive}")
        build_release_archive(args.data_dir, args.release_archive, args.release_root_name)
    if args.paper_ancillary_dir:
        print(f"Building paper supplemental archive: {args.paper_output}")
        build_paper_zip(args.paper_ancillary_dir, args.paper_output, args.data_dir)


if __name__ == "__main__":
    main()
