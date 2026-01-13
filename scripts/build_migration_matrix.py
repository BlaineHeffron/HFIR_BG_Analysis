#!/usr/bin/env python3
"""
Build migration matrix for Ge detector unfolding.

This script:
1. Combines split simulation files for each energy (e.g., 2200_keV_1_Phys.root, 2200_keV_2_Phys.root)
2. Interpolates between simulated energies to fill 1 keV gaps
3. Creates a ROOT file with the migration matrix (TH2F) for GeSpecUnfold.cc

Usage:
    python build_migration_matrix.py <input_dir> <output_file> [options]

Example:
    python build_migration_matrix.py sim/det_response/2026_01_08 migration_matrix.root --elow 40 --ehigh 3000
"""

import uproot
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path
import argparse
import re
from typing import Dict, List, Tuple, Optional
import sys

# Histogram and runtime paths in ROOT files
HIST_PATHS = ["GeEnergyPlugin/hGeEnergy", "miscAccum/hGeEnergy"]
RUNTIME_PATHS = ["accumulated/runtime", "miscAccum/runtime"]


def try_read_object(file, paths):
    """Try to read an object from multiple possible paths."""
    for path in paths:
        try:
            obj = file[path]
            return obj, path
        except:
            continue
    return None, None


def read_runtime(file) -> Optional[float]:
    """Read runtime from ROOT file."""
    runtime_obj, _ = try_read_object(file, RUNTIME_PATHS)
    if runtime_obj is None:
        return None
    for attr in ["fVal", "value"]:
        try:
            val = runtime_obj.member(attr) if hasattr(runtime_obj, 'member') else getattr(runtime_obj, attr)
            if val is not None:
                return float(val)
        except:
            continue
    return None


def read_histogram(file) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Read histogram values, edges, and errors from ROOT file."""
    hist, _ = try_read_object(file, HIST_PATHS)
    if hist is None:
        return None, None, None
    values = hist.values()
    edges = hist.axis().edges()
    errors = hist.errors() if hasattr(hist, 'errors') else np.sqrt(values)
    return values, edges, errors


def collect_files_by_energy(directory: Path, use_front: bool = False) -> Dict[int, List[Path]]:
    """
    Collect all simulation files grouped by energy.

    Args:
        directory: Directory containing ROOT files
        use_front: If True, use directional "front" files (<energy>_keV_dir_front_<num>_Phys.root)
                   If False, use uniform distribution files (<energy>_keV_<num>_Phys.root)

    Returns dict mapping energy (keV) -> list of file paths.
    """
    energy_files = {}

    if use_front:
        # Pattern for directional front files: <energy>_keV_dir_front_<num>_Phys.root
        pattern = re.compile(r'^(\d+)_keV_dir_front_(\d+)_Phys\.root$')
        copy_pattern = r'dir_front_(\d+)_Phys'
    else:
        # Pattern for uniform distribution files: <energy>_keV_<num>_Phys.root
        pattern = re.compile(r'^(\d+)_keV_(\d+)_Phys\.root$')
        copy_pattern = r'_keV_(\d+)_Phys'

    for f in directory.iterdir():
        match = pattern.match(f.name)
        if match:
            energy = int(match.group(1))
            if energy not in energy_files:
                energy_files[energy] = []
            energy_files[energy].append(f)

    # Sort files for each energy by copy number
    for energy in energy_files:
        energy_files[energy].sort(key=lambda p: int(re.search(copy_pattern, p.name).group(1)))

    return energy_files


def combine_files(file_paths: List[Path]) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Combine multiple ROOT files for the same energy.
    Adds histograms and runtimes together.
    Returns (combined_values, edges, total_runtime).
    """
    combined_values = None
    edges = None
    total_runtime = 0.0

    for fp in file_paths:
        with uproot.open(fp) as f:
            values, file_edges, _ = read_histogram(f)
            runtime = read_runtime(f)

            if values is None or runtime is None:
                print(f"  Warning: Could not read {fp}")
                continue

            if combined_values is None:
                combined_values = values.copy()
                edges = file_edges
            else:
                combined_values += values

            total_runtime += runtime

    if combined_values is None:
        raise ValueError(f"Could not combine files: {file_paths}")

    return combined_values, edges, total_runtime


def rebin_histogram(values: np.ndarray, edges_mev: np.ndarray,
                    target_low_kev: float, target_high_kev: float,
                    n_bins: int) -> np.ndarray:
    """
    Rebin histogram from MeV edges to target keV bins.
    Matches the rebinning logic in GeSpecUnfold.cc.
    """
    # Convert edges to keV
    edges_kev = edges_mev * 1000.0
    centers_kev = (edges_kev[:-1] + edges_kev[1:]) / 2.0

    # Create output histogram
    rebinned = np.zeros(n_bins + 2)  # +2 for under/overflow
    target_edges = np.linspace(target_low_kev - 0.5, target_high_kev + 0.5, n_bins + 1)

    # Fill rebinned histogram
    for i, (val, center) in enumerate(zip(values, centers_kev)):
        if center < target_low_kev - 0.5:
            rebinned[0] += val  # Underflow
        elif center > target_high_kev + 0.5:
            rebinned[-1] += val  # Overflow
        else:
            bin_idx = int(np.floor(center - (target_low_kev - 0.5))) + 1
            if 1 <= bin_idx <= n_bins:
                rebinned[bin_idx] += val

    return rebinned


def normalize_histogram(values: np.ndarray, runtime: float, bin_width: float = 1.0) -> np.ndarray:
    """
    Normalize histogram by runtime and bin width.
    Matches GeSpecUnfold.cc: Scale(1./liveTime) then normalize_to_bin_width.
    """
    if runtime <= 0:
        return values
    normalized = values / runtime / bin_width
    return normalized


def get_inverse_warp(target_centers: np.ndarray, source_E_keV: float, target_E_keV: float) -> np.ndarray:
    """
    For each position in target space, find the corresponding position in source space.
    This allows us to sample the source spectrum at the right positions.

    Maps physics features (511 keV, DEP, SEP, FEP) correctly between energies.
    """
    s_mev = source_E_keV / 1000.0
    t_mev = target_E_keV / 1000.0

    # Define anchor points in MeV: [0, 511 keV, DEP, SEP, FEP]
    # These are positions where features appear in each spectrum
    anchors_src = [0.0, 0.511, s_mev - 1.022, s_mev - 0.511, s_mev]
    anchors_tgt = [0.0, 0.511, t_mev - 1.022, t_mev - 0.511, t_mev]

    # Filter anchors: must be > 0 and strictly increasing
    valid_src = [anchors_src[0]]
    valid_tgt = [anchors_tgt[0]]

    for i in range(1, len(anchors_src)):
        if anchors_src[i] > valid_src[-1] + 1e-6 and anchors_tgt[i] > valid_tgt[-1] + 1e-6:
            valid_src.append(anchors_src[i])
            valid_tgt.append(anchors_tgt[i])

    # Map from TARGET space to SOURCE space (inverse mapping)
    # Given a position in the target spectrum, where should we sample in the source?
    inverse_warp = interp1d(valid_tgt, valid_src, kind='linear', fill_value="extrapolate")

    # Convert target_centers from keV to MeV, apply warp, convert back
    target_mev = target_centers / 1000.0
    source_mev = inverse_warp(target_mev)
    return source_mev * 1000.0  # Back to keV


def warp_and_resample(values: np.ndarray, edges: np.ndarray,
                      source_E: float, target_E: float) -> np.ndarray:
    """
    Warps the spectrum using physics-based anchors and resamples to target grid.

    For each bin in the target spectrum, finds the corresponding position in
    the source spectrum and samples there.

    Args:
        values: Histogram bin values (without under/overflow)
        edges: Bin edges (len = len(values) + 1)
        source_E: Source energy in keV
        target_E: Target energy in keV
    """
    centers = (edges[:-1] + edges[1:]) / 2.0

    # For each target bin position, find where to sample in source spectrum
    source_positions = get_inverse_warp(centers, source_E, target_E)

    # Create interpolator for source spectrum
    safe_vals = np.maximum(values, 1e-10)
    try:
        source_interp = interp1d(centers, np.log(safe_vals), kind='cubic',
                                 bounds_error=False, fill_value=-23)
    except ValueError:
        source_interp = interp1d(centers, np.log(safe_vals), kind='linear',
                                 bounds_error=False, fill_value=-23)

    # Sample source spectrum at the mapped positions
    new_values = np.exp(source_interp(source_positions))

    return new_values


def interpolate_histogram(v1: np.ndarray, E1: float, edges: np.ndarray,
                         v2: np.ndarray, E2: float, target_E: float) -> np.ndarray:
    """
    Interpolate between two histograms to get response at target energy.
    Uses physics-aware warping to align spectral features.

    Args:
        v1, v2: Histogram values WITH under/overflow (indices 0 and -1)
        edges: Bin edges for the main histogram (not including under/overflow)
        E1, E2: Energies of the two reference histograms
        target_E: Target energy to interpolate to

    Returns:
        Interpolated histogram WITH under/overflow
    """
    # Extract main histogram (without under/overflow)
    main_v1 = v1[1:-1]
    main_v2 = v2[1:-1]

    # Warp both reference histograms to the target energy
    w_v1 = warp_and_resample(main_v1, edges, E1, target_E)
    w_v2 = warp_and_resample(main_v2, edges, E2, target_E)

    # Weight based on proximity in energy
    weight = (E2 - target_E) / (E2 - E1)

    interp_main = weight * w_v1 + (1 - weight) * w_v2

    # Handle under/overflow with simple linear interpolation
    interp_under = weight * v1[0] + (1 - weight) * v2[0]
    interp_over = weight * v1[-1] + (1 - weight) * v2[-1]

    # Reconstruct full histogram with under/overflow
    interp_v = np.concatenate([[interp_under], interp_main, [interp_over]])
    return interp_v


def build_migration_matrix(input_dir: Path, elow: int, ehigh: int,
                           output_path: Path, use_front: bool = False,
                           verbose: bool = True) -> None:
    """
    Build the full migration matrix.

    Args:
        input_dir: Directory containing simulation ROOT files
        elow: Lower energy bound (keV)
        ehigh: Upper energy bound (keV)
        output_path: Output ROOT file path
        use_front: Use directional "front" files instead of uniform distribution
        verbose: Print progress
    """
    file_type = "directional (front)" if use_front else "uniform distribution"
    if verbose:
        print(f"Building migration matrix from {input_dir}")
        print(f"File type: {file_type}")
        print(f"Energy range: {elow} - {ehigh} keV")

    # Collect all files grouped by energy
    energy_files = collect_files_by_energy(input_dir, use_front=use_front)
    simulated_energies = sorted(energy_files.keys())

    if verbose:
        print(f"Found {len(simulated_energies)} simulated energies")
        print(f"Simulated range: {min(simulated_energies)} - {max(simulated_energies)} keV")

    # Filter to energies within our range
    simulated_energies = [e for e in simulated_energies if elow <= e <= ehigh]

    # Target energies: all 1 keV increments
    target_energies = list(range(elow, ehigh + 1))
    n_energies = len(target_energies)
    n_bins = ehigh - elow + 1

    if verbose:
        print(f"Target energies: {n_energies} (1 keV increments)")

    # Matrix dimensions
    bin_low = elow - 0.5
    bin_high = ehigh + 0.5

    # Store combined/normalized histograms for simulated energies
    sim_histograms = {}  # energy -> normalized rebinned histogram

    if verbose:
        print("\nCombining simulation files...")

    for energy in simulated_energies:
        files = energy_files[energy]
        if verbose:
            print(f"  {energy} keV: combining {len(files)} files...", end=" ")

        try:
            values, edges, runtime = combine_files(files)
            rebinned = rebin_histogram(values, edges, elow, ehigh, n_bins)
            normalized = normalize_histogram(rebinned, runtime, bin_width=1.0)
            sim_histograms[energy] = normalized
            if verbose:
                print(f"done (runtime={runtime:.1f}s, integral={np.sum(rebinned):.0f})")
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    # Build full matrix with interpolation
    if verbose:
        print("\nBuilding migration matrix with interpolation...")

    # Create edges for rebinned histogram
    rebin_edges = np.linspace(bin_low, bin_high, n_bins + 1)

    # Migration matrix: rows = detected energy, cols = generated energy
    # Using n_bins + 2 for under/overflow bins
    migration_data = np.zeros((n_bins + 2, n_energies))

    # Track which energies are simulated vs interpolated
    energy_source = {}  # energy -> 'simulated' or 'interpolated'

    for i, target_E in enumerate(target_energies):
        if target_E in sim_histograms:
            # Use simulated data directly
            migration_data[:, i] = sim_histograms[target_E]
            energy_source[target_E] = 'simulated'
        else:
            # Interpolate from neighboring simulated energies
            lower_energies = [e for e in simulated_energies if e < target_E]
            upper_energies = [e for e in simulated_energies if e > target_E]

            if not lower_energies or not upper_energies:
                # Can't interpolate - outside simulated range
                if verbose:
                    print(f"  Warning: Cannot interpolate {target_E} keV (outside range)")
                energy_source[target_E] = 'extrapolated'
                continue

            E1 = max(lower_energies)
            E2 = min(upper_energies)

            v1 = sim_histograms[E1]
            v2 = sim_histograms[E2]

            interp_v = interpolate_histogram(v1, E1, rebin_edges, v2, E2, target_E)
            migration_data[:, i] = interp_v
            energy_source[target_E] = 'interpolated'

    # Count simulated vs interpolated
    n_sim = sum(1 for s in energy_source.values() if s == 'simulated')
    n_interp = sum(1 for s in energy_source.values() if s == 'interpolated')

    if verbose:
        print(f"  Simulated: {n_sim}, Interpolated: {n_interp}")

    # Write output ROOT file using PyROOT for proper format
    if verbose:
        print(f"\nWriting to {output_path}...")

    import ROOT

    # Use compression to reduce file size
    f = ROOT.TFile.Open(str(output_path), "RECREATE")
    f.SetCompressionLevel(4)

    # Create 2D migration histogram using TH2F (float) instead of TH2D (double)
    # to stay under ROOT's 1GB object limit
    h2 = ROOT.TH2F("MdetgenMC", ";energy(det);energy(gen)",
                   n_bins, bin_low, bin_high,
                   n_energies, -0.5, n_energies - 0.5)

    # Fill the histogram (excluding under/overflow which are in indices 0 and -1)
    for iDet in range(n_bins):
        for iGen in range(n_energies):
            val = float(migration_data[iDet + 1, iGen])  # +1 to skip underflow
            h2.SetBinContent(iDet + 1, iGen + 1, val)

    h2.Write()

    # Store energies as a 1D histogram (use TH1F for consistency)
    h_energies = ROOT.TH1F("hEnergies", "Generated energies",
                           n_energies, -0.5, n_energies - 0.5)
    for i, e in enumerate(target_energies):
        h_energies.SetBinContent(i + 1, float(e))
    h_energies.Write()

    # Store simulated mask
    h_mask = ROOT.TH1F("hSimulatedMask", "Simulated (1) vs interpolated (0)",
                       n_energies, -0.5, n_energies - 0.5)
    for i, e in enumerate(target_energies):
        h_mask.SetBinContent(i + 1, 1.0 if energy_source.get(e) == 'simulated' else 0.0)
    h_mask.Write()

    # Store energy bounds
    h_elow = ROOT.TH1F("hELow", "Lower energy bound", 1, 0, 1)
    h_elow.SetBinContent(1, float(elow))
    h_elow.Write()

    h_ehigh = ROOT.TH1F("hEHigh", "Upper energy bound", 1, 0, 1)
    h_ehigh.SetBinContent(1, float(ehigh))
    h_ehigh.Write()

    f.Close()

    if verbose:
        print("Done!")
        print(f"\nMigration matrix shape: {migration_data.shape}")
        print(f"Output file: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build migration matrix for Ge detector unfolding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('input_dir', type=str,
                        help="Directory containing simulation ROOT files")
    parser.add_argument('output_file', type=str,
                        help="Output ROOT file for migration matrix (will add '_front' suffix if --front is used)")
    parser.add_argument('--elow', type=int, default=40,
                        help="Lower energy bound in keV (default: 40)")
    parser.add_argument('--ehigh', type=int, default=3000,
                        help="Upper energy bound in keV (default: 3000)")
    parser.add_argument('--front', action='store_true',
                        help="Use directional 'front' files (*_keV_dir_front_*_Phys.root) instead of uniform distribution")
    parser.add_argument('--quiet', action='store_true',
                        help="Suppress progress output")

    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    output_path = Path(args.output_file).expanduser()

    # Modify output filename if using front files
    if args.front:
        stem = output_path.stem
        suffix = output_path.suffix
        output_path = output_path.with_name(f"{stem}_front{suffix}")

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    build_migration_matrix(
        input_dir=input_dir,
        elow=args.elow,
        ehigh=args.ehigh,
        output_path=output_path,
        use_front=args.front,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
