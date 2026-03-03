import sys
from os.path import dirname, realpath, join, basename
sys.path.insert(1, dirname(dirname(realpath(__file__))))
from argparse import ArgumentParser
from ROOT import TFile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

def hist_to_arrays(hist):
    """Convert ROOT histogram to numpy arrays"""
    n_bins = hist.GetNbinsX()
    x = np.array([hist.GetBinCenter(i+1) for i in range(n_bins)])
    y = np.array([hist.GetBinContent(i+1) for i in range(n_bins)])
    edges = np.array([hist.GetBinLowEdge(i+1) for i in range(n_bins+1)])
    return x, y, edges


def export_spectrum_to_csv(x, y, filename, columns=('Energy_keV', 'Flux_Hz_per_mm2_per_keV')):
    """Export spectrum data to CSV file"""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        for i in range(len(x)):
            writer.writerow([f'{x[i]:.2f}', f'{y[i]:.6e}'])

def main():
    arg = ArgumentParser()
    arg.add_argument("--outdir", default=None, help="Output directory (default: $HFIRBG_ANALYSIS/unfold/paper_files)")
    args = arg.parse_args()

    fnames = ['MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN',
            'MIF_BOX_AT_REACTOR_RXOFF',
            'CYCLE461_DOWN_FACING_OVERNIGHT',
            'HB4_DOWN_OVERNIGHT_1',
            'EAST_FACE_18',
            'EAST_FACE_1',
            ]

    alias_map = {
        'MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN': 'MIF (Rx On)',
        'MIF_BOX_AT_REACTOR_RXOFF': 'MIF (Rx Off)',
        'CYCLE461_DOWN_FACING_OVERNIGHT': 'Shield Center',
        'HB4_DOWN_OVERNIGHT_1': 'HB4',
        'EAST_FACE_18': 'PROSPECT East 1',
        'EAST_FACE_1': 'PROSPECT East 2',
    }

    # Set output directory
    if args.outdir is None:
        outdir = join(os.environ.get("HFIRBG_ANALYSIS", "."), "unfold/paper_files")
    else:
        outdir = args.outdir

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Store all unfolded spectra for combined plot
    all_unfolded = []

    unfolddir = join(os.environ.get("HFIRBG_ANALYSIS", "."), "unfold/sumita")
    # Process each file
    for idx, fname in enumerate(fnames, start=1):
        filepath = os.path.join(unfolddir, fname + '_unfold_results.root')
        print(f"\nProcessing file {idx}: {filepath}")

        filename = basename(filepath).replace('_unfold_results.root', '')
        alias = alias_map.get(filename, filename)

        f = TFile(filepath, "READ")
        hist_unf = f.Get("UnfoldedEnergy")
        hist_meas = f.Get("Measured")

        if not hist_unf:
            print(f"  ERROR: Could not retrieve UnfoldedEnergy from {filepath}")
            f.Close()
            continue

        x_unf, y_unf, edges_unf = hist_to_arrays(hist_unf)
        if hist_meas:
            x_meas, y_meas, edges_meas = hist_to_arrays(hist_meas)
        else:
            x_meas, y_meas, edges_meas = None, None, None
            print(f"  WARNING: Could not retrieve Measured from {filepath}")

        all_unfolded.append({
            'number': idx,
            'filename': filename,
            'alias': alias,
            'x': x_unf,
            'y': y_unf,
            'edges': edges_unf,
            'x_meas': x_meas,
            'y_meas': y_meas,
        })

        # Export unfolded spectrum to CSV
        csv_filename = join(outdir, f'unfolded_spectrum_{idx:02d}_{filename}.csv')
        export_spectrum_to_csv(x_unf, y_unf, csv_filename)
        print(f"  Saved CSV: {csv_filename}")

        # Export measured spectrum to CSV
        if x_meas is not None:
            meas_csv = join(outdir, f'measured_spectrum_{idx:02d}_{filename}.csv')
            export_spectrum_to_csv(x_meas, y_meas, meas_csv,
                                   columns=('Energy_keV', 'Rate_Hz_per_keV'))
            print(f"  Saved CSV: {meas_csv}")

        f.Close()

    # Create single combined plot with all unfolded spectra on same axes
    if all_unfolded:
        fig, ax = plt.subplots(figsize=(12, 7))

        colors = plt.cm.tab10(np.linspace(0, 0.6, len(all_unfolded)))

        for i, spec in enumerate(all_unfolded):
            ax.step(spec['x'], spec['y'], where='mid',
                   color=colors[i], linewidth=1.5, alpha=0.85,
                   label=spec['alias'])

        ax.set_xlim(50, 11500)
        ax.set_yscale('log')
        ax.set_xlabel('Energy [keV]', fontsize=14)
        ax.set_ylabel('Gamma Flux [Hz/mm$^2$/keV]', fontsize=14)
        ax.set_title('Unfolded Gamma Spectra', fontsize=16)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)

        plt.tight_layout()
        combined_plot_filename = join(outdir, 'all_unfolded_spectra.png')
        plt.savefig(combined_plot_filename, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"\nSaved combined plot: {combined_plot_filename}")

    # Create 3x2 grid comparing measured vs unfolded spectra (dual y-axes)
    if all_unfolded:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        colors_meas = '#1f77b4'
        colors_unf = '#d62728'

        for i, spec in enumerate(all_unfolded[:6]):
            row, col = divmod(i, 3)
            ax1 = axes[row, col]

            # Measured spectrum on left y-axis
            if spec['y_meas'] is not None:
                ax1.step(spec['x_meas'], spec['y_meas'], where='mid',
                        color=colors_meas, linewidth=1.0, alpha=0.8,
                        label='Measured')
            ax1.set_yscale('log')
            ax1.set_ylabel('Measured [counts/keV]', fontsize=10, color=colors_meas)
            ax1.tick_params(axis='y', labelcolor=colors_meas, labelsize=9)

            # Unfolded spectrum on right y-axis
            ax2 = ax1.twinx()
            ax2.step(spec['x'], spec['y'], where='mid',
                    color=colors_unf, linewidth=1.0, alpha=0.8,
                    label='Unfolded')
            ax2.set_yscale('log')
            ax2.set_ylabel('Flux [Hz/mm$^2$/keV]', fontsize=10, color=colors_unf)
            ax2.tick_params(axis='y', labelcolor=colors_unf, labelsize=9)

            ax1.set_xlim(50, 11500)
            ax1.set_xlabel('Energy [keV]', fontsize=10)
            ax1.set_title(spec['alias'], fontsize=12)
            ax1.grid(True, alpha=0.3)

            # Combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

        plt.tight_layout()
        comparison_filename = join(outdir, 'measured_vs_unfolded_comparison.png')
        plt.savefig(comparison_filename, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison plot: {comparison_filename}")

    print(f"\nProcessed {len(all_unfolded)}/{len(fnames)} files")
    print(f"Results saved to: {outdir}")

if __name__ == "__main__":
    main()
