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
from src.database.HFIRBG_DB import HFIRBG_DB
from src.database.CartScanFiles import convert_cart_coord_to_det_coord, convert_coord_to_phi

def hist_to_arrays(hist):
    """Convert ROOT histogram to numpy arrays"""
    n_bins = hist.GetNbinsX()
    x = np.array([hist.GetBinCenter(i+1) for i in range(n_bins)])
    y = np.array([hist.GetBinContent(i+1) for i in range(n_bins)])
    edges = np.array([hist.GetBinLowEdge(i+1) for i in range(n_bins+1)])
    return x, y, edges


def export_spectrum_to_csv(x, y, filename):
    """Export spectrum data to CSV file"""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Energy_keV', 'Flux_Hz_per_cm2_per_keV'])
        for i in range(len(x)):
            writer.writerow([f'{x[i]:.2f}', f'{y[i]:.6e}'])

def export_measured_spectrum_to_csv(x, y, filename):
    """Export measured spectrum data to CSV file"""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Energy_keV', 'Rate'])
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
    
    # Map filenames to aliases - REPLACE WITH YOUR ACTUAL NAMES
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
    
    # Initialize database
    db = HFIRBG_DB()
    
    # Store metadata for location diagram
    measurement_metadata = []
    
    # Store all unfolded spectra for combined plot
    all_unfolded = []
    
    unfolddir = join(os.environ.get("HFIRBG_ANALYSIS", "."), "unfold/sumita")
    # Process each file
    for idx, fname in enumerate(fnames, start=1):
        filepath = os.path.join(unfolddir, fname + '_unfold_results.root')
        print(f"\nProcessing file {idx}: {filepath}")
        
        # Extract measurement name from filepath
        filename = basename(filepath).replace('_unfold_results.root', '')
        alias = alias_map.get(filename, filename)  # Use alias or fall back to filename
        
        # Open ROOT file
        f = TFile(filepath, "READ")
        hist_recon = f.Get("FoldedBack")
        hist_measure = f.Get("Measured")
        hist_unf = f.Get("UnfoldedEnergy")
        
        if not hist_recon or not hist_measure or not hist_unf:
            print(f"  ERROR: Could not retrieve histograms from {filepath}")
            continue
        
        # Convert to numpy arrays
        x_recon, y_recon, edges_recon = hist_to_arrays(hist_recon)
        x_measure, y_measure,  edges_measure = hist_to_arrays(hist_measure)
        x_unf, y_unf, edges_unf = hist_to_arrays(hist_unf)
        
        # Store unfolded spectrum for combined plot
        all_unfolded.append({
            'number': idx,
            'filename': filename,
            'alias': alias,
            'x': x_unf,
            'y': y_unf,
            'edges': edges_unf
        })
        
        # Determine x-axis range
        xlow = max(edges_recon[0], edges_measure[0], 50)  # Start at 50 keV minimum
        xhigh = min(edges_recon[-1], edges_measure[-1])
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.step(x_measure, y_measure, where='mid', label='Measured',
                color='#FF0000', linewidth=2, linestyle='-')
        ax.step(x_recon, y_recon, where='mid', label='Reconstructed',
                color='#0000FF', linewidth=2, linestyle='--')
        ax.set_xlim(xlow, xhigh)
        ax.set_yscale('log')
        ax.set_xlabel('Energy [keV]', fontsize=14)
        ax.set_ylabel('Rate [Hz/keV]', fontsize=14)
        ax.set_title(f'Measurement {idx}: {alias}', fontsize=12)
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plot_filename = join(outdir, f'comparison_{idx:02d}.png')
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved comparison plot: {plot_filename}")
        
        # Export unfolded spectrum to CSV
        csv_filename = join(outdir, f'unfolded_spectrum_{idx:02d}.csv')
        export_spectrum_to_csv(x_unf, y_unf, csv_filename)
        print(f"  Saved unfolded spectrum: {csv_filename}")
        
        # Export measured spectrum to CSV
        measured_csv_filename = join(outdir, f'measured_spectrum_{idx:02d}.csv')
        export_measured_spectrum_to_csv(x_measure, y_measure, measured_csv_filename)
        print(f"  Saved measured spectrum: {measured_csv_filename}")
      
        
        # Get metadata from database for location diagram
        try:
            metadata = db.retrieve_file_metadata(filename)
            if metadata:
                det_coords = convert_cart_coord_to_det_coord(
                    metadata["Rx"], metadata["Rz"], 
                    metadata["Lx"], metadata["Lz"], 
                    metadata["angle"]
                )
                phi_deg = convert_coord_to_phi(
                    metadata["Rx"], metadata["Rz"],
                    metadata["Lx"], metadata["Lz"]
                )
                
                measurement_metadata.append({
                    'number': idx,
                    'filename': filename,
                    'alias': alias,
                    'z_pos': det_coords[1],
                    'x_pos': det_coords[0],
                    'angle': metadata["angle"],
                    'phi_deg': phi_deg
                })
                print(f"  Position: z={det_coords[1]:.1f}, x={det_coords[0]:.1f}, angle={metadata['angle']:.1f}°")
        except Exception as e:
            print(f"  WARNING: Could not retrieve metadata: {e}")
        
        f.Close()
    
    # Create combined plot of all unfolded spectra
    if all_unfolded:
        # Calculate grid dimensions (prefer more columns than rows)
        n_plots = len(all_unfolded)
        n_cols = min(3, n_plots)  # Max 3 columns
        n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        
        # Flatten axes array for easier iteration
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_plots > 1 else [axes]
        
        # Define colors for different measurements
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_unfolded)))
        

        for i, spec in enumerate(all_unfolded):
            ax = axes[i]
            ax.step(spec['x'], spec['y'], where='mid', 
                   color='black', linewidth=2, alpha=0.8)
            
            ax.set_xlim(50, 11500)
            ax.set_yscale('log')
            ax.set_xlabel('Unfolded Energy [keV]', fontsize=12)
            ax.set_ylabel('Gamma Flux [Hz/cm²/keV]', fontsize=11)
            ax.set_title(f"{spec['number']}: {spec['alias']}", 
                        fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Hide any unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        combined_plot_filename = join(outdir, 'all_unfolded_spectra.png')
        plt.savefig(combined_plot_filename, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"\nSaved combined unfolded spectra plot: {combined_plot_filename}")
    
    # Save metadata for location diagram
    metadata_file = join(outdir, 'measurement_metadata.csv')
    if measurement_metadata:
        with open(metadata_file, 'w', newline='') as csvfile:
            fieldnames = ['number', 'filename', 'alias', 'z_pos', 'x_pos', 'angle', 'phi_deg']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for meta in measurement_metadata:
                writer.writerow(meta)
        print(f"\nSaved measurement metadata: {metadata_file}")
    
    print(f"\nProcessed {len(fnames)} files")
    print(f"Results saved to: {outdir}")

if __name__ == "__main__":
    main()