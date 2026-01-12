import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.ticker import AutoMinorLocator
from math import pi, cos, sin
import numpy as np
import os
import sys
import csv
from os.path import dirname, realpath, join

# Add src to path
sys.path.insert(1, dirname(dirname(realpath(__file__))))

def HFIR_diagram_with_numbered_measurements(metadata_file, outdir):
    """
    Create HFIR diagram with numbered measurement locations
    
    Parameters:
    -----------
    metadata_file : str
        Path to CSV file containing measurement metadata
    outdir : str
        Output directory for the plot
    """
    # Read measurement metadata
    measurements = []
    with open(metadata_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            measurements.append({
                'number': int(row['number']),
                'filename': row['filename'],
                'alias': row.get('alias', row['filename']),  # Use alias if available
                'z_pos': float(row['z_pos']),
                'x_pos': float(row['x_pos']),
                'angle': float(row['angle']),
                'phi_deg': float(row['phi_deg'])
            })
    
    # Group measurements by location (with small tolerance for floating point comparison)
    from collections import defaultdict
    location_groups = defaultdict(list)
    tolerance = 0.01  # inch tolerance for considering positions "same"
    
    for meas in measurements:
        # Round to group nearby positions
        z_key = round(meas['z_pos'] / tolerance) * tolerance
        x_key = round(meas['x_pos'] / tolerance) * tolerance
        location_groups[(z_key, x_key)].append(meas)

    # Create diagram
    xmin = -5
    ymin = -160
    xmax = 320
    ymax = 150
    corez = 141.15
    corex = -148.5
    corer = 8.66  # in
    
    ratio = abs((ymax - ymin) / (xmax - xmin))
    fig = plt.figure(figsize=(10, 10 * ratio))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel("z [in]", fontsize=14)
    ax1.set_ylabel("x [in]", fontsize=14)
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymax, ymin)
    
    # Draw reactor core
    ax1.add_patch(Circle((corez, corex), corer,
                         edgecolor='black',
                         fill=False,
                         lw=2))
    ax1.text(corez-30, corex + 5, 'Reactor\nCore', fontsize=12, ha='center')
    
    # HB4 and HB3 beam lines
    HB4line = [(240.56 + np.tan(30 * pi / 180) * x, x) 
               for x in np.linspace(corex - 10, ymax, 20)]
    HB3line = [(67.9 - np.tan(30 * pi / 180) * x, x) 
               for x in np.linspace(corex + 10, ymax, 20)]
    
    ax1.plot([a[0] for a in HB4line], [a[1] for a in HB4line], 
             linestyle='dashed', color='grey', linewidth=1)
    ax1.plot([a[0] + 4 for a in HB4line], [a[1] for a in HB4line], 
             linestyle='dashed', color='grey', linewidth=1)
    ax1.plot([a[0] for a in HB3line], [a[1] for a in HB3line], 
             linestyle='dashed', color='grey', linewidth=1)
    ax1.plot([a[0] + 4 for a in HB3line], [a[1] for a in HB3line], 
             linestyle='dashed', color='grey', linewidth=1)
    
    ax1.text(90, -80, 'HB3', fontsize=12, color='grey')
    ax1.text(190, -100, 'HB4', fontsize=12, color='grey')
    
    # Monolith boundary
    monolith_line1 = [(141.5 + a, 111 - a * .1293) for a in [-44.7, 67.1]]
    monolith_line2 = [(monolith_line1[0][0] - a, monolith_line1[0][1] - a * .50) 
                      for a in [0, 100]]
    monolith_line3 = [(monolith_line1[1][0] + a, monolith_line1[1][1] - a * .58) 
                      for a in [0, 150]]
    
    ax1.plot([a[0] for a in monolith_line1], [a[1] for a in monolith_line1], 
             linestyle='solid', color='grey', linewidth=1)
    ax1.plot([a[0] for a in monolith_line2], [a[1] for a in monolith_line2], 
             linestyle='solid', color='grey', linewidth=1)
    ax1.plot([a[0] for a in monolith_line3], [a[1] for a in monolith_line3], 
             linestyle='solid', color='grey', linewidth=1)
    
    # Pool wall
    ax1.plot([0, xmax], [0, 0], linestyle='solid', color='black', linewidth=2)
    ax1.plot([0, 0], [0, -200], linestyle='solid', color='black', linewidth=2)
    ax1.text(120, -8, 'Pool Wall', fontsize=12, color='black')
    
    # PROSPECT detector
    ax1.add_patch(Rectangle((165, 128), 46.25, -83.4,
                            edgecolor='cornflowerblue',
                            fill=False,
                            lw=2))
    ax1.text(188, 100, 'PROSPECT', fontsize=11, color='cornflowerblue', 
             ha='center', fontweight='bold')
    
    # Lead shield walls
    ax1.add_patch(Rectangle((125, 21.5), 286.5 - 155, -14,
                            edgecolor='black',
                            facecolor='lightgrey',
                            lw=1.5))
    ax1.add_patch(Rectangle((10, 21.5), 54, -14,
                            edgecolor='black',
                            facecolor='lightgrey',
                            lw=1.5))
    ax1.text(155, 14, 'Pb Shield', color='black', fontsize=10)
    
    # MIF box
    ax1.add_patch(Rectangle((70, 10), 30, -10,
                            edgecolor='red',
                            fill=False,
                            lw=1.5))
    ax1.text(85, 5, 'MIF', color='red', fontsize=10, ha='center')
    
    # Plot measurement locations
    arrow_length = 12
    
    for location, meas_list in location_groups.items():
        # Use first measurement for position and arrow
        first_meas = meas_list[0]
        z_pos = first_meas['z_pos']
        x_pos = first_meas['x_pos']
        angle = first_meas['angle']
        phi_deg = first_meas['phi_deg']
        
        # Concatenate all measurement numbers at this location
        numbers = [str(m['number']) for m in meas_list]
        label = ','.join(numbers)
        
        # Draw circle for measurement position
        circle = Circle((z_pos, x_pos), radius=6,
                       edgecolor='darkred',
                       facecolor='white',
                       linewidth=2,
                       zorder=10)
        ax1.add_patch(circle)
        
        # Add concatenated numbers inside circle
        ax1.text(z_pos, x_pos, label,
                ha='center', va='center',
                fontsize=11, fontweight='bold',
                color='darkred', zorder=11)
        
        # Draw arrow for non-downward facing measurements
        if angle != 0:
            phi_rad = phi_deg * pi / 180
            dz = arrow_length * cos(phi_rad)
            dx = -arrow_length * sin(phi_rad)
            
            # Start arrow from edge of circle
            start_z = z_pos + (6 * cos(phi_rad))
            start_x = x_pos - (6 * sin(phi_rad))
            
            ax1.arrow(start_z, start_x, dz, dx,
                     width=1.5, head_width=5, head_length=4,
                     fc='darkred', ec='darkred',
                     zorder=9, alpha=0.7)
    
    # Create legend showing number: alias mappings
    legend_text = "Measurements:\n" + "\n".join([f"{m['number']}: {m['alias']}" for m in measurements])
    ax1.text(0.02, 0.98, legend_text,
            transform=ax1.transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='darkred'))

    # Formatting
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis="x", direction="in", length=8, width=1)
    ax1.tick_params(axis="y", direction="in", length=8, width=1)
    ax1.tick_params(axis="x", which="minor", direction="in", length=4, width=1)
    ax1.tick_params(axis="y", which="minor", direction="in", length=4, width=1)
    ax1.grid(True, alpha=0.2, linestyle='--')
    
    # Save figure
    outfile = join(outdir, "measurement_locations.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"Saved measurement location diagram: {outfile}")
    plt.close()


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    arg = ArgumentParser()
    arg.add_argument("--outdir", default=None, 
                    help="Output directory (default: $HFIRBG_ANALYSIS/unfold/paper_files)")
    args = arg.parse_args()
    
    # Set output directory
    if args.outdir is None:
        outdir = join(os.environ.get("HFIRBG_ANALYSIS", "."), "unfold/paper_files")
    else:
        outdir = args.outdir
    
    # Use metadata file from output directory
    metadata_file = join(outdir, 'measurement_metadata.csv')
    
    if not os.path.exists(metadata_file):
        print(f"ERROR: Metadata file not found: {metadata_file}")
        print("Please run plot_unfolding_results.py first to generate the metadata.")
        sys.exit(1)
    
    HFIR_diagram_with_numbered_measurements(metadata_file, outdir)