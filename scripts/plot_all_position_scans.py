import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import AutoMinorLocator
from matplotlib import cm
from math import pi, sin, cos
import numpy as np
import os
import sys
from os.path import dirname, realpath, join
import glob

# Add src to path for database imports
sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.database.HFIRBG_DB import HFIRBG_DB
from src.database.CartScanFiles import convert_cart_coord_to_det_coord, convert_coord_to_phi
from src.utilities.util import read_rows_csv

outdir = os.path.join(os.environ["HFIRBG_ANALYSIS"], "diagrams")
db_dir = join(dirname(dirname(realpath(__file__))), "db")


def HFIR_diagram_base():
    """Create base HFIR diagram without detectors"""
    xmin = -5
    ymin = -160
    xmax = 320
    ymax = 180
    corez = 141.15
    corex = -148.5
    corer = 8.66  # in
    ratio = abs((ymax - ymin) / (xmax - xmin))
    fig = plt.figure(figsize=(10, 10 * ratio))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel("z [in]")
    ax1.set_ylabel("x [in]")
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymax, ymin)
    
    # core
    from matplotlib.patches import Circle, Rectangle
    ax1.add_patch(Circle((corez, corex), corer,
                         edgecolor='black',
                         fill=False,
                         lw=2))
    # HB4
    HB4line = [(240.56 + np.tan(30 * pi / 180) * x, x) for x in np.linspace(corex - 10, ymax, 20)]
    # HB3
    HB3line = [(67.9 - np.tan(30 * pi / 180) * x, x) for x in np.linspace(corex + 10, ymax, 20)]
    
    ax1.plot([a[0] for a in HB4line], [a[1] for a in HB4line], linestyle='dashed', color='grey')
    ax1.plot([a[0] + 4 for a in HB4line], [a[1] for a in HB4line], linestyle='dashed', color='grey')
    ax1.plot([a[0] for a in HB3line], [a[1] for a in HB3line], linestyle='dashed', color='grey')
    ax1.plot([a[0] + 4 for a in HB3line], [a[1] for a in HB3line], linestyle='dashed', color='grey')
    
    # monolith
    monolith_line1 = [(141.5 + a, 111 - a * .1293) for a in [-44.7, 67.1]]
    monolith_line2 = [(monolith_line1[0][0] - a, monolith_line1[0][1] - a * .50) for a in [0, 100]]
    monolith_line3 = [(monolith_line1[1][0] + a, monolith_line1[1][1] - a * .58) for a in [0, 150]]
    ax1.plot([a[0] for a in monolith_line1], [a[1] for a in monolith_line1], linestyle='solid', color='grey')
    ax1.plot([a[0] for a in monolith_line2], [a[1] for a in monolith_line2], linestyle='solid', color='grey')
    ax1.plot([a[0] for a in monolith_line3], [a[1] for a in monolith_line3], linestyle='solid', color='grey')
    
    # pool wall
    ax1.plot([0, xmax], [0, 0], linestyle='solid', color='black')
    ax1.plot([0, 0], [0, -200], linestyle='solid', color='black')
    
    # prospect
    ax1.add_patch(Rectangle((165, 128), 46.25, -83.4,
                            edgecolor='cornflowerblue',
                            fill=False,
                            lw=2))
    
    # lead shield wall
    ax1.add_patch(Rectangle((125, 21.5), 286.5 - 155, -14,
                            edgecolor='black',
                            facecolor='black',
                            fill=False,
                            lw=1))
    ax1.add_patch(Rectangle((10, 21.5), 54, -14,
                            edgecolor='black',
                            facecolor='black',
                            fill=False,
                            lw=1))
    ax1.add_patch(Rectangle((70, 10), 30, -10,
                            edgecolor='red',
                            fill=False,
                            lw=1))
    
    # russian doll
    ax1.add_patch(Circle((200, 21.5 + 12), 12,
                         edgecolor='green',
                         fill=False,
                         lw=2))
    
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis="x", direction="in", length=16, width=1)
    ax1.tick_params(axis="y", direction="in", length=16, width=1)
    ax1.tick_params(axis="x", which="minor", direction="in", length=8, width=1)
    ax1.tick_params(axis="y", which="minor", direction="in", length=8, width=1)
    
    return fig, ax1


def parse_position_scan_row(row):
    """Parse a row from position scan CSV"""
    R_coords = row[0].split(',')
    L_coords = row[1].split(',')
    
    Rx = float(R_coords[0].strip())
    Rz = float(R_coords[1].strip())
    Lx = float(L_coords[0].strip())
    Lz = float(L_coords[1].strip())
    angle = float(row[2])
    filename = row[6]
    
    return {
        'Rx': Rx, 'Rz': Rz, 'Lx': Lx, 'Lz': Lz,
        'angle': angle, 'filename': filename
    }


def plot_all_position_scans(min_live_time=3600):
    """Plot all position scans with live time > min_live_time"""
    
    # Initialize database
    db = HFIRBG_DB()
    
    # Find all position scan files
    position_files = sorted(glob.glob(join(db_dir, "position_scan_*.csv")))
    
    if not position_files:
        print("No position scan files found!")
        return
    
    # Collect all valid positions
    positions = []
    for fpath in position_files:
        data = read_rows_csv(fpath, delimiter='|')
        # Skip header
        for row in data[1:]:
            try:
                pos_data = parse_position_scan_row(row)
                
                # Get file metadata from database to retrieve live time
                fname = pos_data['filename']
                if fname.endswith('.txt'):
                    fname = fname[:-4]
                
                row = db.retrieve_file_time(fname)
                if row is None:
                    print(f"Warning: Could not find metadata for {fname}")
                    continue
                live_time = float(row[1])
                
                if live_time > min_live_time:
                    pos_data['live_time'] = live_time
                    positions.append(pos_data)
                    
            except (ValueError, IndexError) as e:
                print(f"Skipping row in {fpath}: {e}")
                continue
    
    if not positions:
        print("No positions found with live time > {} seconds".format(min_live_time))
        return
    
    print(f"Found {len(positions)} positions with live time > {min_live_time} seconds")
    
    # Create base diagram
    fig, ax = HFIR_diagram_base()
    
    # Set up colormap based on live time
    live_times = [p['live_time'] for p in positions]
    norm = Normalize(vmin=min(live_times), vmax=max(live_times))
    cmap = cm.viridis
    
    arrow_length = 15  # Length of arrows in plot units
    
    # Plot each position
    for pos in positions:
        # Convert to detector coordinates
        det_coords = convert_cart_coord_to_det_coord(
            pos['Rx'], pos['Rz'], pos['Lx'], pos['Lz'], pos['angle']
        )
        x_pos, z_pos = det_coords[0], det_coords[1]
        
        # Get color based on live time
        color = cmap(norm(pos['live_time']))
        
        # For pointing straight down, use a star
        if pos['angle'] == 0:
            ax.scatter(z_pos, x_pos, marker='*', s=300, c=[color], 
                      edgecolors='black', linewidth=1, zorder=10)
        else:
            # Calculate phi (cart angle) for arrow direction
            phi_deg = convert_coord_to_phi(pos['Rx'], pos['Rz'], pos['Lx'], pos['Lz'])
            phi_rad = phi_deg * pi / 180
            
            # Calculate arrow direction
            dz = arrow_length * cos(phi_rad)
            dx = -arrow_length * sin(phi_rad)
            
            # Create arrow
            arrow = FancyArrowPatch((z_pos, x_pos), 
                                   (z_pos + dz, x_pos + dx),
                                   arrowstyle='->', 
                                   facecolor=color,
                                   edgecolor=color,
                                   fill=True,
                                   linewidth=2,
                                   mutation_scale=15,
                                   zorder=10)
            ax.add_patch(arrow)
        
        # Add label with filename
        # Offset label slightly to avoid overlap
        label_offset_z = 5
        label_offset_x = -3
        ax.text(z_pos + label_offset_z, x_pos + label_offset_x, 
               pos['filename'], fontsize=6, 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'),
               zorder=11)
        
        print(f"Plotted {pos['filename']} at z={z_pos:.1f}, x={x_pos:.1f}, angle={pos['angle']:.1f}°, live_time={pos['live_time']:.1f}s")
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Live Time [s]', rotation=270, labelpad=20)
    
    ax.set_title('Position Scan Measurements (Live Time > {} s)'.format(min_live_time), 
                fontsize=14, pad=10)
    
    return fig


if __name__ == "__main__":
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    # Generate diagram with all position scans
    fig = plot_all_position_scans(min_live_time=10)
    if fig:
        plt.savefig(os.path.join(outdir, "position_scans_all.png"), 
                   bbox_inches='tight', dpi=150)
        print(f"Saved position scan plot to {os.path.join(outdir, 'position_scans_all.png')}")
        plt.close(fig)
    else:
        print("Failed to generate position scan plot")