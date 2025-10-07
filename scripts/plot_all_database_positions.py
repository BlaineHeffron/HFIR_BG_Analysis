import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from matplotlib import cm
from math import pi, sin, cos
import numpy as np
import os
import sys
from os.path import dirname, realpath, join

# Add src to path for database imports
sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.database.HFIRBG_DB import HFIRBG_DB
from src.database.CartScanFiles import convert_cart_coord_to_det_coord, convert_coord_to_phi
from scripts.plot_all_position_scans import HFIR_diagram_base

outdir = os.path.join(os.environ["HFIRBG_ANALYSIS"], "diagrams")


def plot_all_database_positions(min_live_time=3600):
    """Plot all positions from database with live time > min_live_time"""
    
    # Initialize database
    db = HFIRBG_DB()
    
    # Query all position files from database
    position_data = db.query_position_files()
    
    if not position_data:
        print("No position files found in database!")
        return
    

    # Filter by live time and collect valid positions
    positions = []
    for row in position_data:
        if row['A1'] < 0.6:  # higher gain, only interested in low gain runs
            continue
        if row['start_time'] < 0:  # corrupted data
            continue
        if row['track'] == 1:  # skip track scans
            continue
        if 'lead' in row['run_description']:  # skip lead test runs
            continue
        
        live_time = float(row['live_time'])
        
        if live_time > min_live_time:
            # Check for valid calibration
            if row['A0'] is None or row['A1'] is None:
                print(f"Warning: No calibration found for file {row['filename']}, skipping")
                continue
            
            # Convert to detector coordinates
            det_coords = convert_cart_coord_to_det_coord(
                row['Rx'], row['Rz'], row['Lx'], row['Lz'], row['angle']
            )
            
            # Calculate phi (cart angle) for arrow direction
            phi_deg = convert_coord_to_phi(row['Rx'], row['Rz'], row['Lx'], row['Lz'])
            
            pos_data = {
                'x_pos': det_coords[0],
                'z_pos': det_coords[1],
                'angle': row['angle'],
                'phi_deg': phi_deg,
                'live_time': live_time,
                'filename': row['filename'],
                'run_name': row['run_name'],
                'shield_name': row['shield_name']
            }
            positions.append(pos_data)
    
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
        # Get color based on live time
        color = cmap(norm(pos['live_time']))
        
        x_pos = pos['x_pos']
        z_pos = pos['z_pos']
        
        # Check for invalid coordinates
        if not (np.isfinite(x_pos) and np.isfinite(z_pos)):
            print(f"Warning: Skipping {pos['filename']} due to invalid coordinates: x={x_pos}, z={z_pos}")
            continue
        
        # For pointing straight down, use a star
        if pos['angle'] == 0:
            ax.scatter(z_pos, x_pos, marker='*', s=300, c=[color], 
                      edgecolors='black', linewidth=1, zorder=10)
        else:
            # Calculate arrow direction based on phi
            phi_rad = pos['phi_deg'] * pi / 180
            
            # Calculate arrow direction
            dz = arrow_length * cos(phi_rad)
            dx = -arrow_length * sin(phi_rad)
            
            # Validate arrow direction
            if not (np.isfinite(dx) and np.isfinite(dz)):
                print(f"Warning: Invalid arrow direction for {pos['filename']}: dx={dx}, dz={dz}, phi={pos['phi_deg']}")
                # Fall back to marker
                ax.scatter(z_pos, x_pos, marker='o', s=100, c=[color], 
                          edgecolors='black', linewidth=1, zorder=10)
            else:
                # Create arrow
                try:
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
                except Exception as e:
                    print(f"Warning: Failed to create arrow for {pos['filename']}: {e}")
                    print(f"  Position: ({z_pos}, {x_pos}), Arrow end: ({z_pos + dz}, {x_pos + dx})")
                    # Fall back to marker
                    ax.scatter(z_pos, x_pos, marker='o', s=100, c=[color], 
                              edgecolors='black', linewidth=1, zorder=10)
        
        # Add label with filename
        # Offset label slightly to avoid overlap
        label_offset_z = 5
        label_offset_x = -3
        ax.text(z_pos + label_offset_z, x_pos + label_offset_x, 
               pos['filename'], fontsize=6, 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'),
               zorder=11)
        
        print(f"Plotted {pos['filename']} ({pos['run_name']}) at z={z_pos:.1f}, x={x_pos:.1f}, "
              f"angle={pos['angle']:.1f}°, phi={pos['phi_deg']:.1f}°, live_time={pos['live_time']:.1f}s, "
              f"shield={pos['shield_name']}")
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Live Time [s]', rotation=270, labelpad=20)
    
    ax.set_title('All Database Position Measurements (Live Time > {} s)'.format(min_live_time), 
                fontsize=14, pad=10)
    
    return fig


if __name__ == "__main__":
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    # Generate diagram with all database positions
    fig = plot_all_database_positions(min_live_time=3600)
    if fig:
        plt.savefig(os.path.join(outdir, "database_positions_all.png"), 
                   bbox_inches='tight', dpi=150)
        print(f"Saved database position plot to {os.path.join(outdir, 'database_positions_all.png')}")
        plt.close(fig)
    else:
        print("Failed to generate database position plot")