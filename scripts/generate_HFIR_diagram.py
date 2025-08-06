import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.ticker import AutoMinorLocator
from matplotlib import colors
from math import pi, sin, cos
import numpy as np
import os
import sys
from os.path import dirname, realpath, join

# Add src to path for database imports
sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.database.HFIRBG_DB import HFIRBG_DB
from src.database.CartScanFiles import convert_cart_coord_to_det_coord, convert_coord_to_phi

outdir = os.path.join(os.environ["HFIRBG_ANALYSIS"], "diagrams")

def HFIR_diagram():
    xmin = -5
    ymin = -160
    xmax = 320
    ymax = 180
    corez = 141.15
    corex = -148.5
    corer = 8.66 #in
    # rcParams.update({'font.size': 14})
    ratio = abs((ymax - ymin) / (xmax - xmin))# * 4. / 5
    fig = plt.figure(figsize=(7, 7 * ratio))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel("z [in]")
    ax1.set_ylabel("x [in]")
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymax, ymin)
    #core
    ax1.add_patch(Circle((corez, corex), corer,
                         edgecolor='black',
                         fill=False,
                         lw=2))
    # HB4
    HB4line = [(240.56 + np.tan(30 * pi / 180) * x, x) for x in np.linspace(corex -10, ymax, 20)]
    # HB3
    HB3line = [(67.9 - np.tan(30 * pi / 180) * x, x) for x in np.linspace(corex +10, ymax, 20)]
    # "shutter drive rod sleeve" coming off hb3
    # drline = [(60.35 + np.sin(30*pi/180)*x, 13.0 + np.cos(30*pi/180)*x) for x in range(0, 113+10, 12)]
    ax1.plot([a[0] for a in HB4line], [a[1] for a in HB4line], linestyle='dashed', color='grey')
    ax1.plot([a[0] + 4 for a in HB4line], [a[1] for a in HB4line], linestyle='dashed', color='grey')
    ax1.plot([a[0] for a in HB3line], [a[1] for a in HB3line], linestyle='dashed', color='grey')
    ax1.plot([a[0] + 4 for a in HB3line], [a[1] for a in HB3line], linestyle='dashed', color='grey')
    ax1.text(90, -80, 'HB3', fontsize=15, color='black')
    ax1.text(190, -100, 'HB4', fontsize=15, color='black')
    ax1.text(50, -145, 'Reactor Core', fontsize=15, color='black')
    # ax1.plot([a[0] for a in drline], [a[1] for a in drline], linestyle='dashed', color='grey')
    # ax1.plot([a[0] + 4 for a in drline], [a[1] for a in drline], linestyle='dashed', color='grey')
    # next lets plot the monolith, passes through z = 141.5, x = 111, .1293 x/z
    monolith_line1 = [(141.5 + a, 111 - a * .1293) for a in [-44.7, 67.1]]
    monolith_line2 = [(monolith_line1[0][0] - a, monolith_line1[0][1] - a * .50) for a in [0, 100]]
    monolith_line3 = [(monolith_line1[1][0] + a, monolith_line1[1][1] - a * .58) for a in [0, 150]]
    ax1.plot([a[0] for a in monolith_line1], [a[1] for a in monolith_line1], linestyle='solid', color='grey')
    ax1.plot([a[0] for a in monolith_line2], [a[1] for a in monolith_line2], linestyle='solid', color='grey')
    ax1.plot([a[0] for a in monolith_line3], [a[1] for a in monolith_line3], linestyle='solid', color='grey')
    ax1.text(30, 130, 'Monolith Boundary', fontsize=15, color='black')
    #pool wall
    ax1.plot([0,xmax],[0, 0], linestyle='solid', color='black')
    ax1.plot([0,0],[0, -200], linestyle='solid', color='black')
    ax1.text(110, -3, 'Reactor Pool Wall', fontsize=15, color='black')
    # add box for prospect
    ax1.add_patch(Rectangle((165, 128), 46.25, -83.4,
                            edgecolor='cornflowerblue',
                            fill=False,
                            lw=2))
    ax1.text(99, 95, 'PROSPECT\nScintillator\nVolume', fontsize=15, color='black')
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
    ax1.text(76, 10, 'MIF', color='black', fontsize=15)
    ax1.text(145, 18, 'Pb Wall', color='black', fontsize=15)
    ax1.text(20, 18, 'Pb Wall', color='black', fontsize=15)
    # russian doll
    ax1.add_patch(Circle((200, 21.5 + 12), 12,
                         edgecolor='green',
                         fill=False,
                         lw=2))
    ax1.text(72, 38, 'Russian Doll Shield', fontsize=15, color='black')
    # ax1.set_title(title)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis="x", direction="in", length=16, width=1)
    ax1.tick_params(axis="y", direction="in", length=16, width=1)
    ax1.tick_params(axis="x", which="minor", direction="in", length=8, width=1)
    ax1.tick_params(axis="y", which="minor", direction="in", length=8, width=1)
    # plt.subplots_adjust(bottom=-.1)
    # plt.tight_layout()
    return fig

def HFIR_diagram_with_detectors():
    """
    Create HFIR diagram with arrows showing detector orientations for key measurement positions
    """
    # Create a clean version without text labels for the detector diagram
    xmin = -5
    ymin = -160
    xmax = 320
    ymax = 180
    corez = 141.15
    corex = -148.5
    corer = 8.66 #in
    ratio = abs((ymax - ymin) / (xmax - xmin))
    fig = plt.figure(figsize=(7, 7 * ratio))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel("z [in]")
    ax1.set_ylabel("x [in]")
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymax, ymin)
    
    #core
    ax1.add_patch(Circle((corez, corex), corer,
                         edgecolor='black',
                         fill=False,
                         lw=2))
    # HB4
    HB4line = [(240.56 + np.tan(30 * pi / 180) * x, x) for x in np.linspace(corex -10, ymax, 20)]
    # HB3
    HB3line = [(67.9 - np.tan(30 * pi / 180) * x, x) for x in np.linspace(corex +10, ymax, 20)]
    
    ax1.plot([a[0] for a in HB4line], [a[1] for a in HB4line], linestyle='dashed', color='grey')
    ax1.plot([a[0] + 4 for a in HB4line], [a[1] for a in HB4line], linestyle='dashed', color='grey')
    ax1.plot([a[0] for a in HB3line], [a[1] for a in HB3line], linestyle='dashed', color='grey')
    ax1.plot([a[0] + 4 for a in HB3line], [a[1] for a in HB3line], linestyle='dashed', color='grey')
    
    # monolith (no text labels)
    monolith_line1 = [(141.5 + a, 111 - a * .1293) for a in [-44.7, 67.1]]
    monolith_line2 = [(monolith_line1[0][0] - a, monolith_line1[0][1] - a * .50) for a in [0, 100]]
    monolith_line3 = [(monolith_line1[1][0] + a, monolith_line1[1][1] - a * .58) for a in [0, 150]]
    ax1.plot([a[0] for a in monolith_line1], [a[1] for a in monolith_line1], linestyle='solid', color='grey')
    ax1.plot([a[0] for a in monolith_line2], [a[1] for a in monolith_line2], linestyle='solid', color='grey')
    ax1.plot([a[0] for a in monolith_line3], [a[1] for a in monolith_line3], linestyle='solid', color='grey')
    
    #pool wall
    ax1.plot([0,xmax],[0, 0], linestyle='solid', color='black')
    ax1.plot([0,0],[0, -200], linestyle='solid', color='black')
    
    # add box for prospect (no text)
    ax1.add_patch(Rectangle((165, 128), 46.25, -83.4,
                            edgecolor='cornflowerblue',
                            fill=False,
                            lw=2))
    
    # lead shield wall (no text)
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
    
    # russian doll (no text)
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
    
    # Initialize database
    db = HFIRBG_DB()
    
    # Define key measurement positions and their labels
    key_positions = {
        "HB4_DOWN_OVERNIGHT_1.txt": {"label": "HB4 Hotspot", "color": "red"},
        "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN.txt": {"label": "MIF", "color": "blue"}, 
        "CYCLE461_DOWN_FACING_OVERNIGHT.txt": {"label": "Shield Center", "color": "brown"},
        #"SW_1": {"label": "SW 1", "color": "orange"},
        #"EAST_FACE_2": {"label": "EAST 2", "color": "brown"},
    }
    
    arrow_length = 20  # Length of arrows in plot units
    legend_handles = []
    legend_labels = []
    color_map = {
        "red": (1.0, 0.0, 0.0, 1.0),    # For HB4 Hotspot
        "blue": (0.0, 0.0, 1.0, 1.0),   # For MIF
        "brown": (0.647, 0.165, 0.165, 1.0)  # For Shield Center
    }

    for filename, info in key_positions.items():
        try:
            # Get file metadata from database
            metadata = db.retrieve_file_metadata(filename.replace('.txt', ''))
            
            if metadata:
                # Extract coordinates and angle
                Rx, Rz, Lx, Lz = metadata["Rx"], metadata["Rz"], metadata["Lx"], metadata["Lz"]
                angle = metadata["angle"]
                
                # Convert to detector coordinates (z, x in plot coordinates)
                det_coords = convert_cart_coord_to_det_coord(Rx, Rz, Lx, Lz, angle)
                x_pos, z_pos = det_coords[0], det_coords[1]
                
                # Calculate phi (cart angle) for arrow direction
                phi_deg = convert_coord_to_phi(Rx, Rz, Lx, Lz)
                phi_rad = phi_deg * pi / 180
                
                # Get the RGBA tuple
                rgba = color_map[info["color"]]
                
                # For pointing straight down, use a star instead of arrow
                if angle == 0:
                    scatter = ax1.scatter(z_pos, x_pos, marker='*', s=200, c=[rgba], 
                                        edgecolors='black', linewidth=1, zorder=10)
                    legend_handles.append(scatter)
                    legend_labels.append(info["label"])
                else:
                    # Calculate arrow direction
                    dz = arrow_length * cos(phi_rad)  # West is negative z
                    dx = -arrow_length * sin(phi_rad)  # North is negative x
                    
                    # Create arrow with explicit facecolor and edgecolor (no 'color' parameter to avoid conflicts)
                    arrow = FancyArrowPatch((z_pos, x_pos), 
                                            (z_pos + dz, x_pos + dx),
                                            arrowstyle='->', 
                                            facecolor=rgba,
                                            edgecolor=rgba,
                                            fill=True,  # Explicitly ensure the arrowhead is filled
                                            linewidth=2,
                                            mutation_scale=15,
                                            zorder=10)
                    ax1.add_patch(arrow)
                    
                    # Add a scatter point for the legend using RGBA
                    scatter = ax1.scatter(z_pos, x_pos, marker='o', s=50, c=[rgba], 
                                        edgecolors='black', linewidth=1, zorder=10)
                    legend_handles.append(scatter)
                    legend_labels.append(info["label"])
                    
                    print(f"Added {info['label']} at z={z_pos:.1f}, x={x_pos:.1f}, phi={phi_deg:.1f}° with RGBA {rgba}")
                    
        except Exception as e:
            print(f"Could not add detector position for {filename}: {e}")
        
    # Add legend using custom handles
    ax1.legend(legend_handles, legend_labels, loc='upper right', fontsize=12)
    
    return fig


if __name__ == "__main__":
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    # Generate original diagram
    fig = HFIR_diagram()
    plt.savefig(os.path.join(outdir,"HFIR_diagram.png"), bbox_inches='tight')
    plt.close(fig)
    
    # Generate diagram with detector orientations
    fig_with_detectors = HFIR_diagram_with_detectors()
    plt.savefig(os.path.join(outdir,"HFIR_diagram_with_detectors.png"), bbox_inches='tight')
    plt.close(fig_with_detectors)