import sys
from os.path import dirname, realpath, join
import os

sys.path.insert(1, dirname(dirname(realpath(__file__))))

from src.database.HFIRBG_DB import HFIRBG_DB
from src.utilities.util import *

# Define the locations and their corresponding file names for rx on and off
# These are selected based on the provided JSON and scripts
locations = {
    "High Rate (MIF facing reactor)": {
        "on": "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN",
        "off": "MIF_BOX_AT_REACTOR_RXOFF"
    },
    "Low Rate (Near Shield Wall)": {
        "on": "CYCLE461_DOWN_FACING_OVERNIGHT",  # Shield center down
        "off": "NOMINAL_NORTH_RXOFF_2DAY"  # Using nominal rxoff as proxy for low rate off
    },
    "Russian Doll": {
        "on": [i for i in range(4395, 4409)],  # Cycle494_RD_low_gain_lead
        "off": [i for i in range(4410, 4427)]  # PreCycle495_RD_low_gain_lead
    },
    "HB4 Hotspot": {
        "on": "HB4_DOWN_OVERNIGHT_1",
        "off": "HB4_HOT_SPOT_RXOFF"
    }
}

emin = 100  # keV
emax = 11500  # keV
detector_mass = 1.0  # kg
mass_uncertainty = 0.05  # kg (estimated 5% uncertainty)

# For plotting
plot_dir = join(os.environ["HFIRBG_ANALYSIS"], "rate_comparison_plots")
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)
bins = get_bins(100, 11500, 11400)

def compute_rate_density(spec, emin, emax, mass, dmass):
    """
    Compute the average rate density in cts/s/keV/kg, propagating mass uncertainty
    """
    integrated_rate, d_rate = spec.integrate(emin, emax, norm=True)
    rate_density = integrated_rate / mass
    # Propagate errors: relative error from rate and mass
    rel_err_rate = d_rate / integrated_rate
    rel_err_mass = dmass / mass
    err_density = rate_density * np.sqrt(rel_err_rate**2 + rel_err_mass**2)
    return rate_density, err_density

def main():
    db = HFIRBG_DB()
    datadir = get_data_dir()
    
    results = {}
    for loc, files in locations.items():
        results[loc] = {}
        spectra = {}
        for state in ["on", "off"]:
            if isinstance(files[state], str):
                # Single file
                spec = retrieve_data(join(datadir, files[state] + ".txt"), db)
            else:
                # List of files, combine them
                rundata = {loc + "_" + state: files[state]}
                data = populate_data(rundata, datadir, db)
                combine_runs(data)
                spec = data[loc + "_" + state]
            
            spectra[state] = spec
            rate_density, err_density = compute_rate_density(spec, emin, emax, detector_mass, mass_uncertainty)
            results[loc][state] = (rate_density, err_density)
        
        # Plot the spectra for this location
        plot_data = {
            loc + " Rx On": spectra["on"],
            loc + " Rx Off": spectra["off"]
        }
        plot_multi_spectra(plot_data, join(plot_dir, loc.replace(" ", "_") + "_spectrum_comparison"), rebin_edges=bins, emin=emin, emax=emax, ylog=True)
    
    # Print LaTeX table
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{|l|c|c|}")
    print("\\hline")
    print("Location & Rx On (cts/s/keV/kg) & Rx Off (cts/s/keV/kg) \\\\ \\hline")
    for loc in results:
        on_val, on_err = results[loc]["on"]
        off_val, off_err = results[loc]["off"]
        print(f"{loc} & {on_val:.4e} $\\pm$ {on_err:.4e} & {off_val:.4e} $\\pm$ {off_err:.4e} \\\\ \\hline")
    print("\\end{tabular}")
    print("\\caption{Overall rates in the energy range 100 keV to 11.5 MeV.}")
    print("\\label{tab:rate_comparison}")
    print("\\end{table}")

if __name__ == "__main__":
    main()