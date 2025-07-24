#!/usr/bin/env python3
"""
Fit the 437 keV, broad 478 keV and 482/484 keV features in a single pass.

Usage examples
--------------

(1)  Same CLI style as peak_finder.py – edit the ‘rundata’ or ‘config’
     blocks below and just run

        $ python3 wide_peak_fitter.py

(2)  Or import the module and call main() from an interactive session.

Author: <you>
Date  : <today>
"""
import sys
import os
from os.path import dirname, realpath, join

sys.path.insert(1, dirname(dirname(realpath(__file__))))   # project root

from src.database.HFIRBG_DB import HFIRBG_DB
from src.utilities.util      import (get_data_dir, populate_data,
                                     populate_data_config, combine_runs)
from src.analysis.Spectrum   import CustomSigmaSpectrumFitter


# ------------------------------------------------------------------
# 1.  What do we want to fit?
# ------------------------------------------------------------------
PEAKS_TO_FIT      = [463, 478.0, 482.0]           # keV
CUSTOM_SIGMAS_KEV = {478.0: 5.1}                           # 12 keV FWHM ≈ 5.1 keV σ


# ------------------------------------------------------------------
# 2.  Which spectrum files?
#     – uncomment one of the two examples or edit to taste
# ------------------------------------------------------------------
# Example A: use the convenience “config’’ mechanism
CONFIG = {"runs": {"name": "PreCycle494_RD_low_gain"}}
RUNDATA = None                     # leave this None when CONFIG is used

# Example B: explicit list of files
# RUNDATA = {"PROSPECT": "EAST_FACE_1.txt"}
# CONFIG   = None

outdir = join(os.environ["HFIRBG_ANALYSIS"], "fit_plots")

# ------------------------------------------------------------------
# 3.  Main
# ------------------------------------------------------------------
def main(verify=False, plot=True):
    db       = HFIRBG_DB()
    data_dir = get_data_dir()

    if CONFIG:
        spectra = populate_data_config(CONFIG, db, comb_runs=True)
    else:
        spectra = populate_data(RUNDATA, data_dir, db)
        combine_runs(spectra)

    # ----------------------------------------------------------------
    # Fit each spectrum with the custom-sigma fitter
    # ----------------------------------------------------------------
    fitter = CustomSigmaSpectrumFitter(PEAKS_TO_FIT,
                                       custom_sigmas=CUSTOM_SIGMAS_KEV)

    all_fit_results = {}

    for tag, spec in spectra.items():
        print(f"\n=== fitting spectrum '{tag}' ===")
        fitter.name = tag                 # so the plots get reasonable names
        fitter.fit_peaks(spec)

        # Optional user verification / plot
        if verify or plot:
            # produces PNGs in the current directory
            fitter.plot_fits(user_prompt=verify, write_to_dir=outdir)

        # collect the results
        all_fit_results.update(fitter.fit_values)

    # ----------------------------------------------------------------
    # Print a compact summary
    # ----------------------------------------------------------------
    print("\n\n-------------  Fit summary  -------------")
    for peak, fit in all_fit_results.items():
        print(f"\nPeak(s) centred at {peak} keV")
        fit.display()

    return all_fit_results


# ------------------------------------------------------------------
if __name__ == "__main__":
    # set verify=True if you want the interactive yes/no dialogue
    main(verify=True, plot=True)