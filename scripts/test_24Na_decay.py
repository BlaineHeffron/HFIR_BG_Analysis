#!/usr/bin/env python3
"""
Test the hypothesis that the ~2755 keV line observed during reactor-off
periods is due to 24Na decay (half-life ≈ 15 hours).

This script:
- Loads a sequence of 8-hour spectra from the post-cycle reactor-off period.
- Fits the 2755 keV peak in each spectrum to extract its area (intensity).
- Displays (plots and prints) each individual peak fit.
- Collects timestamps for each measurement.
- Fits the intensities to an exponential decay model, allowing the half-life to float.
- Plots the data and fit to assess if it matches the expected decay, showing the best-fit half-life.

Usage:
    python3 test_24Na_decay.py

"""

import sys
import os
from os.path import dirname, realpath, join
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(1, dirname(dirname(realpath(__file__))))

from src.database.HFIRBG_DB import HFIRBG_DB
from src.utilities.util import get_data_dir, populate_data, fit_spectra

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
RUN_NAME = "PostCycle490C_MIF_8hour_sequence"
RUN_NUMBERS = list(range(363, 396))  # 363 to 395 inclusive
PEAK_ENERGY = 2755.0  # keV (approximate energy of 24Na gamma)

OUTDIR = join(os.environ.get("HFIRBG_ANALYSIS", "."), "decay_plots")
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

# ------------------------------------------------------------------
# Decay model with floating half-life
# ------------------------------------------------------------------
def exponential_decay(t, A0, half_life, bg):
    """Exponential decay with floating half-life, plus constant background."""
    lambda_ = np.log(2) / half_life
    return A0 * np.exp(-lambda_ * t) + bg

# ------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------
def main():
    db = HFIRBG_DB()
    datadir = get_data_dir()

    # Prepare rundata as dict with individual run numbers (do not combine)
    rundata = {f"run_{run_num}": [run_num] for run_num in RUN_NUMBERS}

    # Load spectra (one per run)
    spectra = populate_data(rundata, datadir, db)

    # Fit the peak in each spectrum and collect areas + timestamps
    areas = []
    area_errs = []
    times = []  # hours since first measurement
    start_times = []

    first_timestamp = None

    # Use fit_spectra to fit and display each peak
    for tag, spec in spectra.items():
        spec = spec[0]
        print(f"\n=== Fitting peak at {PEAK_ENERGY} keV in spectrum '{tag}' (live time: {spec.live/3600:.2f} hours) ===")

        # Fit the single peak using fit_spectra utility
        fit_results = fit_spectra({tag: spec}, [PEAK_ENERGY], plot_dir=None, user_verify=False, plot_fit=False, auto_set_offset=True)

        # Extract fit (assuming single peak)
        fit_key = 2755.0
        if fit_key not in fit_results:
            print(f"Warning: Fit failed for {tag}, skipping.")
            continue
        fit = fit_results[fit_key]  # PeakFit object

        # Get area and error
        area, area_err = fit.area()
        areas.append(area)
        area_errs.append(area_err)

        # Display the fit (print parameters and show/save plot)
        #print("\nFit Details:")
        #fit.display()
        #plot_path = join(OUTDIR, f"{tag}_2755keV_fit.png")
        #fit.plot(plot_path)
        #print(f"Fit plot saved to: {plot_path}")
        #plt.show()  # Display the plot interactively

        # Get start timestamp
        start_dt = datetime.strptime(spec.start.strip(), "%Y-%m-%d, %H:%M:%S")
        start_timestamp = start_dt.timestamp()
        start_times.append(start_dt)

        if first_timestamp is None:
            first_timestamp = start_timestamp

        # Time in hours since first measurement (use mid-point of acquisition)
        mid_time_hours = (start_timestamp - first_timestamp) / 3600 + spec.live / (2 * 3600)
        times.append(mid_time_hours)

    # Convert to arrays
    times = np.array(times)
    areas = np.array(areas)
    area_errs = np.array(area_errs)

    if len(times) < 3:
        print("Error: Not enough successful fits to perform decay analysis.")
        return

    # Sort by time (in case not already)
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    areas = areas[sort_idx]
    area_errs = area_errs[sort_idx]

    # Fit the decay curve with floating half-life
    p0 = [areas[0] - areas[-1], 15.0, areas[-1]]  # Initial guess: A0, half_life=15, bg
    try:
        params, cov = curve_fit(exponential_decay, times, areas, p0=p0, sigma=area_errs, absolute_sigma=True)
        errs = np.sqrt(np.diag(cov))
    except Exception as e:
        print(f"Fit failed: {e}")
        return

    A0, half_life, bg = params
    A0_err, half_life_err, bg_err = errs

    print("\n=== Decay Fit Results (Floating Half-Life) ===")
    print(f"A0: {A0:.2f} ± {A0_err:.2f} (initial amplitude)")
    print(f"Best-fit Half-Life: {half_life:.2f} ± {half_life_err:.2f} hours")
    print(f"bg: {bg:.2f} ± {bg_err:.2f} (constant background)")

    # Plot the data and fit
    plt.figure(figsize=(10, 6))
    plt.errorbar(times, areas, yerr=area_errs, fmt='o', label='Data', capsize=5)
    
    t_fit = np.linspace(0, max(times), 100)
    y_fit = exponential_decay(t_fit, *params)
    plt.plot(t_fit, y_fit, 'r-', label=f'Fit: A0 exp(-λt) + bg\nBest-fit T_{{1/2}} = {half_life:.2f} ± {half_life_err:.2f} h')
    
    plt.xlabel('Time since first measurement (hours)')
    plt.ylabel('Peak Area Rate (counts/keV/live s)')
    plt.title('Decay of 2755 keV Peak (Hypothesized 24Na)')
    plt.legend()
    plt.grid(True)
    
    plot_path = join(OUTDIR, "24Na_decay_fit.png")
    plt.savefig(plot_path)
    plt.show()
    
    print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()