#!/usr/bin/env python3
import sys
import os
from os.path import dirname, realpath
import argparse
from math import sqrt

sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.database.HFIRBG_DB import HFIRBG_DB
from src.utilities.util import populate_data_db, combine_runs, fit_spectra, get_peak_fit_data, unc_ratio

RUNS_TO_USE = ['Cycle493_RD_low_gain', 'PreCycle494_RD_low_gain']

# Energy display (exact as in your LaTeX), ID (with \el{...} notation), and yield strings
PEAKS = [
    # (energy_float, energy_disp, id_disp, yield_disp)
    (7916.3,  "7916.3",  r"\el{Cu}{63} n,g",                    "100"),
    (7367.9,  "7367.9",  r"\el{Pb}{207} n,g",                   "100"),
    (5824.6,  "5824.6",  r"\el{Cd}{113} n,g",                   "3.96"),
    (5433.1,  "5433.1",  r"\el{Cd}{113} n,g",                   "1.55"),
    (2767.5,  "2767.5",  r"\el{Cd}{113} n,g",                   "1.5"),
    (2660.1,  "2660.1",  r"\el{Cd}{113} n,g",                   "3.7"),
    (2614.533,"2614.533",r"\el{Tl}{208} b-",                    "100"),
    (2550.1,  "2550.1",  r"\el{Cd}{113} n,g",                   "2.05"),
    (2455.8,  "2455.8",  r"\el{Cd}{113} n,g",                   "0.511"),
    (2398.6,  "2398.6",  r"\el{Cd}{113} n,g",                   "1.03"),
    (2223.0,  "2223",    r"\el{H}{1} n,g",                      "100"),
    (2204.2,  "2204.2",  r"\el{Bi}{214} b-",                    "5.08"),
    (1764.5,  "1764.5",  r"\el{Bi}{214} b-",                    "15.4"),
    (1660.368,"1660.368",r"\el{Cd}{113} n,g",                   "3.75"),
    (1489.56, "1489.56", r"\el{Cd}{113} n,g",                   "3.19"),
    (1399.6,  "1399.6",  r"\el{Cd}{113} n, g",                  "4.67"),
    (1377.7,  "1377.7",  r"\el{Bi}{214} b-",                    "4"),
    (1364.3,  "1364.3",  r"\el{Cd}{113} n,g",                   "6.23"),
    (1293.6,  "1293.6",  r"\el{Ar}{41} b-",                     "99.1"),
    (1281.0,  "1281",    r"\el{Bi}{214} b-",                    "1.43"),
    (1238.1,  "1238.1",  r"\el{Bi}{214} b-",                    "5.8"),
    (1209.7,  "1209.7",  r"\el{Cd}{113} n,g",                   "5.57"),
    (1120.3,  "1120.3",  r"\el{Bi}{214} b-",                    "15.1"),
    (805.9,   "805.9",   r"\el{Cd}{113} n,g",                   "6.8"),
    (768.4,   "768.4",   r"\el{Bi}{214} b-",                    "4.9"),
    (725.0,   "725",     r"\el{Cd}{113} n,g",                   "5.99"),
    (707.4,   "707.4",   r"\el{Cd}{113} n,g",                   "1.55"),
    (651.3,   "651.3",   r"\el{Cd}{113} n,g",                   "18.9"),
    (609.3,   "609.3",   r"\el{Bi}{214} b-",                    "46.1"),
    (558.5,   "558.5",   r"\el{Cd}{113} n,g",                   "100"),
    (478.0,   "478",     r"\el{B}{10}(n,$\\alpha$)\el{Li}{7}",  "94"),
    (351.9,   "351.9",   r"\el{Pb}{214} b-",                    "37.6"),
    (295.2,   "295.2",   r"\el{Pb}{214} b-",                    "19.3"),
    (242.0,   "242",     r"\el{Pb}{214} b-",                    "7.43"),
    (238.6,   "238.6",   r"\el{Pb}{212} b-",                    "43.3"),
]
EXPECTED_PEAKS = [e for e, _, _, _ in PEAKS]

def fmt_key(e):
    return "{:.2f}".format(float(e))

def get_low_gain_specs(db: HFIRBG_DB, rxon_only=False, rxoff_only=False):
    # Combine like scripts/calibrate_rd.py
    rd = db.get_rd_files(True, gain_setting=['low'], rxon_only=rxon_only, rxoff_only=rxoff_only)
    rd = {key: value for key, value in rd.items() if key in RUNS_TO_USE}
    rd = populate_data_db(rd, db)
    combine_runs(rd, max_interval=None)
    out = []
    for v in rd.values():
        if isinstance(v, list):
            out.extend(v)
        else:
            out.append(v)
    return out

def fit_relative_heights(specs, expected_peaks, ref_energy):
    ref_key = fmt_key(ref_energy)
    sum_w = {fmt_key(e): 0.0 for e in expected_peaks}
    sum_wr = {fmt_key(e): 0.0 for e in expected_peaks}

    for spec in specs:
        try:
            peak_fits = fit_spectra({spec.fname: spec}, expected_peaks,
                                    plot_dir=None, user_verify=True, plot_fit=False)
        except Exception:
            continue
        pdict = {}
        get_peak_fit_data(peak_fits, pdict)
        if ref_key not in pdict:
            continue
        href, dhref = pdict[ref_key][0], pdict[ref_key][1]
        if href <= 0:
            continue

        for e in expected_peaks:
            k = fmt_key(e)
            if k not in pdict:
                continue
            h, dh = pdict[k][0], pdict[k][1]
            if h <= 0:
                continue
            r = h / href
            dr = unc_ratio(h, href, dh, dhref)
            if dr is None or dr <= 0 or not (dr == dr):
                continue
            w = 1.0 / (dr * dr)
            sum_w[k] += w
            sum_wr[k] += r * w

    out = {}
    for e in expected_peaks:
        k = fmt_key(e)
        if sum_w[k] > 0:
            out[k] = (sum_wr[k] / sum_w[k], sqrt(1.0 / sum_w[k]))
        else:
            out[k] = None
    return out

def render_latex_table(rel_on, rel_off):
    print("\\begin{tabular}{llllll}")
    print("\\toprule")
    print("& energy [keV] & ID & yield [\\%]~\\cite{nndc} & rel. height (Rx ON) & rel. height (Rx OFF) \\\\")
    print("\\midrule")
    for e, e_disp, id_disp, yld in PEAKS:
        k = fmt_key(e)
        on_str = "-"
        off_str = "-"
        if rel_on.get(k) is not None:
            mu, err = rel_on[k]
            on_str = f"{mu:.3f} $\\pm$ {err:.3f}"
        if rel_off.get(k) is not None:
            mu, err = rel_off[k]
            off_str = f"{mu:.3f} $\\pm$ {err:.3f}"
        print(f"& {e_disp} & {id_disp} & {yld} & {on_str} & {off_str} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

def main():
    parser = argparse.ArgumentParser(
        description="Add relative peak heights (background-subtracted fit amplitudes) for RD low-gain runs (Rx ON/OFF) to your LaTeX table."
    )
    parser.add_argument("--ref-energy", type=float, default=558.5,
                        help="Reference peak energy in keV (default: 558.5 keV)")
    args = parser.parse_args()

    db = HFIRBG_DB()
    specs_on = get_low_gain_specs(db, rxon_only=True, rxoff_only=False)
    specs_off = get_low_gain_specs(db, rxon_only=False, rxoff_only=True)

    rel_on = fit_relative_heights(specs_on, EXPECTED_PEAKS, args.ref_energy)
    rel_off = fit_relative_heights(specs_off, EXPECTED_PEAKS, args.ref_energy)

    render_latex_table(rel_on, rel_off)
    print(f"% Reference energy for relative heights: {args.ref_energy:.3f} keV")

if __name__ == "__main__":
    main()