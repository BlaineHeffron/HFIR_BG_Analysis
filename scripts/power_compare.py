import sys
from os.path import dirname, realpath
sys.path.insert(1, dirname(dirname(realpath(__file__))))

from src.database.SqliteManager import HFIRBG_DB
from src.utilities.util import *
from math import floor
from src.utilities.PlotUtils import MultiScatterPlot
import matplotlib.pyplot as plt
import argparse
from ntpath import basename

pwr = {10: [17, 19], 30: [20], 50: [21], 70: [22], 90: [23], 100: [28]}
ENERGY_RANGES = [i * 400. + 50. for i in range(10)]
RANGE_LABELS = ["{0} - {1}".format(ENERGY_RANGES[i], ENERGY_RANGES[i + 1]) for i in range(len(ENERGY_RANGES) - 1)]


def find_power(n):
    for pwrlevel in pwr:
        if n in pwr[pwrlevel]:
            return pwrlevel
    return 0


def energy_index(e, A0, A1):
    """retrieves the index of the next energy bin value greater than e"""
    return int(floor((e - A0) / A1))


def integrate_ranges(ene_range, data, A0, A1):
    results = np.zeros((len(ene_range) - 1), )
    for i in range(len(ene_range) - 1):
        ind0 = energy_index(ene_range[i], A0, A1)
        if ind0 < 0:
            ind0 = 0
        ind1 = energy_index(ene_range[i + 1], A0, A1)
        results[i] = np.sum(data[ind0:ind1])
    return results


def retrieve_power_data(ene_range, fs, db=None):
    pwr_data = {0: np.zeros((len(ene_range) - 1,), dtype=np.float32)}
    live_time = {0: 0.}
    for key in pwr:
        pwr_data[key] = np.zeros((len(ene_range) - 1,), dtype=np.float32)
        live_time[key] = 0.
    for f in fs:
        spec = retrieve_data(f, db)
        l, A0, A1, data = spec.live, spec.A0, spec.A1, spec.data
        power = find_power(file_number(basename(f)))
        live_time[power] += l
        pwr_data[power] += integrate_ranges(ene_range, data, A0, A1)
    return pwr_data, live_time


def calc_rate_errors(pwr, lt):
    error = {}
    for key in pwr:
        error[key] = np.divide(np.sqrt(pwr[key]), lt[key])
        pwr[key] = np.divide(pwr[key], lt[key])
    return pwr, error


def plot_power_data(pwr, lt, write=True):
    bin_width = ENERGY_RANGES[1] - ENERGY_RANGES[0]
    midpoints = get_bin_midpoints(ENERGY_RANGES[0], ENERGY_RANGES[-1], len(ENERGY_RANGES) - 1)
    labels = ["{}% power".format(key) for key in pwr.keys()]
    rates, errors = calc_rate_errors(pwr, lt)
    nonzerokeys = [key for key in rates.keys() if key != 0]
    ys = [[(rates[key][i] - rates[0][i]) / bin_width for i in range(rates[key].shape[0])] for key in nonzerokeys]
    errs = [[(errors[key][i] - errors[0][i]) / bin_width for i in range(errors[key].shape[0])] for key in nonzerokeys]
    fig = MultiScatterPlot(midpoints, ys, errs, labels[1:], "energy midpoint [keV]", "rate [hz/keV]", xmin=0, xmax=3700,
                           title="rxoff subtracted rates (400 keV bins)", ylog=True)
    plt.savefig("power_rates.png")
    if write:
        for key in rates.keys():
            fout = "power_{}_rates.csv".format(key)
            write_x_y_csv(fout, "energy midpoint [keV]", "rate [hz]", "error [hz]", midpoints, rates[key],
                          errors[key])


def plot_power_spectra(files):
    file_sets = {0: []}
    for key in pwr:
        file_sets[key] = []
    for f in files:
        power = find_power(file_number(basename(f)))
        file_sets[power].append(f)
    for power in file_sets:
        plot_spectra(file_sets[power], "{}_power".format(power), rebin=20)

    multi_plot = {"rx on": file_sets[100], "rx off": file_sets[0]}
    multi_plot = populate_data(multi_plot)
    plot_multi_spectra(multi_plot, "rxonvsoff.png", rebin=20)


def main():
    datadir = get_data_dir()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="path to data")
    args = parser.parse_args()
    db = HFIRBG_DB()
    if not args.data:
        files = retrieve_files(datadir)
    else:
        files = retrieve_files(args.data)
    plot_power_spectra(files)
    pwr, lt = retrieve_power_data(ENERGY_RANGES, files, db)
    plot_power_data(pwr, lt)


if __name__ == "__main__":
    main()
