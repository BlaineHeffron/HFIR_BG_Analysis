import sys
from os.path import dirname, realpath
sys.path.insert(1, dirname(dirname(realpath(__file__))))

from src.database.HFIRBG_DB import HFIRBG_DB
from src.utilities.util import *
from math import floor
from src.utilities.PlotUtils import MultiScatterPlot
import matplotlib.pyplot as plt
from ntpath import basename

#pwr = {10: [17, 19], 30: [20], 50: [21], 70: [22], 90: [23], 100: [28]}
pwr = {0: [388 + i for i in range(395-388)], 5: [638], 10: [639,640], 20: [641], 40: [642], 60: [643], 80: [644], 95: [645], 100: [747+i for i in range(5)]}
pwr_unc = {5:2.5, 10:0.1, 20:5, 40: 5,60:5,80:5,95:2.5,100:0.5}
pwr_names = {0:"0",5:"0-10", 10:"10", 20:"10-30", 40: "30-50",60:"50-70",80:"70-90",95:"90-100",100:"100"}
ENERGY_RANGES = [i * 500. + 100. for i in range(24)]
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


def retrieve_power_data(ene_range, datadir, fs, db=None):
    pwr_data = {}
    live_time = {}
    for key in pwr:
        pwr_data[key] = np.zeros((len(ene_range) - 1,), dtype=np.float32)
        live_time[key] = 0.
    pwr_spec = populate_data(pwr, datadir, db)
    combine_runs(pwr_spec)
    for power, spec in pwr_spec.items():
        l, A0, A1, data = spec.live, spec.A0, spec.A1, spec.data
        #power = find_power(file_number(basename(f)))
        live_time[power] += l
        pwr_data[power] += integrate_ranges(ene_range, data, A0, A1)
    return pwr_data, live_time


def calc_rate_errors(pwr, lt):
    error = {}
    for key in pwr:
        error[key] = np.divide(np.sqrt(pwr[key]), lt[key])
        pwr[key] = np.divide(pwr[key], lt[key])
    return pwr, error

#def calc_chisqr(pwr_percent,vals,pwr_unc):

def plot_power_data(pwr, lt, write=True):
    bin_width = ENERGY_RANGES[1] - ENERGY_RANGES[0]
    midpoints = get_bin_midpoints(ENERGY_RANGES[0], ENERGY_RANGES[-1], len(ENERGY_RANGES) - 1)
    labels = ["{}%".format(pwr_names[key]) for key in pwr.keys()]
    rates, errors = calc_rate_errors(pwr, lt)
    nonzerokeys = [key for key in rates.keys() if key != 0]
    ys = [[(rates[key][i] - rates[0][i]) / bin_width for i in range(rates[key].shape[0])] for key in nonzerokeys]
    errs = [[np.sqrt(errors[key][i]**2 + errors[0][i]**2) / bin_width for i in range(errors[key].shape[0])] for key in nonzerokeys]
    fig = MultiScatterPlot(midpoints, ys, errs, labels[1:], "energy midpoint [keV]", "rate [hz/keV]", xmin=0,
                           xmax=11400, ylog=True, figsize=(6,5), legend_font_size=12)
    plt.savefig("power_rates.png")
    rate_ratios = [[(ys[i][j] / ys[-1][j])  for j in range(len(ys[-1]))] for i in range(len(ys)-1)]
    ratio_errs = [[rate_ratios[i][j]*np.sqrt((errs[i][j]/ys[i][j])**2 + (errs[-1][j]/ys[-1][j])**2) for j in range(len(ys[-1]))] for i in range(len(ys)-1)]
    fig = MultiScatterPlot(midpoints, rate_ratios, ratio_errs, labels[1:], "energy midpoint [keV]", "rate / 100% pwr rate", xmin=0,
                           xmax=11400, ylog=False, figsize=(6,6), legend_font_size=12,  legend_ncol=int((len(rate_ratios)/2)))
    plt.savefig("power_ratios.png")
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


def main():
    datadir = get_data_dir()
    db = HFIRBG_DB()
    #plot_power_spectra(files)
    pwr, lt = retrieve_power_data(ENERGY_RANGES, datadir, db)
    plot_power_data(pwr, lt)


if __name__ == "__main__":
    main()
