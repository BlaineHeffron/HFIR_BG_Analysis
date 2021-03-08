import csv
import os
import re
from os.path import join
import matplotlib.pyplot as plt

import numpy as np

from PlotUtils import MultiLinePlot, MultiScatterPlot

FILEEXPR = re.compile('[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].txt')


class SpectrumData:
    def __init__(self, data, start, live, A0, A1):
        self.data = data
        self.start = start
        self.live = live
        self.A0 = A0
        self.A1 = A1

    def add(self, s):
        if s.A0 != self.A0 or s.A1 != self.A1:
            raise ValueError("error: spectrum  addition not possible without rebinning")
        self.data += s.data
        self.live += s.live

def subtract_spectra(s1, s2):
    if s1.A0 != s2.A0 or s1.A1 != s2.A1:
        raise ValueError("error: spectrum  addition not possible without rebinning")
    return SpectrumData(((s1.data/s1.live) - (s2.data/s2.live))*s1.live, s1.start, s1.live, s1.A0, s1.A1)


def retrieve_spectra_and_files(n, datadir):
    fs = retrieve_files(datadir)
    return retrieve_spectra(n, fs), fs


def retrieve_spectra(n, flist):
    for f in flist:
        file = os.path.basename(f)
        if n == file_number(file):
            return retrieve_data(f)

def retrieve_files(mydir, recursive=False):
    myfiles = []
    for root, dirs, files in os.walk(mydir):
        for f in files:
            if FILEEXPR.match(f):
                myfiles.append(join(root, f))
        if recursive:
            for d in dirs:
                myfiles += retrieve_files(join(root, d), True)
    return myfiles


def file_number(fname):
    return int(fname[0:-4])


def retrieve_data(myf):
    data = np.zeros((16384,), dtype=np.int32)
    A0 = 0.
    A1 = 0.
    start = ''
    live = 0.
    counter = 0
    with open(myf, 'r') as f:
        for line in f.readlines():
            if 'Start time: ' in line:
                start = line[17:]
            if 'Live time ' in line:
                live = float(line[17:])
            if '#     A0: ' in line:
                A0 = float(line[10:])
            if '#     A1: ' in line:
                A1 = float(line[10:])
            if not line.startswith("#"):
                data[counter] = int(line.split('\t')[2])
                counter += 1
    return SpectrumData(data, start, live, A0, A1)


def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def get_bins(start, stop, n):
    width = (stop - start) / n
    return np.arange(start, stop + width / 2, width)


def get_bin_midpoints(amin, amax, n):
    return np.arange(amin, amax, (amax - amin) / n) + (amax - amin) / (2 * n)


def write_x_y_csv(name, xlabel, ylabel, err_label, xs, ys, errors):
    with open(name, 'w') as csvfile:
        writer = csv.writer(csvfile,
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([xlabel, ylabel, err_label])
        for x, y, e in zip(xs, ys, errors):
            writer.writerow([x, y, e])


def pad_data(a, new_shape):
    result = np.zeros(new_shape)
    if len(new_shape) == 2:
        result[:a.shape[0], :a.shape[1]] = a
    else:
        result[:a.shape[0]] = a
    return result


def rebin_data(a, n):
    mod = a.shape[0] % n
    if mod != 0:
        a = pad_data(a, (a.shape[0] + n - mod,))
    shape = [int(a.shape[0] / n)]
    sh = shape[0], a.shape[0] // shape[0]
    a = a.reshape(sh).sum(axis=-1)
    return a

def populate_data(data_dict, data_dir):
    """
    given data_dict, a dictionary whose values are numbers or list of numbers, populate
    a new dictionary with same keys and either SpectrumData or lists of SpectrumData for the
    spectrum filepaths equating the numbers
    """
    fs = retrieve_files(data_dir)
    out = {}
    for key in data_dict:
        if isinstance(data_dict[key], list):
            out[key] = []
            for n in data_dict[key]:
                out[key].append(retrieve_spectra(n, fs))
        else:
            out[key] = retrieve_spectra(data_dict[key], fs)
    return out

def combine_runs(data_dict):
    """
    given a data dictionary with values of lists of spectrumdata, adds the lists of spectrum data
    together and replaces with spectrumdata of combined data
    """
    for key in data_dict:
        if isinstance(data_dict[key], list):
            s = data_dict[key][0]
            if len(data_dict[key]) > 1:
                for i in range(1, len(data_dict[key])):
                    s.add(data_dict[key][i])
            data_dict[key] = s


def background_subtract(data_dict, subkey):
    """
    given a data dictionary of spectrumdata, returns a new datadict
    with spectrum minus the spectrum in key position
    """
    d = {}
    subtract = data_dict[subkey]
    for key in data_dict:
        d[key] = subtract_spectra(data_dict[key], subtract)
    return d


def plot_subtract_spectra(fdict, compare_name, fname, rebin=1, emin=20):
    ys = []
    absys = []
    percys = []
    names = []
    start_index = 0
    x = []
    errs = []
    if len(fdict.keys()) < 2:
        raise ValueError("fdict must contain at least 2 keys")
    comparespec = fdict[compare_name]
    comparespec_rebinned = comparespec.data if rebin == 1 else rebin_data(comparespec.data, rebin)
    compare_spec_norm = comparespec.data / comparespec.live / comparespec.A1 if rebin == 1 else \
        rebin_data(comparespec.data, rebin) / comparespec.live / (rebin*comparespec.A1)
    for name in fdict.keys():
        if name == compare_name:
            continue
        spec = fdict[name]
        if rebin > 1:
            data = rebin_data(spec.data, rebin)
        else:
            data = spec.data
        A1 = spec.A1 * rebin
        data_norm = data / spec.live / A1
        subtracted = data_norm - compare_spec_norm
        subtracted_perc = 100.*(data_norm - compare_spec_norm) / compare_spec_norm
        if start_index == 0:
            start_index = find_start_index(data, spec.A0, A1, emin)
        y = [abs(d) for d in subtracted[start_index:]]
        absys.append(y)
        y = [d for d in subtracted[start_index:]]
        ys.append(y)
        y = [d for d in subtracted_perc[start_index:]]
        percys.append(y)
        sub_errs = np.sqrt(data / (spec.live**2) + comparespec_rebinned / (comparespec.live**2)) / A1
        err = [e for e in sub_errs[start_index:]]
        x = [i * A1 + spec.A0 - A1 / 2 for i in range(start_index,data.shape[0])]
        errs.append(err)
        names.append("{0} minus {1}".format(name, compare_name))
    MultiScatterPlot(x, ys, errs, names, "Energy [keV]", "Rate Difference [hz/keV]", ylog=False)
    if rebin > 1:
        fname = fname + "_rebin{}".format(rebin)
    plt.savefig("{}.png".format(fname))
    MultiScatterPlot(x, absys, errs, names, "Energy [keV]", "Absolute Rate Difference [hz/keV]", ylog=True)
    plt.savefig("{}_absdiff.png".format(fname))
    MultiScatterPlot(x, percys, errs, names, "Energy [keV]", "Rate Difference Percentage", ylog=True)
    plt.savefig("{}_percdiff.png".format(fname))


def find_start_index(data, A0, A1, emin):
    start_index = 0
    for i in range(data.shape[0]):
        if i * A1 + A0 - A1 / 2. > emin:
            start_index = i
            break
    return start_index

def plot_multi_spectra(fdict, n, rebin=1, emin=20):
    ys = []
    names = []
    x = []
    start_index = 0
    for name in fdict.keys():
        if isinstance(fdict[name], SpectrumData):
            spec = fdict[name]
            start, live, A0, A1, data = spec.start, spec.live, spec.A0, spec.A1, spec.data
        else:
            spec = retrieve_data(fdict[name][0])
            start, live, A0, A1, data = spec.start, spec.live, spec.A0, spec.A1, spec.data
            for i, f in enumerate(fdict[name]):
                if i == 0:
                    continue
                spec2 = retrieve_data(f)
                live += spec2.live
                data += spec2.data
        if rebin > 1:
            data = rebin_data(data, rebin)
            A1 = A1 * rebin
        y = [d / live / A1 for d in data]
        errs = np.sqrt(data)
        if start_index == 0:
            start_index = find_start_index(data, A0, A1, emin)
        x = [i * A1 + A0 - A1 / 2 for i in range(start_index, data.shape[0])]
        ys.append(y[start_index:])
        names.append(name)
        # err = [d / live / A1 for d in errs]
        # MultiScatterPlot(x, [y], [err], [name], "Energy [keV]", "Rate [hz/keV]")
        # plt.savefig("{}_errors.png".format(name))
    MultiLinePlot(x, ys, names, "Energy [keV]", "Rate [hz/keV]")
    plt.savefig("{}.png".format(n))


def plot_spectra(fs, name, rebin=1):
    spec = retrieve_data(fs[0])
    start, live, A0, A1, data = spec.start, spec.live, spec.A0, spec.A1, spec.data
    for i, f in enumerate(fs):
        if i == 0:
            continue
        spec2 = retrieve_data(f)
        live += spec2.live
        data += spec2.data
    if rebin > 1:
        data = rebin_data(data, rebin)
        A1 = A1 * rebin
    y = [d / live / A1 for d in data]
    errs = np.sqrt(data)
    x = [i * A1 + A0 - A1 / 2 for i in range(data.shape[0])]
    err = [d / live / A1 for d in errs]
    MultiScatterPlot(x, [y], [err], [name], "Energy [keV]", "Rate [hz/keV]")
    plt.savefig("{}_errors.png".format(name))
    MultiLinePlot(x, [y], [name], "Energy [keV]", "Rate [hz/keV]")
    plt.savefig("{}.png".format(name))
