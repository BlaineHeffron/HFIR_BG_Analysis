import csv
import os
import re
from os.path import join
import matplotlib.pyplot as plt
import numba as nb
import ntpath
import platform
from csv import reader

import numpy as np

from src.utilities.PlotUtils import MultiLinePlot, MultiScatterPlot
from copy import copy

FILEEXPR = re.compile('[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].txt')


class SpectrumData:
    def __init__(self, data, start, live, A0, A1):
        """hist convention is first bin edge inclusive, last bin edge exclusive,
        index 0 is underflow bin, index nbins+1 is overflow"""
        self.data = data
        self.start = start
        self.live = live
        self.A0 = A0
        self.A1 = A1
        self.hist = None
        self.bin_edges = None
        self.bin_midpoints = None
        self.nbins = None
        self.rebin()

    def add(self, s):
        if s.A0 != self.A0 or s.A1 != self.A1:
            raise ValueError("error: spectrum  addition not possible without rebinning")
        self.data += s.data
        self.live += s.live

    def calculate_bin_midpoints(self):
        if self.bin_edges is None:
            raise RuntimeError("bin edges not set")
        else:
            self.bin_midpoints = np.zeros((self.bin_edges.shape[0]-1,))
            for j in range(0,self.bin_edges.shape[0]-1):
                self.bin_midpoints[j] = self.bin_edges[j] + (self.bin_edges[j+1] - self.bin_edges[j]) / 2.

    def rebin_factor(self, factor):
        if factor <= 1:
            raise RuntimeError("rebin factor must be > 1")
        new_hist = rebin_data(self.hist[1:-1], factor)
        under = self.hist[0]
        over = self.hist[-1]
        self.hist = np.zeros((new_hist.shape[0]+2,))
        self.hist[0] = under
        self.hist[1] = over
        self.hist[1:-1] = new_hist
        old_edges = copy(self.bin_edges)
        self.bin_edges = np.zeros((self.hist.shape[0]-1,))
        j = 0
        for i in range(old_edges.shape[0]):
            if i % factor == 0:
                self.bin_edges[j] = old_edges[i]
                j += 1
        self.bin_edges[-1] = old_edges[-1]
        self.nbins = self.bin_edges.shape[0] - 1
        self.calculate_bin_midpoints()

    def rebin(self, bin_edges=None):
        set_to_data = False
        if bin_edges is None:
            set_to_data = True
            bin_edges = np.zeros(self.data.shape[0]+1)
            for i in range(self.data.shape[0]+1):
                bin_edges[i] = self.A0 + self.A1*i + self.A1/2.
        self.hist = np.zeros((bin_edges.shape[0]+1,))
        self.bin_edges = bin_edges
        self.calculate_bin_midpoints()
        self.nbins = bin_edges.shape[0]-1
        if set_to_data:
            self.hist[1:-1] = self.data
            return
        numba_rebin(self.data, self.hist, self.A0, self.A1, self.bin_edges)

    def hist_subtract(self, s):
        if self.bin_edges is None:
            self.rebin()
        if s.bin_edges is None:
            s.rebin()
        if not np.array_equal(s.bin_edges, self.bin_edges):
            s.rebin(self.bin_edges)
        #normalize before subtracting
        self.hist /= self.live
        self.hist -= (s.hist/s.live)
        self.hist[self.hist < 0] = 0
        self.hist *= self.live

    def get_normalized_hist(self):
        """
        :return: normalized histogram (no underflow and overflow)
        """
        norm = np.zeros((self.hist.shape[0]-2,))
        for i in range(1,self.nbins+1):
            norm[i-1] = self.hist[i] / (self.bin_edges[i] - self.bin_edges[i-1])
        norm /= self.live
        return norm

    def find_start_index(self, emin):
        """
        :param emin: minimum energy in same units as bin edges
        :return: index of hist that is the first value with x greater than emin
        """
        if emin < self.bin_edges[0]:
            return 0
        for i, bedge in enumerate(self.bin_edges):
            if bedge > emin:
                return i
        return len(self.bin_edges)

def find_end_index(self, emax):
    """
    :param emin: minimum energy in same units as bin edges
    :return: index of hist that is the last value with x less than emin
    """
    for i, bedge in enumerate(self.bin_edges):
        if bedge > emax:
            return i
    return len(self.bin_edges)


def subtract_spectra(s1, s2):
    """
    :param s1: spectrum you want to subtract from
    :param s2: spectrum you are subtracting
    :return:  a copy of spectrum s1 with hist values subtracted by s2
    """
    subtracted = copy(s1)
    subtracted.hist_subtract(s2)
    return subtracted


def retrieve_spectra_and_files(n, datadir):
    fs = retrieve_files(datadir)
    return retrieve_spectra(n, fs), fs


def spectrum_name_check(name, flist):
    if isinstance(name, int):
        for f in flist:
            file = os.path.basename(f)
            if name == file_number(file):
                return retrieve_data(f)
    else:
        if name.endswith(".txt"):
            checkname = name
        else:
            checkname = name + ".txt"
        for f in flist:
            path, fname = ntpath.split(f)
            if fname == checkname:
                return retrieve_data(f)


def retrieve_spectra(n, flist):
    return spectrum_name_check(n, flist)

def retrieve_file_extension(mydir, ext):
    myfiles = []
    for root, dirs, files in os.walk(mydir):
        for f in files:
            if f.endswith(ext):
                myfiles.append(join(root, f))
    return myfiles


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
    try:
        return int(fname[0:-4])
    except ValueError:
        return -1


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


def get_data_dir():
    if "HFIRBGDATA" not in os.environ:
        raise RuntimeError("Error: set environment variable HFIRBGDATA to the directory with your files")
    return os.path.expanduser(os.environ["HFIRBGDATA"])


def populate_data(data_dict, data_dir):
    """
    given data_dict, a dictionary whose values are numbers or list of numbers, populate
    a new dictionary with same keys and either SpectrumData or lists of SpectrumData for the
    spectrum filepaths equating the numbers
    """
    fs = retrieve_file_extension(data_dir, ".txt")
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

def rebin_spectra(data_dict, bin_edges):
    for key in data_dict:
        data_dict[key].rebin(bin_edges)

def background_subtract(data_dict, subkey, bin_edges):
    """
    given a data dictionary of spectrumdata, returns a new datadict
    with spectrum minus the spectrum in key position
    """
    d = {}
    subtract = data_dict[subkey]
    subtract.rebin(bin_edges)
    for key in data_dict:
        if key == subkey:
            continue
        data_dict[key].rebin(bin_edges)
        d[key] = subtract_spectra(data_dict[key], subtract)
    return d


def plot_subtract_spectra(fdict, compare_name, fname, rebin=1, emin=20, emax=None):
    ys = []
    absys = []
    percys = []
    names = []
    start_index = 0
    end_index = 0
    x = []
    errs = []
    if len(fdict.keys()) < 2:
        raise ValueError("fdict must contain at least 2 keys")
    comparespec = fdict[compare_name]
    if rebin > 1:
        comparespec = copy(comparespec)
        comparespec.rebin_factor(rebin)
    compare_spec_norm = comparespec.get_normalized_hist()
    for name in fdict.keys():
        if name == compare_name:
            continue
        spec = fdict[name]
        if rebin > 1:
            spec = copy(spec)
            spec.rebin_factor(rebin)
        data_norm = spec.get_normalized_hist()
        subtracted = data_norm - compare_spec_norm
        subtracted_perc = 100. * safe_divide(data_norm - compare_spec_norm, compare_spec_norm)
        if start_index == 0 and emin is not None:
            start_index = spec.find_start_index(emin)
            if start_index < 0:
                start_index = 0
        if emax is not None and end_index == 0:
            end_index = spec.find_start_index(emax) - 1
        elif end_index == 0:
            end_index = spec.find_start_index(1.0e12) - 1
        y = [abs(d) for d in subtracted[start_index:end_index]]
        absys.append(y)
        y = [d for d in subtracted[start_index:end_index]]
        ys.append(y)
        y = [d for d in subtracted_perc[start_index:end_index]]
        percys.append(y)
        sub_errs = np.sqrt(spec.hist / (spec.live ** 2) + comparespec.hist / (comparespec.live ** 2)) / (spec.bin_edges[1] - spec.bin_edges[0])
        err = [e for e in sub_errs[start_index+1:end_index+1]]
        x = spec.bin_midpoints[start_index:end_index]
        errs.append(err)
        names.append("{0} minus {1}".format(name, compare_name))
    MultiScatterPlot(x, ys, errs, names, "Energy [keV]", "Rate Difference [hz/keV]", ylog=False)
    if rebin > 1:
        fname = fname + "_rebin{}".format(rebin)
    plt.savefig("{}.png".format(fname))
    MultiScatterPlot(x, absys, errs, names, "Energy [keV]", "Absolute Rate Difference [hz/keV]", ylog=True)
    plt.savefig("{}_absdiff.png".format(fname))
    MultiScatterPlot(x, percys, errs, names, "Energy [keV]", "Rate Difference Percentage", ylog=False)
    plt.savefig("{}_percdiff.png".format(fname))


def find_start_index(data, A0, A1, emin):
    start_index = 0
    for i in range(data.shape[0]):
        if i * A1 + A0 + A1 / 2. > emin:
            start_index = i
            break
    return start_index


def set_indices(start_index, end_index, emin, emax, spec):
    if start_index == 0 and emin is not None:
        start_index = spec.find_start_index(emin)
    if emax is not None and end_index == 0:
        end_index = spec.find_start_index(emax) - 1
    elif end_index == 0:
        end_index = spec.find_start_index(1.e12) - 1
    return start_index, end_index


def plot_multi_spectra(fdict, n, rebin=1, emin=20, emax=None):
    ys = []
    names = []
    x = []
    end_index = 0
    start_index = 0
    for name in fdict.keys():
        if isinstance(fdict[name], SpectrumData):
            spec = fdict[name]
        else:
            spec = retrieve_data(fdict[name][0])
            for i, f in enumerate(fdict[name]):
                if i == 0:
                    continue
                spec2 = retrieve_data(f)
                spec.add(spec2)
        if rebin > 1:
            spec = copy(spec)
            spec.rebin_factor(rebin)
        y = spec.get_normalized_hist()
        start_index, end_index = set_indices(start_index, end_index, emin, emax, spec)
        x = spec.bin_midpoints[start_index:end_index]
        ys.append(y[start_index:end_index])
        names.append(name)
        # err = [d / live / A1 for d in errs]
        # MultiScatterPlot(x, [y], [err], [name], "Energy [keV]", "Rate [hz/keV]")
        # plt.savefig("{}_errors.png".format(name))
    fig=MultiLinePlot(x, ys, names, "Energy [keV]", "Rate [hz/keV]")
    plt.savefig("{}.png".format(n))
    plt.close(fig)


def retrieve_spec_data(f):
    if isinstance(f, SpectrumData):
        spec = f
    else:
        spec = retrieve_data(f)
    return spec


def plot_spectra(fs, name, rebin=1, emin=None, emax=None):
    spec = retrieve_spec_data(fs[0])
    for i, f in enumerate(fs):
        if i == 0:
            continue
        spec.add(retrieve_spec_data(f))
    if rebin > 1:
        spec = copy(spec)
        spec.rebin_factor(rebin)
    start_index, end_index = set_indices(0, 0, emin, emax, spec)
    y = spec.get_normalized_hist()[start_index:end_index]
    errs = np.sqrt(spec.hist)
    x = spec.bin_midpoints[start_index:end_index]
    err = [errs[i] / spec.live / (spec.bin_edges[i+1] - spec.bin_edges[i]) for i in range(len(spec.bin_edges)-1)]
    err = err[start_index:end_index]
    MultiScatterPlot(x, [y], [err], [name], "Energy [keV]", "Rate [hz/keV]")
    plt.savefig("{}_errors.png".format(name))
    MultiLinePlot(x, [y], [name], "Energy [keV]", "Rate [hz/keV]")
    plt.savefig("{}.png".format(name))


@nb.jit(nopython=True)
def numba_rebin(data, hist, A0, A1, bin_edges):
    nbins = bin_edges.shape[0] - 1
    for i in range(data.shape[0]):
        low = A0 + A1 * i + A1 / 2.
        up = A0 + A1 * (i + 1) + A1 / 2.
        if low < bin_edges[0]:
            hist[0] += data[i]
        elif up > bin_edges[-1]:
            hist[-1] += data[i]
        else:
            for j, bedge in enumerate(bin_edges):
                if bedge >= low:
                    if bedge > up:
                        hist[j] += data[i]
                    else:
                        hist[j] += data[i] * (bedge - low) / A1
                        if j == nbins:
                            hist[j + 1] += data[i] * (up - bedge) / A1
                        elif bin_edges[j + 1] > up:
                            hist[j + 1] += data[i] * (up - bedge) / A1
                        else:
                            # new bins larger than old
                            k = 1
                            while j + k <= nbins and bin_edges[j + k] < up:
                                hist[j + k] += data[i] * (bin_edges[j + k] - bin_edges[j + k - 1]) / A1
                                k += 1
                            if j + k == nbins + 1 and up > bin_edges[-1]:
                                # rest of it goes in overflow
                                hist[j + k] += data[i] * (up - bin_edges[-1]) / A1
                            else:
                                # get rest of the value in the last bin
                                hist[j + k] += data[i] * (bin_edges[j + k] - up) / A1
                    break

def creation_date(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime

def read_csv_list_of_tuples(fpath):
    with open(fpath, 'r') as read_obj:
        csv_reader = reader(read_obj)
        list_of_tuples = list(map(tuple, csv_reader))
    return list_of_tuples