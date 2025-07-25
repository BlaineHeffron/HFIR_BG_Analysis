import csv
import os
import pickle
import json
import re
from datetime import datetime
from os.path import join
import matplotlib.pyplot as plt
import ntpath
import platform
from csv import reader
import shutil
import ctypes

import numpy as np
from scipy import stats

from src.analysis.Spectrum import SpectrumData, SpectrumFitter, SubtractSpectrum
from src.utilities.PlotUtils import MultiLinePlot, MultiScatterPlot, ScatterLinePlot, ScatterDifferencePlot, \
    MultiXScatterPlot
from src.utilities.FitUtils import linfit, sqrtfit
from copy import copy
from ROOT import TFile, TVectorF

FILEEXPR = re.compile('[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].txt')

c_float_p = ctypes.POINTER(ctypes.c_float)


def find_end_index(self, emax):
    """
    :param emax: minimum energy in same units as bin edges
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


def spectrum_name_check(name, flist, db):
    if isinstance(name, int):
        for f in flist:
            file = os.path.basename(f)
            if name == file_number(file):
                return retrieve_data(f, db)
    else:
        if name.endswith(".txt"):
            checkname = name
        else:
            checkname = name + ".txt"
        for f in flist:
            path, fname = ntpath.split(f)
            if fname == checkname:
                return retrieve_data(f, db)


def retrieve_spectra(n, flist, db):
    return spectrum_name_check(n, flist, db)


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


def retrieve_data(myf, db=None):
    """
    :param myf: path to .txt file containing cnf converted spectrum
    :param db: HFIRBG_DB object for retrieving calibration
    :return:  SpectrumData object containing the spectrum data and metadata,
    raises IOError if the .txt file is not a valid cnf file
    """
    data = np.zeros((16384,), dtype=np.int32)
    A0 = 0.
    A1 = 0.
    start = ''
    live = 0.
    counter = 0
    fname = os.path.basename(myf)
    with open(myf, 'r') as f:
        try:
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
        except Exception as e:
            f.close()
            raise IOError("{0} is not a valid .cnf formatted file. Error: {1}".format(myf, e))
    if db is not None:
        row = db.retrieve_calibration(fname)
        if row:
            # print(
            #    "found calibration for file {0} in database, using values A0 = {1}, A1 = {2} instead of A0 = {3}, A1 = {4}".format(
            #        fname, row[0], row[1], A0, A1))
            A0 = row[0]
            A1 = row[1]
        row = db.retrieve_file_time(fname)
        if row:
            start = row[0]
            live = row[1]
            start = dt_to_string(start)

    return SpectrumData(data, start, live, A0, A1, fname)


def dt_to_string(t):
    try:
        dt = datetime.fromtimestamp(t)
        return dt.strftime("%Y-%m-%d, %H:%M:%S")
    except (ValueError, OverflowError, OSError):
        return "Invalid date"


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


def write_rows_csv(f, header, rows, delimiter="|"):
    writer = csv.writer(f, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)

def read_rows_csv(f, delimiter=','):
    data = []
    with open(f, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        for row in reader:
            data.append(row)
    return data

def get_data_dir():
    if "HFIRBGDATA" not in os.environ:
        raise RuntimeError("Error: set environment variable HFIRBGDATA to the directory with your files")
    return os.path.expanduser(os.environ["HFIRBGDATA"])

def populate_rd_data(data_dict, db):
    out = {}
    for key1 in data_dict.keys():
        if key1 not in out.keys():
            out[key1] = {}
        for key2 in data_dict[key1].keys():
            if key2 not in out[key1].keys():
                out[key1][key2] = []
            for f in data_dict[key1][key2]:
                out[key1][key2].append(retrieve_data(f, db))
    return out

def populate_data_db(data_dict, db):
    """
    given data_dict, a dictionary whose values are numbers or list of numbers, populate
    a new dictionary with same keys and either SpectrumData or lists of SpectrumData for the
    spectrum filepaths equating the numbers
    """
    out = {}
    for key in data_dict:
        if isinstance(data_dict[key], list):
            out[key] = []
            for n in data_dict[key]:
                out[key].append(retrieve_data(n, db))
        else:
            out[key] = retrieve_data(data_dict[key], db)
    return out

def populate_data(data_dict, data_dir, db):
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
                out[key].append(retrieve_spectra(n, fs, db))
        else:
            out[key] = retrieve_spectra(data_dict[key], fs, db)
    return out


def populate_data_config(config, db, comb_runs=False):
    """
    given config, a dictionary that specifies the files to retrive using database, populates
    a new dictionary with  keys as filenames and values as spectrumdata objects
    """
    fs = db.get_files_from_config(config)
    if comb_runs:
        run_dict = {}
        for f in fs:
            name = os.path.basename(f)
            rname = db.retrieve_run_name_from_file_name(name)
            if rname:
                if rname in run_dict.keys():
                    run_dict[rname].add(retrieve_data(f, db))
                else:
                    run_dict[rname] = set([retrieve_data(f, db)])
        combine_runs(run_dict)
        return run_dict
    else:
        out = {}
        for f in fs:
            name = os.path.basename(f)
            if name.endswith(".txt"):
                name = name[0:-4]
            out[name] = retrieve_data(f, db)
        return out


def populate_data_root(data_dict, spec_path, live_path, isParam=False, xScale=1, rebin=1):
    """
    given data_dict, a dictionary whose values are numbers or list of numbers, populate
    a new dictionary with same keys and either SpectrumData or lists of SpectrumData for the
    spectrum filepaths equating the numbers
    """
    out = {}
    for key in data_dict:
        if isinstance(data_dict[key], list):
            out[key] = []
            for n in data_dict[key]:
                out[key].append(get_spec_from_root(n, spec_path, live_path, isParam, xScale, rebin))
        else:
            out[key] = get_spec_from_root(data_dict[key], spec_path, live_path, isParam, xScale, rebin)
    return out


def combine_runs(data_dict, max_interval=None, ignore_failed_add=False):
    """
    given a data dictionary with values of lists of spectrumdata, adds the lists of spectrum data
    together and replaces with spectrumdata of combined data
    if max_interval is given, will combine only if the acquisition start time is less than max_interval in seconds from
    first run in sequence. value of dict will be list of spectrumdata objects instead of a single spectrumdata object
    """
    for key in data_dict:
        if isinstance(data_dict[key], set):
            mylist = list(data_dict[key])
        elif isinstance(data_dict[key], list):
            mylist = data_dict[key]
        else:
            continue
        if max_interval is not None:
            mylist.sort(key=lambda x: timestring_to_dt(x.start))
            s = mylist[0]
            newlist = []
            if len(mylist) > 1:
                max_time = timestring_to_dt(s.start) + max_interval
                for i in range(1, len(mylist)):
                    if timestring_to_dt(mylist[i].start) > max_time:
                        newlist.append(s)
                        s = mylist[i]
                        max_time = timestring_to_dt(s.start) + max_interval
                    else:
                        if ignore_failed_add:
                            try:
                                s.add(mylist[i])
                            except Exception as e:
                                print(e)
                                print("skipping...")
                        else:
                            s.add(mylist[i])
            newlist.append(s)
            data_dict[key] = newlist
        else:
            s = mylist[0]
            if len(mylist) > 1:
                for i in range(1, len(mylist)):
                    if ignore_failed_add:
                        try:
                            s.add(mylist[i])
                        except Exception as e:
                            print(e)
                            print("skipping...")
                    else:
                        s.add(mylist[i])
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


def plot_ratio_spectra(fdict, compare_name, fname, rebin=1, emin=20, emax=None):
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
        subtracted = safe_divide(data_norm, compare_spec_norm)
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
        sub_errs = np.sqrt(spec.hist / (spec.live ** 2) + comparespec.hist / (comparespec.live ** 2)) / (
                spec.bin_edges[1] - spec.bin_edges[0])
        err = [e for e in sub_errs[start_index + 1:end_index + 1]]
        x = spec.bin_midpoints[start_index:end_index]
        errs.append(err)
        names.append("{0} / {1}".format(name, compare_name))
    MultiScatterPlot(x, ys, errs, names, "Energy [keV]", "Rate Ratio ", ylog=False)
    if rebin > 1:
        fname = fname + "_rebin{}".format(rebin)
    plt.savefig("{}.png".format(fname))


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
        sub_errs = np.sqrt(spec.hist / (spec.live ** 2) + comparespec.hist / (comparespec.live ** 2)) / (
                spec.bin_edges[1] - spec.bin_edges[0])
        err = [e for e in sub_errs[start_index + 1:end_index + 1]]
        x = spec.bin_midpoints[start_index:end_index]
        errs.append(err)
        names.append("{0} minus {1}".format(name, compare_name))
    MultiScatterPlot(x, ys, errs, names, "Energy [keV]", "Rate Difference [hz/keV]", ylog=False)
    if rebin > 1:
        fname = fname + "_rebin{}".format(rebin)
    plt.savefig("{}.png".format(fname), bbox_inches="tight")
    MultiScatterPlot(x, absys, errs, names, "Energy [keV]", "Absolute Rate Difference [hz/keV]", ylog=True)
    plt.savefig("{}_absdiff.png".format(fname), bbox_inches="tight")
    MultiScatterPlot(x, percys, errs, names, "Energy [keV]", "Rate Difference Percentage", ylog=False)
    plt.savefig("{}_percdiff.png".format(fname), bbox_inches="tight")


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


def load_pyspec(fn):
    try:
        f = open(fn, 'rb')
        return pickle.load(f)
    except Exception as e:
        print(e)
        return None

def write_pyspec(fn, spec):
    f = open(fn, 'wb')
    pickle.dump(spec, f)


def write_spectra(fdict, outdir, db):
    for name in fdict.keys():
        if isinstance(fdict[name], SpectrumData):
            spec = fdict[name]
        else:
            spec = retrieve_data(fdict[name][0], db)
            for i, f in enumerate(fdict[name]):
                if i == 0:
                    continue
                spec2 = retrieve_data(f, db)
                spec.add(spec2)
        x = spec.get_data_energies()
        y = spec.data
        rate = spec.data / spec.live
        rows = [(xi, yi, ri) for xi, yi, ri in zip(x, y, rate)]
        f = open(join(outdir, name + ".csv"), 'w')
        write_rows_csv(f, ["energy [keV]", "counts", "rate [Hz]"], rows, delimiter=",")
        f.flush()
        f.close()


def plot_multi_spectra(fdict, n, rebin=1, emin=20, emax=None, loc="upper right", rebin_edges=None, ebars=True, figsize=(14,8), ylog=True):
    ys = []
    names = []
    x = []
    end_index = 0
    start_index = 0
    ymin = 9e10
    yerrs = []
    for name in fdict.keys():
        if isinstance(fdict[name], SpectrumData):
            spec = fdict[name]
        else:
            raise Exception("not supported")
        if rebin > 1:
            spec = copy(spec)
            spec.rebin_factor(rebin)
        elif rebin_edges is not None:
            spec = copy(spec)
            spec.rebin(rebin_edges)
        y = spec.get_normalized_hist()
        start_index, end_index = set_indices(start_index, end_index, emin, emax, spec)
        x = spec.bin_midpoints[start_index:end_index]
        ys.append(y[start_index:end_index])
        yerrs.append(spec.get_normalized_err()[start_index:end_index])
        minval = 9e10
        for i in range(len(ys)):
            mval = np.min(ys[i][ys[i] > 0])
            if mval < minval:
                minval = mval
        if ymin > minval:
            ymin = minval
        names.append(name)
        # err = [d / live / A1 for d in errs]
        # MultiScatterPlot(x, [y], [err], [name], "Energy [keV]", "Rate [hz/keV]")
        # plt.savefig("{}_errors.png".format(name))
    if ebars:
        fig = MultiScatterPlot(x, ys, yerrs, names, "Energy [keV]", "Rate [hz/keV]", ylog=ylog, ymin=ymin, figsize=figsize)
    else:
        fig = MultiLinePlot(x, ys, names, "Energy [keV]", "Rate [hz/keV]", ylog=ylog, ymin=ymin, figsize=figsize)
    plt.savefig("{}.png".format(n), bbox_inches="tight")
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
    if start_index >= end_index:
        return
    y = spec.get_normalized_hist()[start_index:end_index]
    errs = np.sqrt(spec.hist)
    x = spec.bin_midpoints[start_index:end_index]
    err = [errs[i] / spec.live / (spec.bin_edges[i + 1] - spec.bin_edges[i]) for i in range(len(spec.bin_edges) - 1)]
    err = err[start_index:end_index]
    MultiScatterPlot(x, [y], [err], [name], "Energy [keV]", "Rate [hz/keV]")
    plt.savefig("{}_errors.png".format(name), bbox_inches="tight")
    plt.close()
    MultiLinePlot(x, [y], [name], "Energy [keV]", "Rate [hz/keV]")
    plt.savefig("{}.png".format(name), bbox_inches="tight")
    plt.close()


def plot_time_series(data, outdir, emin=30, emax=None, legend_map=None, ymin=None, legend_fraction=None, figsize=(5,4)):
    if emin is None and emax is None:
        energystring = "full"
    elif emin is None and emax is not None:
        energystring = "below{0}keV".format(emax)
    elif emax is None and emin is not None:
        energystring = "above{0}keV".format(emin)
    else:
        energystring = "{0}-{1}keV".format(emin, emax)
    xlabel = "date"
    for key in data.keys():
        if not isinstance(data[key], list):
            print("{} is not a list of data, skipping".format(key))
            continue
        plot_name = join(outdir, "{0}_{1}.png".format(key, energystring))
        A0_name = join(outdir, "{0}_{1}_A0.png".format(key, energystring))
        A1_name = join(outdir, "{0}_{1}_A1.png".format(key, energystring))
        ylabel = "rate [Hz/keV]"
        rates = []
        drates = []
        times = []
        A0 = []
        A1 = []
        e1, e2 = data[key][0].get_e_limits(emin, emax)
        if legend_map is not None and key in legend_map.keys():
            legend = legend_map[key]
            line_label = []
        else:
            legend = None
            line_label = ["{0:.0f} - {1:.0f} keV".format(e1, e2)]
        for spec in data[key]:
            r, dr = spec.integrate(emin, emax, norm=True)
            rates.append(r)
            drates.append(dr)
            A0.append(spec.A0)
            A1.append(spec.A1)
            times.append(spec.start_timestamp())
        if legend:
            data_dict = {}
            for i, label in enumerate(legend):
                if label not in data_dict.keys():
                    data_dict[label] = [[],[],[]]
                data_dict[label][0].append(rates[i])
                data_dict[label][1].append(drates[i])
                data_dict[label][2].append(times[i])
            rates = [data_dict[l][0] for l in data_dict.keys()]
            drates = [data_dict[l][1] for l in data_dict.keys()]
            times = [data_dict[l][2] for l in data_dict.keys()]
            
            rates = [x for _, x in sorted(zip(times, rates), key=lambda pair: pair[0][0])]
            drates = [x for _, x in sorted(zip(times, drates), key=lambda pair: pair[0][0])]
            keys = data_dict.keys()
            keys = [x for _, x in sorted(zip(times, keys), key=lambda pair: pair[0][0])]
            times.sort(key=lambda x: x[0])
            MultiXScatterPlot(times, rates, drates, keys, xlabel, ylabel, xdates=True, ymin=ymin, figsize=figsize, legend_outside=True, legend_fraction=legend_fraction, title=energystring)
        else:
            MultiScatterPlot(times, [rates], [drates], line_label, xlabel, ylabel, xdates=True, ymin=ymin, title=energystring)
        plt.savefig(plot_name)
        plt.close()
        #MultiScatterPlot(times, [A0], [[0]*len(A0)], line_label, xlabel, "A0", xdates=True)
        #plt.savefig(A0_name)
        #plt.close()
        #MultiScatterPlot(times, [A1], [[0]*len(A1)], line_label, xlabel, "A1", xdates=True)
        #plt.savefig(A1_name)
        #plt.close()


def get_calibration(file):
    # assumes file has a line like this:
    # A0:
    # returns A0 and A1
    A0 = None
    A1 = None
    with open(file) as f:
        for line in f.readlines():
            if "A0:" in line:
                A0 = float(line.split(":")[1].strip())
            if "A1:" in line:
                A1 = float(line.split(":")[1].strip())
                break
    return A0, A1


def start_date(file):
    # assumes file has a line like this:
    # Start time:    2021-02-20, 00:37:56
    # returns start time and live time
    dt = None
    lt = None
    with open(file) as f:
        for line in f.readlines():
            if 'Start time: ' in line:
                data = line[13:].strip()
                dt = datetime.strptime(data, "%Y-%m-%d, %H:%M:%S")
            if 'Live time ' in line:
                lt = float(line[17:])
                break
    if dt:
        return dt.timestamp(), lt
    else:
        print("couldnt find datetime, using creation date of file")
        return creation_date(file), lt


def timestring_to_dt(time):
    dt = datetime.strptime(time, "%Y-%m-%d, %H:%M:%S")
    return dt.timestamp()


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


def read_csv_list_of_tuples(fpath, delimiter=','):
    with open(fpath, 'r') as read_obj:
        csv_reader = reader(read_obj, delimiter=delimiter)
        list_of_tuples = list(map(tuple, csv_reader))
    return list_of_tuples


def retrieve_position_scans():
    path = os.path.abspath(os.path.join(__file__, "../../.."))
    path = join(path, "db")
    files = []
    for root, directories, file in os.walk(path):
        for file in file:
            if file.endswith(".csv") and file.startswith("position_scan"):
                files.append(join(path, file))
    position_metadata = []
    for f in files:
        data = read_csv_list_of_tuples(f, delimiter='|')
        r_pos = -1
        l_pos = -1
        angle_pos = -1
        fname_pos = -1
        live_pos = -1
        for i, colname in enumerate(data[0]):
            if colname == "R" or colname.startswith("R"):
                r_pos = i
            elif colname == "L" or colname.startswith("L"):
                l_pos = i
            elif colname.startswith("angle") or colname.startswith("theta"):
                angle_pos = i
            elif colname.startswith("file"):
                fname_pos = i
            elif colname.startswith("live "):
                live_pos = i
        if r_pos == -1 or l_pos == -1 or angle_pos == -1 or fname_pos == -1 or live_pos == -1:
            raise IOError("Error: cannot parse file {0}, cant find header info".format(f))
        for row in data[1:]:
            position_metadata.append((row[r_pos], row[l_pos], row[angle_pos], row[live_pos], row[fname_pos]))
    return position_metadata


def get_json(fpath):
    with open(fpath, 'r') as f:
        data = json.load(f)
    return data


def write_json(fpath, data, prettyprint=False):
    with open(fpath, 'w') as f:
        if prettyprint:
            json.dump(data, f, indent=4)
        else:
            json.dump(data, f)
    return True


def write_spe(fpath, spec):
    """
    writes spectrum to .spe file format
    :param fpath: path to output .spe file
    :param spec: numpy 1d array
    :return: null
    """
    if os.path.exists(fpath):
        print("{} already exists, skipping".format(fpath))
        return
    path, fname = ntpath.split(fpath)
    if path:
        os.chdir(path)
    fname_bytes = fname.encode('utf-8')
    dir_path = os.path.dirname(os.path.realpath(__file__))

    spe_write = ctypes.CDLL(os.path.join(dir_path, "write_spe.so"))
    spec = spec.astype(np.float32)
    spec_p = spec.ctypes.data_as(c_float_p)
    spe_write.wspec.argtypes = [ctypes.c_char_p, c_float_p, ctypes.c_int]
    spe_write.wspec(fname_bytes, spec_p, spec.shape[0])
    # from write_spe.so import WSPEC
    # WSPEC(fname, spec, spec.shape[0])
    shutil.move(fname, fpath)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def write_root_with_db(spec, name, db, title=''):
    if title == '':
        title = name
    hist = spec.generate_root_hist(name, title)
    fpath = db.get_file_path_from_name(spec.fname)
    if fpath.endswith(".txt"):
        fpath = fpath[0:-4] + ".root"
    else:
        fpath = fpath + ".root"
    myFile = TFile.Open(fpath, "RECREATE")
    myFile.WriteObject(hist, "GeDataHist")
    lt = TVectorF(1)
    lt[0] = spec.live
    myFile.WriteObject(lt, "LiveTime")
    myFile.Close()


def write_root(spec, name, fpath, title=''):
    if title == '':
        title = name
    hist = spec.generate_root_hist(name, title)
    myFile = TFile.Open(fpath, "RECREATE")
    myFile.WriteObject(hist, "GeDataHist")
    lt = TVectorF(1)
    lt[0] = spec.live
    myFile.WriteObject(lt, "LiveTime")
    myFile.Close()


def scale_to_bin_width(hist):
    for i in range(hist.GetNbinsX()):
        bin_width = hist.GetXaxis().GetBinWidth(i + 1)
        hist.SetBinContent(i + 1, hist.GetBinContent(i + 1) / bin_width)


def get_spec_from_root(fname, spec_path, live_path, isParam=False, xScale=1, rebin=1, projectionX=False, has_live=True):
    myFile = TFile.Open(fname, "READ")
    myHist = myFile.Get(spec_path)
    if projectionX:
        myHist = myHist.ProjectionX()
    if rebin != 1:
        myHist.Rebin(rebin)
    data = np.zeros((myHist.GetNbinsX(),))
    for i in range(myHist.GetNbinsX()):
        data[i] = myHist.GetBinContent(i + 1)
    if has_live:
        if isParam:
            live = abs(myFile.Get(live_path).GetVal())
        else:
            live = abs(myFile.Get(live_path)[0])
    else:
        live = 1 #careful, cannot perform any addition or subtraction
    A1 = myHist.GetXaxis().GetBinLowEdge(2) - myHist.GetXaxis().GetBinLowEdge(1)
    A0 = myHist.GetXaxis().GetBinLowEdge(1) - A1 / 2.
    if (xScale != 1):
        A1 *= xScale
        A0 *= xScale
    start = 0
    spec = SpectrumData(data, start, live, A0, A1, fname)
    myFile.Close()
    return spec


def fix_table(fpath):
    base_path = os.path.dirname(fpath)

    def write_rows_new_file(f, header, rows, n):
        if len(header) == 0:
            return f, n
        write_rows_csv(f, header, rows)
        f.flush()
        f.close()
        f = open(join(base_path, "position_scan_{}.csv".format(n + 1)), 'w')
        return f, n + 1

    with open(fpath, 'r') as read_f:
        cur_header = []
        cur_rows = []
        cur_row = []
        check_header = False
        n_csv = 1
        f = open(join(base_path, "position_scan_{}.csv".format(n_csv)), 'w')
        lines = read_f.readlines()
        line_num = len(lines)
        for line_number, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            if line == "direction" and check_header is False:
                cur_rows.append(cur_row)
                print(cur_header)
                print(cur_rows)
                print(n_csv)
                f, n_csv = write_rows_new_file(f, cur_header, cur_rows, n_csv)
                cur_header = [line]
                check_num = False
                check_header = True
                cur_rows = []
                cur_row = []
            elif (line.startswith("R ") or line == "R") and check_header is False:
                if len(cur_header) > 0:
                    cur_rows.append(cur_row)
                    f, n_csv = write_rows_new_file(f, cur_header, cur_rows, n_csv)
                cur_header = [line]
                cur_rows = []
                cur_row = []
                check_num = True
                check_header = True
            else:
                if check_header:
                    if check_num:
                        data = line.split(',')
                        if len(data) == 2 and is_number(data[0].strip()) and is_number(data[1].strip()):
                            check_header = False
                    else:
                        if line == "north":
                            check_header = False
                if check_header:
                    cur_header.append(line)
                else:
                    if len(cur_row) == len(cur_header):
                        cur_rows.append(copy(cur_row))
                        cur_row = []
                    cur_row.append(line)
            if line_number == line_num - 2:
                write_rows_new_file(f, cur_header, cur_rows, n_csv)


def fit_spectra(data, expected_peaks, plot_dir=None, user_verify=False, plot_fit=False, auto_set_offset=True):
    fit_data = {}
    for name, spec in data.items():
        spec_fitter = SpectrumFitter(expected_peaks, name=name)
        spec_fitter.auto_set_offset_factor = auto_set_offset
        spec_fitter.fit_peaks(spec)
        if plot_dir is not None:
            plot_dir = join(plot_dir, name)
            if not os.path.exists(plot_dir):
                os.mkdir(plot_dir)
        if plot_fit or user_verify:
            spec_fitter.plot_fits(user_verify, plot_dir)
        # _, _ = spec_fitter.retrieve_calibration(user_verify, tolerate_fails, plot_dir, plot_fit)
        fit_data.update(spec_fitter.fit_values)
    return fit_data

def retrieve_peak_areas(spec, peaks):
    spec_fitter = SpectrumFitter(peaks)
    spec_fitter.fit_peaks(spec)
    areas = {}
    dareas = {}
    get_areas(spec_fitter.fit_values, areas, dareas, lt=spec.live)
    return areas, dareas


def calibrate_spectra(data, expected_peaks, db, plot_dir=None, user_verify=False, tolerate_fails=False, plot_fit=False, allow_undetermined=False):
    for name, spec in data.items():
        spec_fitter = SpectrumFitter(expected_peaks, name=name)
        try:
            spec_fitter.fit_peaks(spec)
            if plot_dir is not None:
                plot_dir = join(plot_dir, name)
                if not os.path.exists(plot_dir):
                    os.mkdir(plot_dir)
            cal, _ = spec_fitter.retrieve_calibration(user_verify, tolerate_fails, plot_dir, plot_fit, allow_undetermined=allow_undetermined)
            if cal:
                db.insert_calibration(cal[0], cal[1], spec.fname)
        except Exception as e:
            print("failed to fit peak for spectrum {0}, error: {1}".format(spec.fname, e))


def calibrate_nearby_runs(data, expected_peaks, db, plot_dir=None, user_verify=False, tolerate_fails=False,
                          plot_fit=False, dt=604800):
    files_to_calibrate = set()
    nearby_groups = set()
    for name, spec in data.items():
        if spec.fname.endswith(".txt"):
            files_to_calibrate.add(spec.fname[0:-4])
        else:
            files_to_calibrate.add(spec.fname)
    for name, spec in data.items():
        spec_fitter = SpectrumFitter(expected_peaks, name=name)
        spec_fitter.fit_peaks(spec)
        if plot_dir is not None:
            plot_dir = join(plot_dir, name)
            if not os.path.exists(plot_dir):
                os.mkdir(plot_dir)
        cal, _ = spec_fitter.retrieve_calibration(user_verify, tolerate_fails, plot_dir, plot_fit)
        if cal:
            db.insert_calibration(cal[0], cal[1], spec.fname)
            # retrieve nearby runs, update those calibrations
            nearby_calgroups = db.retrieve_compatible_calibration_groups(spec.fname, dt)
            for group_id in nearby_calgroups:
                # ignore groups that are already being calibrated in files_to_calibrate
                file_names = db.retrieve_file_names_from_calibration_group_id(group_id)
                cal_this = True
                for fname in file_names:
                    if fname in files_to_calibrate:
                        cal_this = False
                        break
                if cal_this:
                    db.update_calibration(group_id, cal[0], cal[1], True)
                    nearby_groups.add(group_id)
    return nearby_groups


def get_sigmas(peak_data, sigmas, dsigmas):
    for peak, data in peak_data.items():
        if isinstance(peak, str):
            peak = peak.split(',')
            for i, p in enumerate(peak):
                p = float(p)
                sigmas["{:.2f}".format(p)] = data.sigmas[i]
                dsigmas["{:.2f}".format(p)] = data.sigma_errs[i]
        else:
            peak = float(peak)
            sigmas["{:.2f}".format(peak)] = data.sigma
            dsigmas["{:.2f}".format(peak)] = data.sigma_err

def get_skews(peak_data, skews, dskews, Rs, dRs):
    for peak, data in peak_data.items():
        if isinstance(peak, str):
            peak = peak.split(',')
            for i, p in enumerate(peak):
                p = float(p)
                skews["{:.2f}".format(p)] = data.skews[i]
                dskews["{:.2f}".format(p)] = data.skew_errs[i]
                Rs["{:.2f}".format(p)] = data.Rs[i]
                dRs["{:.2f}".format(p)] = data.R_errs[i]
        else:
            p = float(peak)
            skews["{:.2f}".format(p)] = data.skew
            dskews["{:.2f}".format(p)] = data.skew_err
            Rs["{:.2f}".format(p)] = data.R
            dRs["{:.2f}".format(p)] = data.R_err


def get_areas(peak_data, areas, dareas, lt=1):
    """lt is live time for time normalization"""
    for peak, data in peak_data.items():
        if isinstance(peak, str):
            peak = peak.split(',')
            a = data.area()
            for i, p in enumerate(peak):
                p = float(p)
                areas["{:.2f}".format(p)], dareas["{:.2f}".format(p)] = tuple([b/lt for b in a[i]])
        else:
            peak = float(peak)
            areas["{:.2f}".format(peak)], dareas["{:.2f}".format(peak)] = tuple([b/lt for b in data.area()])


def get_peak_fit_data(peak_data, d):
    # height, ratio, centroid, stdev, skewness,
    for peak, data in peak_data.items():
        if isinstance(peak, str):
            peak = peak.split(',')
            for i, p in enumerate(peak):
                p = float(p)
                key = "{:.2f}".format(p)
                params = data.get_peak_parameters(i)
                errors = data.get_peak_errors(i)
                d[key] = []
                for param, e in zip(params, errors):
                    d[key].append(param)
                    d[key].append(e)
                d[key].append(data.parameters[-3])
                d[key].append(data.errors[-3])
                d[key].append(data.parameters[-2])
                d[key].append(data.errors[-2])
                d[key].append(data.parameters[-1])
                d[key].append(data.errors[-1])
        else:
            peak = float(peak)
            key = "{:.2f}".format(peak)
            d[key] = []
            for p, e in zip(data.parameters, data.errors):
                d[key].append(p)
                d[key].append(e)


def unc_ratio(A, B, dA, dB):
    Asqr = A * A
    Bsqr = B * B
    return np.sqrt((Asqr / Bsqr) * ((dA * dA / Asqr) + (dB * dB / Bsqr)))


def fit_peak_sigmas(data, expected_peaks, plot_dir=None, user_verify=False,
                    plot_fit=False, use_sqrt_fit=False):
    print("fitting peaks for data")
    peak_data = fit_spectra(data, expected_peaks, plot_dir, user_verify, plot_fit)
    sigmas = {}
    dsigmas = {}
    skews = {}
    dskews = {}
    Rs = {}
    dRs = {}
    get_sigmas(peak_data, sigmas, dsigmas)
    get_skews(peak_data, skews, dskews, Rs, dRs)
    ens = []
    sigs = []
    dsigs = []
    sks = []
    dsks = []
    r = []
    dr = []
    for e in sigmas.keys():
        ens.append(float(e))
        sigs.append(sigmas[e])
        dsigs.append(dsigmas[e])
        sks.append(skews[e])
        dsks.append(dskews[e])
        r.append(Rs[e])
        dr.append(dRs[e])
    ens = np.array(ens)
    sigs = np.array(sigs)
    dsigs = np.array(dsigs)
    sks = np.array(sks)
    dsks = np.array(dsks)
    r = np.array(r)
    dr = np.array(dr)
    name_app = "linear"
    if use_sqrt_fit:
        name_app = "sqrt"
    if plot_dir is not None:
        nm = ""
        for key in data.keys():
            nm = key
            break
        plot_name = nm + "_E_sigma_{0}_fit".format(name_app)
        if use_sqrt_fit:
            coeff, cov, chisqr = sqrtfit(ens, sigs, dsigs, join(plot_dir, plot_name), "Energy [keV]",
                                         "Peak Sigma [keV]")
        else:
            coeff, cov, chisqr, _ = linfit(ens, sigs, dsigs, join(plot_dir, plot_name), "Energy [keV]",
                                           "Peak Sigma [keV]")
        plot_name = nm + "_E_skew_{0}_fit".format(name_app)
        skew_coeff, skew_cov, skew_chisqr, _ = linfit(ens, sks, dsks, join(plot_dir, plot_name), "Energy [keV]", "skewness [keV]")
        plot_name = nm + "_E_R_{0}_fit".format(name_app)
        r_coeff, r_cov, r_chisqr, _ = linfit(ens, r, dr, join(plot_dir, plot_name), "Energy [keV]", "R")
    else:
        if use_sqrt_fit:
            coeff, cov, chisqr = sqrtfit(ens, sigs, dsigs)
        else:
            coeff, cov, chisqr, _ = linfit(ens, sigs, dsigs)
        skew_coeff, skew_cov, skew_chisqr, _ = linfit(ens, sks, dsks)
        r_coeff, r_cov, r_chisqr, _ = linfit(ens, r, dr)
    errs = np.sqrt(np.diag(cov))
    skew_errs = np.sqrt(np.diag(skew_cov))
    r_errs = np.sqrt(np.diag(r_cov))
    n_param = 2
    if use_sqrt_fit:
        n_param = 3
        B = 1 / (coeff[1] ** 2)
        errB = abs(B * 0.5 * errs[1] / coeff[1])
        print(
            "fit values are sigma = A + sqrt(E/B) + C*E, A = {0} ~ {1} keV, B = {2} ~ {3} charge/keV, C = {4} ~ {5}".format(
                coeff[0], errs[0], B, errB, coeff[2], errs[2]))
    else:
        print("fit values are sigma = A + B*(E), A = {0} ~ {1} keV, B = {2} ~ {3}".format(coeff[1], errs[1], coeff[0],
                                                                                          errs[0]))

    print("fit values are skew = A + B*(E), A = {0} ~ {1} keV, B = {2} ~ {3}".format(skew_coeff[1], skew_errs[1], skew_coeff[0], skew_errs[0]))
    print("fit values are R = A + B*(E), A = {0} ~ {1}, B = {2} ~ {3} 1/keV".format(r_coeff[1], r_errs[1], r_coeff[0], r_errs[0]))
    alpha = 0.05
    p_value = 1 - stats.chi2.cdf(chisqr, len(ens) - n_param)
    conclusion = "Failed to reject the null hypothesis."
    if p_value <= alpha:
        conclusion = "Null Hypothesis is rejected."

    print("chisquare of fit is {0} with p value {1} and {2} degrees of freedom".format(chisqr, p_value, len(ens) - n_param))
    print(conclusion)


def compare_peaks(data, simdata, expected_peaks, plot_dir=None, user_verify=False, plot_fit=False):
    print("fitting peaks for data")
    peak_data = fit_spectra(data, expected_peaks, plot_dir, user_verify, plot_fit)
    print("fitting peaks for simulation")
    peak_data_sim_true = {}
    peak_data_sim_false = {}
    for sd in simdata.keys():
        for p in expected_peaks:
            if str(p) in sd:
                mysim = {sd: simdata[sd]}
                if "false" in sd:
                    peak_data_sim_false.update(fit_spectra(mysim, [p, p - 511, p - 511 * 2], plot_dir, user_verify,
                                                           plot_fit))
                else:
                    peak_data_sim_true.update(
                        fit_spectra(mysim, [p, p - 511, p - 511 * 2], plot_dir, user_verify, plot_fit))
    ratios_data = {}
    ratios_data_sim_true = {}
    ratios_data_sim_false = {}
    orig_peaks = []
    areas = {}
    areas_sim_true = {}
    areas_sim_false = {}
    dareas = {}
    dareas_sim_true = {}
    dareas_sim_false = {}
    for p in expected_peaks:
        if p - 511 in expected_peaks and p - 2 * 511 in expected_peaks:
            orig_peaks.append(p)
    peak_data_vals = {}
    peak_data_sim_true_vals = {}
    peak_data_sim_false_vals = {}
    get_areas(peak_data, areas, dareas)
    get_areas(peak_data_sim_true, areas_sim_true, dareas_sim_true)
    get_areas(peak_data_sim_false, areas_sim_false, dareas_sim_false)
    get_peak_fit_data(peak_data, peak_data_vals)
    get_peak_fit_data(peak_data_sim_true, peak_data_sim_true_vals)
    get_peak_fit_data(peak_data_sim_false, peak_data_sim_false_vals)
    for peak in orig_peaks:
        key = "{:.2f}".format(peak)
        key1 = "{:.2f}".format(peak - 511)
        key2 = "{:.2f}".format(peak - 2 * 511)
        ratios_data[key] = [areas[key] / areas[key1], unc_ratio(areas[key], areas[key1], dareas[key], dareas[key1]),
                            areas[key] / areas[key2], unc_ratio(areas[key], areas[key2], dareas[key], dareas[key2]),
                            areas[key1] / areas[key2], unc_ratio(areas[key1], areas[key2], dareas[key1], dareas[key2])]
        ratios_data_sim_true[key] = [areas_sim_true[key] / areas_sim_true[key1],
                                     unc_ratio(areas_sim_true[key], areas_sim_true[key1], dareas_sim_true[key],
                                               dareas_sim_true[key1]),
                                     areas_sim_true[key] / areas_sim_true[key2],
                                     unc_ratio(areas_sim_true[key], areas_sim_true[key2], dareas_sim_true[key],
                                               dareas_sim_true[key2]),
                                     areas_sim_true[key1] / areas_sim_true[key2],
                                     unc_ratio(areas_sim_true[key1], areas_sim_true[key2], dareas_sim_true[key1],
                                               dareas_sim_true[key2])]
        ratios_data_sim_false[key] = [areas_sim_false[key] / areas_sim_false[key1],
                                      unc_ratio(areas_sim_false[key], areas_sim_false[key1], dareas_sim_false[key],
                                                dareas_sim_false[key1]),
                                      areas_sim_false[key] / areas_sim_false[key2],
                                      unc_ratio(areas_sim_false[key], areas_sim_false[key2], dareas_sim_false[key],
                                                dareas_sim_false[key2]),
                                      areas_sim_false[key1] / areas_sim_false[key2],
                                      unc_ratio(areas_sim_false[key1], areas_sim_false[key2], dareas_sim_false[key1],
                                                dareas_sim_false[key2])]
    print("peak data")
    header = ["energy [keV]", "type", "area", "darea", "height", "dheight", "ratio", "dratio", "centroid", "dcentroid",
              "sigma", "dsigma", "beta", "dbeta", "A", "dA", "B", "dB", "C", "dC"]
    print("| {} |".format(" | ".join(header)))
    print("| {} |".format(" | ".join(["---"] * len(header))))
    rows = []
    for e, dat in peak_data_vals.items():
        rows.append([e, "data", areas[e], dareas[e]] + dat)
        if e in peak_data_sim_true_vals.keys():
            rows.append([e, "sim through", areas_sim_true[e], dareas_sim_true[e]] + peak_data_sim_true_vals[e])
        if e in peak_data_sim_false_vals.keys():
            rows.append([e, "sim all", areas_sim_false[e], dareas_sim_false[e]] + peak_data_sim_false_vals[e])
    ens = []
    sigmas = [[], [], []]
    sig_errs = [[], [], []]
    for row in rows:
        strow = ["{:.2f}".format(r) if isinstance(r, float) else r for r in row]
        print("| {} |".format(" | ".join(strow)))
        if row[1] == "data":
            ens.append(float(row[0]))
            sigmas[0].append(row[10])
            sig_errs[0].append(row[11])
        elif row[1] == "sim through":
            sigmas[1].append(row[10])
            sig_errs[1].append(row[11])
        else:
            sigmas[2].append(row[10])
            sig_errs[2].append(row[11])
    if plot_dir is not None:
        namebase = ""
        for k in data.keys():
            namebase = k
        out_fname = join(plot_dir, namebase + "_peak_data.csv")
        print("writing peak data to {}".format(out_fname))
        f = open(out_fname, 'w')
        write_rows_csv(f, header, rows, delimiter=",")
        f.flush()
        f.close()
        MultiScatterPlot(ens, sigmas, sig_errs, ["real", "sim through", "sim all"], "peak energy [keV]",
                         "peak sigma [keV]", ylog=False)
        plt.savefig(join(plot_dir, namebase + "_peak_sigmas.png"))
    rows = []
    header = ["energy [keV]", "type", "ratio 0/1", "dratio 0/1", "ratio 0/2", "dratio 0/2", "ratio 1/2", "dratio 1/2"]
    for e, ratios in ratios_data.items():
        rows.append([e, "data"] + ratios)
        rows.append([e, "sim through"] + ratios_data_sim_true[e])
        rows.append([e, "sim all"] + ratios_data_sim_false[e])
    print("| {} |".format(" | ".join(header)))
    print("| {} |".format(" | ".join(["---"] * len(header))))
    ens = []
    ratios01 = [[], [], []]
    ratios12 = [[], [], []]
    ratios02 = [[], [], []]
    ratios01_e = [[], [], []]
    ratios12_e = [[], [], []]
    ratios02_e = [[], [], []]
    for row in rows:
        strow = ["{:.2f}".format(r) if isinstance(r, float) else r for r in row]
        print("| {} |".format(" | ".join(strow)))
        if row[1] == "data":
            ens.append(float(row[0]))
            ratios01[0].append(row[2])
            ratios01_e[0].append(row[3])
            ratios02[0].append(row[4])
            ratios02_e[0].append(row[5])
            ratios12[0].append(row[6])
            ratios12_e[0].append(row[7])
        elif row[1] == "sim through":
            ratios01[1].append(row[2])
            ratios01_e[1].append(row[3])
            ratios02[1].append(row[4])
            ratios02_e[1].append(row[5])
            ratios12[1].append(row[6])
            ratios12_e[1].append(row[7])
        else:
            ratios01[2].append(row[2])
            ratios01_e[2].append(row[3])
            ratios02[2].append(row[4])
            ratios02_e[2].append(row[5])
            ratios12[2].append(row[6])
            ratios12_e[2].append(row[7])
    if plot_dir is not None:
        nm = ""
        for k in data.keys():
            nm = k
        out_fname = join(plot_dir, nm + "_peak_ratios.csv")
        print("writing peak ratios to {}".format(out_fname))
        f = open(out_fname, 'w')
        write_rows_csv(f, header, rows, delimiter=",")
        f.flush()
        f.close()
        ens = np.array(ens)
        ratios01 = np.array(ratios01)
        ratios01_e = np.array(ratios01_e)
        ratios02 = np.array(ratios02)
        ratios02_e = np.array(ratios02_e)
        ratios12 = np.array(ratios12)
        ratios12_e = np.array(ratios12_e)
        MultiScatterPlot(ens, ratios01, ratios01_e, ["real", "sim through", "sim all"], "full energy [keV]",
                         "full to 1st escape area ratio", ylog=False)
        plt.savefig(join(plot_dir, nm + "_peak_ratios_01.png"))
        ScatterDifferencePlot(ens, ratios01[0], ratios01_e[0], ratios01[1:], ratios01_e[1:],
                              ["sim through - real", "sim all - real"], "energy [keV]",
                              "full to 1st escape area ratio difference")
        plt.savefig(join(plot_dir, nm + "_peak_ratios_01_diff.png"))
        MultiScatterPlot(ens, ratios02, ratios02_e, ["real", "sim through", "sim all"], "full energy [keV]",
                         "full to 2nd escape area ratio", ylog=False)
        plt.savefig(join(plot_dir, nm + "_peak_ratios_02.png"))
        ScatterDifferencePlot(ens, ratios02[0], ratios02_e[0], ratios02[1:], ratios02_e[1:],
                              ["sim through - real", "sim all - real"], "energy [keV]",
                              "full to 2nd escape area ratio difference")
        plt.savefig(join(plot_dir, nm + "_peak_ratios_02_diff.png"))
        MultiScatterPlot(ens, ratios12, ratios12_e, ["real", "sim through", "sim all"], "full energy [keV]",
                         "1st to 2nd escape area ratio", ylog=False)
        plt.savefig(join(plot_dir, nm + "_peak_ratios_12.png"))
        ScatterDifferencePlot(ens, ratios12[0], ratios12_e[0], ratios12[1:], ratios12_e[1:],
                              ["sim through - real", "sim all - real"], "energy [keV]",
                              "1st to 2nd escape area ratio difference")
        plt.savefig(join(plot_dir, nm + "_peak_ratios_12_diff.png"))


def get_rd_data(db, rxon_only=False, rxoff_only=False, min_time=100):
    rd_data = db.get_rd_files(min_time=min_time, rxon_only=rxon_only, rxoff_only=rxoff_only)
    rd_data = populate_rd_data(rd_data, db)
    dels = []
    for key in rd_data:
        del_keys = []
        for key2 in rd_data[key]:
            if not rd_data[key][key2]:
                del_keys.append(key2)
        for key2 in del_keys:
            del rd_data[key][key2]
        if rd_data[key]:
            combine_runs(rd_data[key], ignore_failed_add=True)
        else:
            dels.append(key)
    for key in dels:
        del rd_data[key]
    return rd_data


def subtract_rd_data(rd_data, rd_data_off, acq_id_bin_edges=None):
    sub_data = {}
    for key in rd_data_off.keys():
        if not key in sub_data.keys():
            sub_data[key] = {}
        for key2 in rd_data_off[key].keys():
            if key2 in acq_id_bin_edges.keys():
                sub_data[key][key2] = SubtractSpectrum(rd_data[key][key2], rd_data_off[key][key2], bin_edges=acq_id_bin_edges[key2])
            else:
                sub_data[key][key2] = SubtractSpectrum(rd_data[key][key2], rd_data_off[key][key2])
    return sub_data




if __name__ == "__main__":
    fix_table(os.path.expanduser("~/src/HFIR_BG_Analysis/db/position_scans.txt"))
