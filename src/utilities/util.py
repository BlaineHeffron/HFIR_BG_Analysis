import csv
import os
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

from src.analysis.Spectrum import SpectrumData, SpectrumFitter
from src.utilities.PlotUtils import MultiLinePlot, MultiScatterPlot, scatter_plot
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


def retrieve_spectra_and_files(n, datadir):
    fs = retrieve_files(datadir)
    return retrieve_spectra(n, fs), fs


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
            #print(
            #    "found calibration for file {0} in database, using values A0 = {1}, A1 = {2} instead of A0 = {3}, A1 = {4}".format(
            #        fname, row[0], row[1], A0, A1))
            A0 = row[0]
            A1 = row[1]

    return SpectrumData(data, start, live, A0, A1, fname)


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


def get_data_dir():
    if "HFIRBGDATA" not in os.environ:
        raise RuntimeError("Error: set environment variable HFIRBGDATA to the directory with your files")
    return os.path.expanduser(os.environ["HFIRBGDATA"])


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

def populate_data_config(config,db):
    """
    given config, a dictionary that specifies the files to retrive using database, populates
    a new dictionary with  keys as filenames and values as spectrumdata objects
    """
    fs = db.get_files_from_config(config)
    out = {}
    for f in fs:
        out[f] = retrieve_spectra(f, f, db)
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


def plot_multi_spectra(fdict, n, rebin=1, emin=20, emax=None, loc="upper right"):
    ys = []
    names = []
    x = []
    end_index = 0
    start_index = 0
    for name in fdict.keys():
        if isinstance(fdict[name], SpectrumData):
            spec = fdict[name]
        else:
            raise Exception("not supported")
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
    fig = MultiLinePlot(x, ys, names, "Energy [keV]", "Rate [hz/keV]", ylog=True)
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
    err = [errs[i] / spec.live / (spec.bin_edges[i + 1] - spec.bin_edges[i]) for i in range(len(spec.bin_edges) - 1)]
    err = err[start_index:end_index]
    MultiScatterPlot(x, [y], [err], [name], "Energy [keV]", "Rate [hz/keV]")
    plt.savefig("{}_errors.png".format(name))
    plt.close()
    MultiLinePlot(x, [y], [name], "Energy [keV]", "Rate [hz/keV]")
    plt.savefig("{}.png".format(name))
    plt.close()


def plot_time_series(data, outdir, emin=30, emax=None):
    if emin is None and emax is None:
        energystring="full"
    elif emin is None and emax is not None:
        energystring="below{0}keV".format(emax)
    elif emax is None and emin is not None:
        energystring="above{0}keV".format(emin)
    else:
        energystring="{0}-{1}keV".format(emin,emax)
    xlabel = "date"
    for key in data.keys():
        if not isinstance(data[key], list):
            print("{} is not a list of data, skipping".format(key))
            continue
        plot_name = join(outdir, "{0}_{1}.png".format(key, energystring))
        ylabel = "rate [Hz/keV]"
        rates = []
        drates = []
        times = []
        e1, e2 = data[key][0].get_e_limits(emin, emax)
        line_label = ["{0:.0f} - {1:.0f} keV".format(e1,e2)]
        for spec in data[key]:
            r, dr = spec.integrate(emin, emax, norm=True)
            rates.append(r)
            drates.append(dr)
            times.append(spec.start_timestamp())
        MultiScatterPlot(times, [rates], [drates], line_label, xlabel, ylabel, xdates=True)

        plt.savefig(plot_name)
        plt.close()

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
            if line.startswith("# Start time:"):
                data = line[13:].strip()
                dt = datetime.strptime(data, "%Y-%m-%d, %H:%M:%S")
            if line.startswith("# Live time:"):
                data = line[16:].strip()
                lt = float(data)
                break
    if dt:
        return dt.timestamp(), lt
    else:
        print("couldnt find datetime, using creation date of file")
        return creation_date(file), lt


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
        for i, colname in enumerate(data[0]):
            if colname == "R" or colname.startswith("R"):
                r_pos = i
            elif colname == "L" or colname.startswith("L"):
                l_pos = i
            elif colname.startswith("angle") or colname.startswith("theta"):
                angle_pos = i
            elif colname.startswith("file"):
                fname_pos = i
        if r_pos == -1 or l_pos == -1 or angle_pos == -1 or fname_pos == -1:
            raise IOError("Error: cannot parse file {0}, cant find header info".format(f))
        for row in data[1:]:
            position_metadata.append((row[r_pos], row[l_pos], row[angle_pos], row[fname_pos]))
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


def get_spec_from_root(fname, spec_path, live_path, isParam=False, xScale=1, rebin=1):
    myFile = TFile.Open(fname, "READ")
    myHist = myFile.Get(spec_path)
    if rebin != 1:
        myHist.Rebin(rebin)
    data = np.zeros((myHist.GetNbinsX(),))
    for i in range(myHist.GetNbinsX()):
        data[i] = myHist.GetBinContent(i + 1)
    if isParam:
        live = myFile.Get(live_path).GetVal()
    else:
        live = myFile.Get(live_path)[0]
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


def fit_spectra(data, expected_peaks, plot_dir=None, user_verify=False, tolerate_fails=False, plot_fit=False):
    fit_data = {}
    for name, spec in data.items():
        spec_fitter = SpectrumFitter(expected_peaks, name=name)
        spec_fitter.fit_peaks(spec)
        if plot_dir is not None:
            plot_dir = join(plot_dir, name)
            if not os.path.exists(plot_dir):
                os.mkdir(plot_dir)
        _, _ = spec_fitter.retrieve_calibration(user_verify, tolerate_fails, plot_dir, plot_fit)
        fit_data.update(spec_fitter.fit_values)
    return fit_data


def calibrate_spectra(data, expected_peaks, db, plot_dir=None, user_verify=False, tolerate_fails=False, plot_fit=False):
    for name, spec in data.items():
        spec_fitter = SpectrumFitter(expected_peaks, name=name)
        spec_fitter.fit_peaks(spec)
        if plot_dir is not None:
            plot_dir = join(plot_dir, name)
            if not os.path.exists(plot_dir):
                os.mkdir(plot_dir)
        cal, _ = spec_fitter.retrieve_calibration(user_verify, tolerate_fails, plot_dir, plot_fit)
        db.insert_calibration(cal[0], cal[1], spec.fname)


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

def get_areas(peak_data, areas, dareas):
    for peak, data in peak_data.items():
        if isinstance(peak, str):
            peak = peak.split(',')
            a = data.area()
            for i, p in enumerate(peak):
                p = float(p)
                areas["{:.2f}".format(p)], dareas["{:.2f}".format(p)] = a[i]
        else:
            peak = float(peak)
            areas["{:.2f}".format(peak)], dareas["{:.2f}".format(peak)] = data.area()


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


def fit_peak_sigmas(data,  expected_peaks, plot_dir=None, user_verify=False, tolerate_fails=False,
                      plot_fit=False, use_sqrt_fit=False):
    print("fitting peaks for data")
    peak_data = fit_spectra(data, expected_peaks, plot_dir, user_verify, tolerate_fails, plot_fit)
    sigmas = {}
    dsigmas = {}
    get_sigmas(peak_data, sigmas, dsigmas)
    ens = []
    sigs = []
    dsigs = []
    for e in sigmas.keys():
        ens.append(float(e))
        sigs.append(sigmas[e])
        dsigs.append(dsigmas[e])
    ens = np.array(ens)
    sigs = np.array(sigs)
    dsigs = np.array(dsigs)
    name_app = "linear"
    ens /= 1000.
    sigs /= 1000.
    dsigs /= 1000.
    if use_sqrt_fit:
        name_app = "sqrt"
    if plot_dir is not None:
        nm = ""
        for key in data.keys():
            nm = key
            break
        plot_name = nm + "_E_sigma_{0}_fit".format(name_app)
        if use_sqrt_fit:
            coeff, cov, chisqr = sqrtfit(ens,sigs,dsigs,join(plot_dir,plot_name),"Energy [keV]", "Peak Sigma [keV]")
        else:
            coeff, cov, chisqr = linfit(ens, sigs, dsigs, join(plot_dir, plot_name), "Energy [keV]", "Peak Sigma [keV]")
    else:
        if use_sqrt_fit:
            coeff, cov, chisqr = sqrtfit(ens,sigs,dsigs)
        else:
            coeff, cov, chisqr = linfit(ens, sigs, dsigs)
    errs = np.sqrt(np.diag(cov))
    if use_sqrt_fit:
        B = 1/(coeff[1]**2)
        errB = abs(B*0.5*errs[1]/coeff[1])
        print("fit values are sigma = A + sqrt(E/B) + C*E, A = {0} ~ {1} keV, B = {2} ~ {3} charge/keV, C = {4} ~ {5}".format(coeff[0], errs[0], B, errB, coeff[2], errs[2]))
    else:
        print("fit values are sigma = A + B*(E), A = {0} ~ {1} keV, B = {2} ~ {3}".format(coeff[1], errs[1], coeff[0], errs[0]))
    alpha = 0.05
    p_value = 1 - stats.chi2.cdf(chisqr, len(ens)-1)
    conclusion = "Failed to reject the null hypothesis."
    if p_value <= alpha:
        conclusion = "Null Hypothesis is rejected."

    print("chisquare of fit is {0} with p value {1} and {2} degrees of freedom".format(chisqr, p_value, len(ens)-1))
    print(conclusion)



def compare_peaks(data, simdata, expected_peaks, plot_dir=None, user_verify=False, tolerate_fails=False,
                  plot_fit=False):
    print("fitting peaks for data")
    peak_data = fit_spectra(data, expected_peaks, plot_dir, user_verify, tolerate_fails, plot_fit)
    print("fitting peaks for simulation")
    peak_data_sim_true = {}
    peak_data_sim_false = {}
    for sd in simdata.keys():
        for p in expected_peaks:
            if str(p) in sd:
                mysim = {sd: simdata[sd]}
                if "false" in sd:
                    peak_data_sim_false.update(fit_spectra(mysim, [p, p - 511, p - 511 * 2], plot_dir, user_verify,
                                                           tolerate_fails, plot_fit))
                else:
                    peak_data_sim_true.update(
                        fit_spectra(mysim, [p, p - 511, p - 511 * 2], plot_dir, user_verify, tolerate_fails, plot_fit))
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
                                unc_ratio(areas_sim_true[key], areas_sim_true[key1], dareas_sim_true[key], dareas_sim_true[key1]),
                                areas_sim_true[key] / areas_sim_true[key2],
                                unc_ratio(areas_sim_true[key], areas_sim_true[key2], dareas_sim_true[key], dareas_sim_true[key2]),
                                areas_sim_true[key1] / areas_sim_true[key2],
                                unc_ratio(areas_sim_true[key1], areas_sim_true[key2], dareas_sim_true[key1], dareas_sim_true[key2])]
        ratios_data_sim_false[key] = [areas_sim_false[key] / areas_sim_false[key1],
                            unc_ratio(areas_sim_false[key], areas_sim_false[key1], dareas_sim_false[key], dareas_sim_false[key1]),
                            areas_sim_false[key] / areas_sim_false[key2],
                            unc_ratio(areas_sim_false[key], areas_sim_false[key2], dareas_sim_false[key], dareas_sim_false[key2]),
                            areas_sim_false[key1] / areas_sim_false[key2],
                            unc_ratio(areas_sim_false[key1], areas_sim_false[key2], dareas_sim_false[key1], dareas_sim_false[key2])]
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
    sigmas = [[],[],[]]
    sig_errs = [[],[],[]]
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
        MultiScatterPlot(ens, sigmas, sig_errs, ["real", "sim through", "sim all"], "peak energy [keV]", "peak sigma [keV]", ylog=False)
        plt.savefig(join(plot_dir,namebase + "_peak_sigmas.png"))
    rows = []
    header = ["energy [keV]", "type", "ratio 0/1", "dratio 0/1", "ratio 0/2", "dratio 0/2", "ratio 1/2", "dratio 1/2"]
    for e, ratios in ratios_data.items():
        rows.append([e, "data"] + ratios)
        rows.append([e, "sim through"] + ratios_data_sim_true[e])
        rows.append([e, "sim all"] + ratios_data_sim_false[e])
    print("| {} |".format(" | ".join(header)))
    print("| {} |".format(" | ".join(["---"] * len(header))))
    ens = []
    ratios01 = [[],[],[]]
    ratios12 = [[],[],[]]
    ratios02 = [[],[],[]]
    ratios01_e = [[],[],[]]
    ratios12_e = [[],[],[]]
    ratios02_e = [[],[],[]]
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
        MultiScatterPlot(ens, ratios01,ratios01_e, ["real", "sim through", "sim all"], "full energy [keV]", "full to 1st escape area ratio", ylog=False)
        plt.savefig(join(plot_dir,namebase + "_peak_ratios_01.png"))
        MultiScatterPlot(ens, ratios02,ratios02_e, ["real", "sim through", "sim all"], "full energy [keV]", "full to 2nd escape area ratio", ylog=False)
        plt.savefig(join(plot_dir,namebase + "_peak_ratios_02.png"))
        MultiScatterPlot(ens, ratios12,ratios12_e, ["real", "sim through", "sim all"], "full energy [keV]", "1st to 2nd escape area ratio", ylog=False)
        plt.savefig(join(plot_dir,namebase + "_peak_ratios_12.png"))


if __name__ == "__main__":
    fix_table(os.path.expanduser("~/src/HFIR_BG_Analysis/db/position_scans.txt"))
