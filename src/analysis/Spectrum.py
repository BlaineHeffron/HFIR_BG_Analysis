from copy import copy

import numba as nb
import numpy as np
from ROOT import TH1F
from math import sqrt, floor
from os.path import join
from itertools import combinations

from scipy.special import erfc
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


from src.utilities.FitUtils import linfit_to_calibration, linfit
from src.utilities.PlotUtils import ScatterLinePlot
from src.utilities.NumbaFunctions import average_median, integrate_lininterp_range, find_peaks


class SpectrumData:
    def __init__(self, data, start, live, A0, A1, fname):
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
        self.fname = fname
        self.rebin()

    def get_data_x(self):
        """return raw data x values"""
        return np.linspace(self.A0 + self.A1, self.A0 + self.A1 * self.data.shape[0], self.data.shape[0])

    def data_at_x(self, x):
        """return the index of the raw data closest to x"""
        closest = int(round((x - self.A0) / self.A1 - 1))
        if closest < 0:
            return 0
        elif closest >= self.data.shape[0]:
            return self.data.shape[0] - 1
        return closest

    def add(self, s):
        if s.A0 != self.A0 or s.A1 != self.A1:
            raise ValueError("error: spectrum  addition not possible without rebinning")
        self.data += s.data
        self.live += s.live

    def calculate_bin_midpoints(self):
        if self.bin_edges is None:
            raise RuntimeError("bin edges not set")
        else:
            self.bin_midpoints = np.zeros((self.bin_edges.shape[0] - 1,))
            for j in range(0, self.bin_edges.shape[0] - 1):
                self.bin_midpoints[j] = self.bin_edges[j] + (self.bin_edges[j + 1] - self.bin_edges[j]) / 2.

    def rebin_factor(self, factor):
        if factor <= 1:
            raise RuntimeError("rebin factor must be > 1")
        new_hist = rebin_data(self.hist[1:-1], factor)
        under = self.hist[0]
        over = self.hist[-1]
        self.hist = np.zeros((new_hist.shape[0] + 2,), dtype=np.float32)
        self.hist[0] = under
        self.hist[1] = over
        self.hist[1:-1] = new_hist
        old_edges = copy(self.bin_edges)
        self.bin_edges = np.zeros((self.hist.shape[0] - 1,))
        j = 0
        for i in range(old_edges.shape[0]):
            if i % factor == 0:
                self.bin_edges[j] = old_edges[i]
                j += 1
        self.bin_edges[-1] = old_edges[-1]
        self.nbins = self.bin_edges.shape[0] - 1
        self.calculate_bin_midpoints()

    def get_data_energies(self):
        return [self.A0 + self.A1 * i for i in range(1, self.data.shape[0] + 1)]

    def rebin(self, bin_edges=None):
        set_to_data = False
        if bin_edges is None:
            set_to_data = True
            bin_edges = np.zeros(self.data.shape[0] + 1)
            for i in range(self.data.shape[0] + 1):
                bin_edges[i] = self.A0 + self.A1 * i + self.A1 / 2.
        self.hist = np.zeros((bin_edges.shape[0] + 1,), dtype=np.float32)
        self.bin_edges = bin_edges
        self.calculate_bin_midpoints()
        self.nbins = bin_edges.shape[0] - 1
        if set_to_data:
            self.hist[1:-1] = self.data.astype(np.float32)
            return
        numba_rebin(self.data, self.hist, self.A0, self.A1, self.bin_edges)

    def hist_subtract(self, s):
        if self.bin_edges is None:
            self.rebin()
        if s.bin_edges is None:
            s.rebin()
        if not np.array_equal(s.bin_edges, self.bin_edges):
            s.rebin(self.bin_edges)
        # normalize before subtracting
        self.hist /= self.live
        self.hist -= (s.hist / s.live)
        self.hist[self.hist < 0] = 0
        self.hist *= self.live

    def get_normalized_hist(self):
        """
        :return: normalized histogram (no underflow and overflow)
        """
        norm = np.zeros((self.hist.shape[0] - 2,))
        for i in range(1, self.nbins + 1):
            norm[i - 1] = self.hist[i] / (self.bin_edges[i] - self.bin_edges[i - 1])
        norm /= self.live
        return norm

    def sum_hist_rate(self):
        """
        :return: sum of histogram (excluding underflow, overflow) divided by live time
        """
        return np.sum(self.hist[1:-1]) / self.live

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

    def generate_root_hist(self, name, title):
        bin_low = self.A0 + self.A1 / 2.
        bin_high = self.A0 + self.A1 * self.data.shape[0] + self.A1 / 2.
        hist = TH1F(name, title, self.data.shape[0], bin_low, bin_high)
        for i, d in enumerate(self.data):
            hist.SetBinContent(i + 1, d)
        return hist


def rebin_data(a, n):
    mod = a.shape[0] % n
    if mod != 0:
        a = pad_data(a, (a.shape[0] + n - mod,))
    shape = [int(a.shape[0] / n)]
    sh = shape[0], a.shape[0] // shape[0]
    a = a.reshape(sh).sum(axis=-1)
    return a


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


def pad_data(a, new_shape):
    result = np.zeros(new_shape)
    if len(new_shape) == 2:
        result[:a.shape[0], :a.shape[1]] = a
    else:
        result[:a.shape[0]] = a
    return result

def gauss(x, H, R, c, s):
    return H * (1. - R) * np.exp(-0.5 * np.power(((x - c) / s),2))

def dgaussdR(x, H, R, c, s):
    return -H * np.exp(-0.5 * np.power(((x - c) / s),2))

def dgaussds(x, H, R, c, s):
    return np.power(c-x,2)*gauss(x,H,R,c,s) / (s**3)

def dgaussdc(x, H, R, c, s):
    return (x-c)*gauss(x,H,R,c,s) / (s**2)


def skewguass(x, H, R, c, s, b):
    return H * R * np.exp((x - c) / b) * erfc(
        (x - c) / (sqrt(2) * s) + s / (sqrt(2) * b))

def dskewgaussds(x, H, R, c, s, b):
    return -(2 * H * R * np.exp((x - c) / b - np.power((s / (sqrt(2) * b) + (x - c) / (sqrt(2) * s)), 2)) * (1 / (sqrt(2) * b) - (x - c) / (sqrt(2) * s**2))) / sqrt(np.pi)

def dskewgaussdc(x, H, R, c, s, b):
    return sqrt(2 / np.pi) * H * R * np.exp((x - c) / b - np.power((s / (sqrt(2) * b) + (x - c) / (sqrt(2) * s)),2)) / s - (H * R * np.exp((x - c) / b) * erfc(s / (sqrt(2) * b) + (x - c) / (sqrt(2) * s))) / b

def dskewgaussdb(x, H, R, c, s, b):
    return (sqrt(2 / np.pi) * H * R * s * np.exp((x - c) / b - np.power((s / (sqrt(2) * b) + (x - c) / (sqrt(2) * s)), 2)) - (H * R * (x - c) * np.exp((x - c) / b) * erfc( s / (sqrt(2) * b) + (x - c) / (sqrt(2) * s)))) / (b**2)

def peak_background(x, H, step, c, s):
    return H * step * erfc((x - c) / (sqrt(2) * s))

def dpeak_backgrounddc(x, H, step, c, s):
    return (sqrt(2 / np.pi) * H * step * np.exp(-np.power((x - c), 2) / (2 * s**2))) / s

def dpeak_backgroundds(x, H, c, step, s):
    return (sqrt(2 / np.pi) * H * step * (x - c) * np.exp(-np.power((x - c), 2) / (2 * s**2))) / (s**2)

def background(x, A, B, C):
    return A + B*x + C*x*x

def dbackgrounddA(x):
    return np.ones(len(x))

def dbackgrounddB(x):
    return x

def dbackgrounddC(x):
    return x*x

def ge_peak_function(x, H, R, centroid, sigma, beta, A, B, C):
    """
    modelled after https://radware.phy.ornl.gov/gf3/
    """
    #return gauss(x, H, R, centroid, sigma) + skewguass(x, H, R, centroid, sigma, beta) + peak_background(x, H, step, centroid, sigma) + background(x, A, B, C)
    return gauss(x, H, R, centroid, sigma) + skewguass(x, H, R, centroid, sigma, beta) + background(x, A, B, C)


def ge_multi_peak_function(*params):
    if (len(params) - 4) % 5 == 0:
        n = int((len(params) - 4) / 5)
        if n <= 1:
            raise ValueError("Incorrect number of parameters, should be n*5 + 4 for n peaks")
        x = params[0]
        A = params[-3]
        B = params[-2]
        C = params[-1]
        total = np.zeros((len(x),))
        ind = 1
        while ind + 8 <= len(params):
            if ind == 1:
                total += ge_peak_function(x, *params[ind:ind+5], A, B, C)
            else:
                total += ge_peak_function(x, *params[ind:ind + 5], 0, 0, 0)
            ind += 5
        return total
    else:
        raise ValueError("Incorrect number of parameters, should be n*5 + 4 for n peaks")


def ge_peak_function_jac(x, H, R, centroid, sigma, beta, A, B, C):
    dH = gauss(x, 1, R, centroid, sigma) + skewguass(x, 1, R, centroid, sigma, beta) # + peak_background(x, 1, step, centroid, sigma)
    dR = dgaussdR(x, H, R, centroid, sigma) + skewguass(x, H, 1, centroid, sigma, beta)
    #dstep = peak_background(x, step, 1, centroid, sigma)
    dcentroid = dgaussdc(x, H, R, centroid, sigma) + dskewgaussdc(x, H, R, centroid, sigma, beta) #+ dpeak_backgrounddc(x, H, step, centroid, sigma)
    dsigma = dgaussds(x, H, R, centroid, sigma) + dskewgaussds(x, H, R, centroid, sigma, beta) #+ dpeak_backgroundds(x, H, step, centroid, sigma)
    dbeta = dskewgaussdb(x, H, R, centroid, sigma, beta)
    dA = dbackgrounddA(x)
    dB = dbackgrounddB(x)
    dC = dbackgrounddC(x)
    return np.hstack( ( dH.reshape(-1,1), dR.reshape(-1,1), dcentroid.reshape(-1,1), dsigma.reshape(-1,1), dbeta.reshape(-1,1), dA.reshape(-1,1), dB.reshape(-1,1), dC.reshape(-1,1)))

def ge_multi_peak_function_jac(*params):
    if (len(params) - 4) % 5 == 0:
        n = int((len(params) - 4) / 5)
        if n <= 1:
            raise ValueError("Incorrect number of parameters, should be n*5 + 4 for n peaks")
        x = params[0]
        A = params[-3]
        B = params[-2]
        C = params[-1]
        total = np.zeros((len(x),len(params)-1))
        ind = 1
        while ind + 8 <= len(params):
            jac = ge_peak_function_jac(x, *params[ind:ind+5], A, B, C)
            total[:, ind-1:ind+4] = jac[:, 0:5]
            if ind == 1:
                total[:, -3:] = jac[:, -3:]
            ind += 5
        return total
    else:
        raise ValueError("Incorrect number of parameters, should be n*5 + 4 for n peaks")



class MultiPeakFit:
    def __init__(self, parameters, errors, cov, xs, ys):
        self.parameters = parameters
        self.errors = errors
        self.cov = cov
        self.xs = xs
        self.ys = ys
        self.centroids = []
        self.sigmas = []
        self.sigma_errs = []
        self.centroid_errs = []
        ind = 0
        while ind + 8 <= len(self.parameters):
            self.centroids.append(self.parameters[ind + 2])
            self.sigmas.append(self.parameters[ind + 3])
            self.centroid_errs.append(self.errors[ind + 2])
            self.sigma_errs.append(self.errors[ind + 3])
            ind += 5

    def get_y(self):
        return ge_multi_peak_function(self.xs, *self.parameters)

    def bound_to_index(self, bounds):
        inds = [0,0]
        if bounds[0] < self.xs[0]:
            inds[0] = 0
        else:
            for i,x in enumerate(self.xs):
                if bounds[0] < x:
                    inds[0] = i - ((x - bounds[0]) / (self.xs[i] - self.xs[i-1]))
                    break
        if bounds[1] > self.xs[-1]:
            inds[1] = len(self.xs)
        else:
            for i,x in enumerate(self.xs):
                if bounds[1] < x:
                    inds[1] = i - ((x - bounds[1]) / (self.xs[i] - self.xs[i-1]))
                    break
        return inds

    def area(self):
        areas = []
        for centroid, sigma in zip(self.centroids, self.sigmas):
            bounds = [centroid - 3.5*sigma, centroid + 3.5*sigma]
            inds = self.bound_to_index(bounds)
            area = integrate_lininterp_range(self.ys, inds[0], inds[1])
            bgs = self.parameters[-3] + self.xs * self.parameters[-2] + self.xs * self.xs * self.parameters[-1]
            bgtot = integrate_lininterp_range(bgs, inds[0], inds[1])
            #dbgsqr = self.errors[-3] * self.errors[-3]# + self.xs * self.xs * self.errors[-2] * self.errors[-2] #+ self.xs * self.xs * self.xs * self.xs * self.errors[-1] * self.errors[-1]
            #dbgtotsqr = integrate_lininterp_range(dbgsqr, inds[0], inds[1])
            tot = area - bgtot
            #dtot = sqrt(area + dbgtotsqr)
            dtot = sqrt(area + .25 * bgtot * bgtot)
            areas.append((tot,dtot))
            #for x, y in zip(self.xs, self.ys):
            #    if (x > centroid - 3.5 * sigma) and (x < centroid + 3 * sigma):
            #        tot += y - self.parameters[-3] - self.parameters[-2] * x - self.parameters[-1] * x * x
            #        unc += (y + self.errors[-3] * self.errors[-3] + self.errors[-2] * self.errors[-2] * x * x + self.errors[-1]*self.errors[-1] * x * x * x * x)
        return areas

    def get_peak_parameters(self, peak_number):
        if peak_number > (len(self.centroids) - 1):
            raise ValueError("requested peak number larger than there are peaks available")
        return self.parameters[peak_number*5:peak_number*5+5]

    def get_peak_errors(self, peak_number):
        if peak_number > (len(self.centroids) - 1):
            raise ValueError("requested errors for peak number larger than there are peaks available")
        return self.errors[peak_number*5:peak_number*5+5]

    def plot(self, outname=None):
        yvals = self.ys
        linex = np.linspace(self.xs[0], self.xs[-1], 100)
        liney = ge_multi_peak_function(linex, *self.parameters)
        jac = ge_multi_peak_function_jac(linex, *self.parameters)
        lineerr = np.sqrt(np.diag(np.matmul(jac,np.matmul(self.cov, jac.T))))
        ScatterLinePlot(self.xs, yvals, np.sqrt(yvals), linex, liney, lineerr, ["fit", r'1 $\sigma$ error', "data"], "uncorrected energy [keV]", "counts")
        if outname is not None:
            plt.savefig(outname + ".png")

    def display(self):
        ind = 0
        while ind + 8 <= len(self.parameters):
            i = int(ind / 5) + 1
            print("H {0}: {1} ~ {2} counts".format(i,self.parameters[ind + 0], self.errors[ind + 0]))
            print("R {0}: {1} ~ {2} ".format(i,self.parameters[ind + 1], self.errors[ind + 1]))
            print("centroid {0}: {1} ~ {2} keV".format(i,self.parameters[ind + 2], self.errors[ind + 2]))
            print("std deviation {0}: {1} ~ {2} keV".format(i,self.parameters[ind + 3], self.errors[ind + 3]))
            print("skewness {0}: {1} ~ {2} keV".format(i,self.parameters[ind + 4], self.errors[ind + 4]))
            ind += 5
        print("A: {0} ~ {1} counts".format(self.parameters[-3], self.errors[-3]))
        print("B: {0} ~ {1} counts/keV".format(self.parameters[-2], self.errors[-2]))
        print("C: {0} ~ {1} counts/keV^2".format(self.parameters[-1], self.errors[-1]))
        #print("data energies:")
        #print(self.xs)
        #print("data values:")
        #print(self.ys)
        #print("fit values:")
        #print(self.get_y().astype('int'))

class PeakFit:
    def __init__(self, parameters, errors, cov, xs, ys):
        #self.c = parameters[0]
        #self.R = parameters[1]
        #self.step = parameters[2]
        self.centroid = parameters[2]
        self.sigma = parameters[3]
        self.sigma_err = errors[3]
        self.centroid_err = errors[2]
        #self.beta = parameters[5]
        #self.dc = errors[0]
        #self.dR = errors[1]
        #self.dstep = errors[2]
        #self.dcentroid = errors[3]
        #self.dsigma = errors[4]
        #self.dbeta = errors[5]
        self.parameters = parameters
        self.errors = errors
        self.cov = cov
        self.xs = xs
        self.ys = ys

    def get_y(self):
        return ge_peak_function(self.xs, *self.parameters)

    def bound_to_index(self, bounds):
        inds = [0,0]
        if bounds[0] < self.xs[0]:
            inds[0] = 0
        else:
            for i,x in enumerate(self.xs):
                if bounds[0] < x:
                    inds[0] = i - ((x - bounds[0]) / (self.xs[i] - self.xs[i-1]))
                    break
        if bounds[1] > self.xs[-1]:
            inds[1] = len(self.xs)
        else:
            for i,x in enumerate(self.xs):
                if bounds[1] < x:
                    inds[1] = i - ((x - bounds[1]) / (self.xs[i] - self.xs[i-1]))
                    break
        return inds

    def area(self):
        bounds = [self.centroid - 3.5 * self.sigma, self.centroid + 3.5 * self.sigma]
        inds = self.bound_to_index(bounds)
        area = integrate_lininterp_range(self.ys, inds[0], inds[1])
        bgs = self.parameters[-3] + self.xs*self.parameters[-2] + self.xs*self.xs*self.parameters[-1]
        bgtot = integrate_lininterp_range(bgs, inds[0], inds[1])
        #dbgsqr = self.errors[-3]*self.errors[-3] #+ self.xs*self.xs*self.errors[-2]*self.errors[-2]# + self.xs*self.xs*self.xs*self.xs*self.errors[-1]*self.errors[-1]
        #dbgtotsqr = integrate_lininterp_range(dbgsqr, inds[0], inds[1])
        tot = area - bgtot
        dtot = sqrt(area + .25*bgtot*bgtot)
        #dtot = sqrt(area + dbgtotsqr)
        return tot, dtot
        #tot = 0.
        #unc = 0.
        #for x, y in zip(self.xs, self.ys):
        #    if (x > self.centroid - 3*self.sigma) and (x < self.centroid + 3*self.sigma):
        #        tot += y - self.parameters[5] - self.parameters[6]*x - self.parameters[7]*x*x
        #        unc += (y + self.errors[5]*self.errors[5] + self.errors[6]*self.errors[6]*x*x + self.errors[7]*self.errors[7]*x*x*x*x)
        #return tot, sqrt(unc)


    def plot(self, outname=None):
        yvals = self.ys
        linex = np.linspace(self.xs[0], self.xs[-1], 100)
        liney = ge_peak_function(linex, *self.parameters)
        jac = ge_peak_function_jac(linex, *self.parameters)
        lineerr = np.sqrt(np.diag(np.matmul(jac,np.matmul(self.cov, jac.T))))
        ScatterLinePlot(self.xs, yvals, np.sqrt(yvals), linex, liney, lineerr, ["fit", r'1 $\sigma$ error', "data"], "uncorrected energy [keV]", "counts")
        if outname is not None:
            plt.savefig(outname + ".png")

    def display(self):
        print("H: {0} ~ {1} counts".format(self.parameters[0], self.errors[0]))
        print("R: {0} ~ {1} ".format(self.parameters[1], self.errors[1]))
        #print("step: {0} ~ {1} ".format(self.parameters[2], self.errors[2]))
        print("centroid: {0} ~ {1} keV".format(self.parameters[2], self.errors[2]))
        print("std deviation: {0} ~ {1} keV".format(self.parameters[3], self.errors[3]))
        print("skewness: {0} ~ {1} keV".format(self.parameters[4], self.errors[4]))
        print("A: {0} ~ {1} counts".format(self.parameters[5], self.errors[5]))
        print("B: {0} ~ {1} counts/keV".format(self.parameters[6], self.errors[6]))
        print("C: {0} ~ {1} counts/keV^2".format(self.parameters[7], self.errors[7]))
        #print("data energies:")
        #print(self.xs)
        #print("data values:")
        #print(self.ys)
        #print("fit values:")
        #print(self.get_y().astype('int'))


class SpectrumFitter:
    def __init__(self, expected_peaks=None, name=None):
        # 2.26 keV sigma for 11386 keV peak
        # 0.79 keV at 1332.5 keV peak
        # using a simple linear fit to estimate the window size we get sigma = .5918 + 0.0001465*x
        self.expected_peaks = expected_peaks
        self.peak_groups = []
        self.R = 0.1
        self.step = 0.002
        self.low_bound_gaus_scale = 0.8
        self.up_bound_gaus_scale = 1.2
        self.low_bound_skew_scale = 0.1
        self.up_bound_skew_scale = 3.0
        self.low_bound_bkg_scale = 0.1
        self.up_bound_bkg_scale = 30.0
        self.sigma_guess_offset = 0.59  # keV
        self.sigma_guess_slope = 0.00015  # keV
        self.window_factor = 20
        self.fit_values = {}
        self.n_retries = 5
        self.expected_offset_factor = 1
        self.name = None
        self.A0 = None
        self.A1 = None

    def get_sigma_guess(self, x):
        return self.sigma_guess_offset + self.sigma_guess_slope * x

    def fit_peaks(self, spec: SpectrumData):
        self.A0 = spec.A0
        self.A1 = spec.A1
        self.group_peaks(spec)
        for peaks in self.peak_groups:
            self.fit_multiple(spec, peaks)

    def group_peaks(self, spec):
        peak_groups = [[]]
        if len(self.expected_peaks) == 1:
            self.peak_groups = [[self.expected_peaks[0]]]
            return
        grouped = [False]*len(self.expected_peaks)
        current_group = 0
        for i, peak_x in enumerate(self.expected_peaks[0:-1]):
            if grouped[i]:
                continue
            sigma_guess = self.get_sigma_guess(peak_x)
            peak_guess = spec.data_at_x(peak_x)
            num_samples = int(round(self.window_factor * sigma_guess / spec.A1))
            start_ind = peak_guess - int(floor(num_samples / 2))
            stop_ind = peak_guess + int(floor(num_samples / 2)) + 1
            peak_groups[current_group].append(peak_x)
            for j, peak_x_2 in enumerate(self.expected_peaks[i+1:]):
                sigma_guess_2 = self.get_sigma_guess(peak_x_2)
                peak_guess_2 = spec.data_at_x(peak_x_2)
                num_samples_2 = int(round(self.window_factor * sigma_guess_2 / spec.A1))
                start_ind_2 = peak_guess_2 - int(floor(num_samples_2 / 2))
                stop_ind_2 = peak_guess_2 + int(floor(num_samples_2 / 2)) + 1
                if (start_ind_2 < stop_ind and start_ind_2 > start_ind) or (stop_ind_2 > start_ind and stop_ind_2 < stop_ind): #overlap detected
                    peak_groups[current_group].append(peak_x_2)
                    grouped[i+j+1] = True
                    if start_ind_2 < start_ind:
                        start_ind = start_ind_2
                    if stop_ind_2 > stop_ind:
                        stop_ind = stop_ind_2
            current_group += 1
            peak_groups.append([])
        if not grouped[-1]:
            if not peak_groups[-1]:
                peak_groups[-1] = [self.expected_peaks[-1]]
        else:
            if not peak_groups[-1]:
                del peak_groups[-1]
        self.peak_groups = peak_groups
        print(self.peak_groups)

    def smallest_separation(self, peaks):
        diffs = []
        for i in range(len(peaks)-1):
            for j in range(i+1, len(peaks)):
                diffs.append(abs(peaks[j] - peaks[i]))
        diffs.sort()
        return int(floor(diffs[0]/self.A1))

    def match_peak_locs(self, maxlocs, peaks):
        """assumes peaks are sorted, removes locs from maxlocs that dont match"""
        if len(maxlocs) <= len(peaks):
            return maxlocs
        peak_diffs_list = []
        max_diffs_list = []
        ndiff = len(maxlocs) - len(peaks)
        for i in range(len(peaks)-1):
            peak_diffs = []
            for j in range(i+1, len(peaks)):
                peak_diffs.append((peaks[j] - peaks[i])/self.A1)
            peak_diffs_list.append(np.array(peak_diffs))
        for i in range(len(maxlocs) - 1):
            max_diffs = []
            for j in range(i+1, len(maxlocs)):
                max_diffs.append(maxlocs[j] - maxlocs[i])
            max_diffs_list.append(np.array(max_diffs))
        cur_match = 0
        while len(maxlocs) > len(peaks):
            best_match_ind = 0
            best_match_val = np.inf
            for i in range(cur_match, ndiff + 1):
                #look at all possible combinations
                #print("current match is {0}, ndiff is {1}".format(cur_match,ndiff))
                #print("max diffs list length is {0} index is {1}".format(len(max_diffs_list),i))
                indices = list(combinations(range(len(max_diffs_list[i])), len(peak_diffs_list[cur_match])))
                if not indices:
                    continue
                tots = np.zeros((len(indices),))
                for j, inds in enumerate(indices):
                    inds = list(inds)
                    tots[j] = np.sum(np.abs(max_diffs_list[i][inds] - peak_diffs_list[cur_match]))
                best_match = np.amin(tots)
                if best_match < best_match_val:
                    best_match_val = best_match
                    best_match_ind = i
            inds_to_del = []
            for i in range(cur_match, ndiff + 1):
                if i != best_match_ind:
                    inds_to_del.append(i)
            j = 0
            for delind in inds_to_del:
                maxlocs = np.delete(maxlocs,delind-j)
                del max_diffs_list[delind - j]
                j += 1
            ndiff = len(maxlocs) - len(peaks)
            cur_match += 1
        return maxlocs



    def fit_multiple(self, spec: SpectrumData, peaks):
        if len(peaks) == 1:
            self.fit(spec, peaks[0])
        else:
            peaks.sort()
            sigma_guesses = []
            centroid_guesses = [] #indice of centroid
            maxlocs = np.full((len(peaks)*3,),-1)
            smallest_sep = self.smallest_separation(peaks)
            for peak in peaks:
                sigma_guesses.append(self.get_sigma_guess(peak))
                centroid_guesses.append( spec.data_at_x(peak * self.expected_offset_factor))
            num_samples_first = int(round(self.window_factor * sigma_guesses[0] / spec.A1))
            start_ind = centroid_guesses[0] - int(floor(num_samples_first / 2))
            num_samples_last = int(round(self.window_factor * sigma_guesses[-1] / spec.A1))
            stop_ind = centroid_guesses[-1] + int(floor(num_samples_last / 2)) + 1
            #find_peaks(spec.data[start_ind:stop_ind], maxlocs, smallest_sep)
            #maxlocs = maxlocs + start_ind
            #maxlocs.sort()
            #maxlocs = np.unique(maxlocs)
            #maxlocs = self.match_peak_locs(maxlocs, peaks)
            #centroid_guesses = []
            #for peak in maxlocs:
            #    centroid_guesses.append(peak)
            #start_ind = centroid_guesses[0] - int(floor(num_samples_first / 2))
            #stop_ind = centroid_guesses[-1] + int(floor(num_samples_last / 2)) + 1
            xs = spec.get_data_x()[start_ind:stop_ind]
            bkg_guess = average_median(spec.data[start_ind:start_ind+10])
            bkg_guess2 = average_median(spec.data[stop_ind-10:stop_ind])
            B_guess = (bkg_guess2 - bkg_guess) / (self.A1 * (stop_ind - start_ind))
            A_guess = bkg_guess - B_guess * start_ind * self.A1
            # guesses = [max_val - A_guess - B_guess*centroid_guess, self.R, self.step, centroid_guess, sigma_guess,
            guess_list = []
            lower_bound_list = []
            upper_bound_list = []
            for i in range(len(peaks)):
                peak_val = spec.data[centroid_guesses[i]]
                centroid_guess = spec.A0 + spec.A1*(centroid_guesses[i] + 1)
                beta_guess = sigma_guesses[i]
                guesses = [peak_val - A_guess - B_guess * centroid_guess, self.R, centroid_guess, sigma_guesses[i],
                           beta_guess]
                lower_bounds = [guesses[0] * self.low_bound_gaus_scale, 0,
                                centroid_guess * 0.98, sigma_guesses[i] * 0.3, beta_guess * 0.1]
                upper_bounds = [guesses[0] * self.up_bound_gaus_scale, 1,
                                centroid_guess * 1.02, sigma_guesses[i] * 3.0, beta_guess * 10.0]
                if guesses[0] < 0:
                    guesses[0] = peak_val/2.
                    lower_bounds[0] = 0
                    upper_bounds[0] = peak_val
                guess_list.append(guesses)
                lower_bound_list.append(lower_bounds)
                upper_bound_list.append(upper_bounds)
            ys = spec.data[start_ind:stop_ind]
            sigma = np.sqrt(ys)
            sigma[sigma == 0] = 1
            guesses = []
            lower_bounds = []
            upper_bounds = []
            for g, l, u in zip(guess_list, lower_bound_list, upper_bound_list):
                for guess in g:
                    guesses.append(guess)
                for low in l:
                    lower_bounds.append(low)
                for up in u:
                    upper_bounds.append(up)
            guesses.append(A_guess)
            guesses.append(B_guess)
            guesses.append(0.)
            lower_bounds.append(-np.inf)
            lower_bounds.append(-np.inf)
            lower_bounds.append(-np.inf)
            upper_bounds.append(np.inf)
            upper_bounds.append(np.inf)
            upper_bounds.append(np.inf)
            parameters, covariance = curve_fit(ge_multi_peak_function, xs, ys, p0=guesses,  # sigma=sigma,
                                               bounds=(lower_bounds, upper_bounds), sigma=sigma,
                                               #absolute_sigma=True,  maxfev=100000,
                                               absolute_sigma=True, jac=ge_multi_peak_function_jac, maxfev=100000)
            errs = np.sqrt(np.diag(covariance))

            peak_str = [str(peak) for peak in peaks]
            peak_str = ','.join(peak_str)
            self.fit_values[peak_str] = MultiPeakFit(parameters, errs, covariance, xs, ys)

    def fit(self, spec: SpectrumData, peak_x):
        sigma_guess = self.get_sigma_guess(peak_x)
        peak_guess = spec.data_at_x(peak_x * self.expected_offset_factor)
        if self.expected_offset_factor == 1:
            #use a larger window at first to get the expected offset
            num_samples = int(round(3*self.window_factor * sigma_guess / spec.A1))
        else:
            num_samples = int(round(self.window_factor * sigma_guess / spec.A1))
        start_ind = peak_guess - int(floor(num_samples / 2))
        stop_ind = peak_guess + int(floor(num_samples / 2)) + 1
        max_val = np.amax(spec.data[start_ind:stop_ind])
        indice = np.where(spec.data[start_ind:stop_ind] == max_val)[0]
        if indice.shape[0] > 1:
            max_ind = int(round(np.average(indice) + start_ind))
        else:
            max_ind = start_ind + indice[0]
        if self.expected_offset_factor == 1:
            self.expected_offset_factor = max_ind / peak_guess
            print("setting expected offset factor to {} for subsequent peak searches".format(self.expected_offset_factor))
        num_samples = int(round(self.window_factor * sigma_guess / spec.A1))
        start_ind = max_ind - int(floor(num_samples / 2))
        stop_ind = max_ind + int(floor(num_samples / 2)) + 1
        xs = spec.get_data_x()[start_ind:stop_ind]
        centroid_guess = (max_ind + 1) * spec.A1 + spec.A0
        beta_guess = sigma_guess
        bkg_guess = average_median(spec.data[start_ind-10:start_ind])
        if stop_ind + 10 > len(spec.data):
            bkg_guess2 = average_median(spec.data[-5:])
        else:
            bkg_guess2 = average_median(spec.data[stop_ind:stop_ind+10])
        B_guess = (bkg_guess2 - bkg_guess) / (self.A1*(stop_ind - start_ind))
        A_guess = bkg_guess - B_guess*(start_ind*self.A1 + self.A0)
        #guesses = [max_val - A_guess - B_guess*centroid_guess, self.R, self.step, centroid_guess, sigma_guess,
        guesses = [max_val - A_guess - B_guess*centroid_guess, self.R, centroid_guess, sigma_guess,
                    beta_guess, A_guess, B_guess, 0]
        lower_bounds = [guesses[0]*self.low_bound_gaus_scale, 0,
                        centroid_guess*0.98, sigma_guess*0.3, beta_guess*0.1, -np.inf, -np.inf, -np.inf]
        upper_bounds = [guesses[0]*self.up_bound_gaus_scale, 1,
                        centroid_guess*1.02, sigma_guess*3.0, beta_guess*10.0, np.inf, np.inf, np.inf]
        ys = spec.data[start_ind:stop_ind]
        sigma = np.sqrt(ys)
        sigma[sigma == 0] = 1
        parameters, covariance = curve_fit(ge_peak_function, xs, ys, p0=guesses, #sigma=sigma,
                                           bounds=(lower_bounds, upper_bounds), sigma=sigma,
                                           absolute_sigma=True, jac=ge_peak_function_jac, maxfev=100000)
        errs = np.sqrt(np.diag(covariance))
        self.fit_values[peak_x] = PeakFit(parameters, errs, covariance, xs, ys)

    def get_plot_name(self, peak_x, outdir):
        if self.name is None:
            return join(outdir, "spectrum_{}_fit".format(peak_x))
        else:
            return join(outdir, "{0}_{1}_fit".format(self.name,peak_x))

    def plot_fits(self, user_prompt=False, write_to_dir=None):
        accept = [True]*len(self.expected_peaks)
        if not user_prompt and write_to_dir is None:
            print("nothing to do")
            return
        i = 0
        for peak_x, fit in self.fit_values.items():
            figname = None
            if write_to_dir is not None:
                figname = self.get_plot_name(peak_x, write_to_dir)
            fit.plot(figname)
            if user_prompt:
                answer = None
                retried = 0
                while(answer is None):
                    print("displying fit for peak {} keV".format(peak_x))
                    fit.display()
                    if retried == 0:
                        plt.show(block=False)
                        plt.gcf().canvas.draw_idle()
                        plt.gcf().canvas.start_event_loop(0.3)
                        plt.close()
                    answer = str(input("Accept fit? [y/n] (hit enter for default = y)") or "y")
                    if answer.lower() in ["y", "yes"]:
                        print("accepting fit.")
                        accept[i] = True
                        break
                    elif answer.lower() in["n", "no"]:
                        print("rejecting fit.")
                        accept[i] = False
                        break
                    else:
                        answer = None
                        if(retried >= self.n_retries):
                            print("retry limit reached, rejecting fit.")
                            accept[i] = False
                            break
                        print("input must be yes (y) or no (n), retrying prompt..")
                        retried += 1
            i += 1
            plt.close()
        return accept

    def get_index_from_energy(self, e):
        return (e - self.A0) / self.A1

    def retrieve_calibration(self, user_prompt=False, tolerate_fails=False, write_to_dir=None, plot_fit=False):
        accept = self.plot_fits(user_prompt, write_to_dir)
        use_fits = True
        if tolerate_fails:
            n_fails = 0
            for b in accept:
                if not b:
                    n_fails += 1
            if n_fails > (len(accept) - 2):
                use_fits = False
        else:
            use_fits = False not in accept
        if not use_fits:
            print("not enough fits passed acceptance criteria, calibration failed.")
            return
        i = 0
        centroids = []
        sigmas = []
        values = []
        for peak_x, fit in self.fit_values.items():
            if not isinstance(peak_x, str):
                if not accept[i]:
                    i+=1
                    continue
                centroids.append(self.get_index_from_energy(fit.centroid))
                sigmas.append(fit.sigma)
                values.append(peak_x)
            else:
                peak_x = peak_x.split(',')
                for p, centroid, sigma in zip(peak_x, fit.centroids, fit.sigmas):
                    centroids.append(centroid)
                    sigmas.append(sigma)
                    values.append(float(p))
            i += 1
        sigmas = np.array(sigmas)
        values = np.array(values)
        centroids = np.array(centroids)
        if plot_fit:
            if self.name:
                fit_name = "{0}_calibration_fit".format(self.name)
            else:
                fit_name = "spectrum_calibration_fit"
            if write_to_dir is not None:
                fit_name = join(write_to_dir, fit_name)
            coeff, errs = linfit_to_calibration(*linfit(values, centroids, sigmas, plot=fit_name))
        else:
            coeff, errs = linfit_to_calibration(*linfit(values, centroids, sigmas))
        print("old coefficients were A0 = {0} A1 = {1}".format(self.A0, self.A1))
        print("new coefficients are A0 = {0} ~ {1} A1 = {2} ~ {3}".format(coeff[0], errs[0], coeff[1], errs[1]))
        return coeff, errs

