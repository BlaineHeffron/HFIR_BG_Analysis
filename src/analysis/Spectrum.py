from copy import copy

import numba as nb
import numpy as np
from ROOT import TH1F
from math import sqrt, erfc, exp, floor

from scipy.optimize import curve_fit


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


def ge_peak_function(x, c1, c2, c3, centroid, sigma, beta):
    """
    modelled after https://radware.phy.ornl.gov/gf3/
    """
    gaussian = c1 * exp(-0.5 * ((x - centroid) / sigma) ** 2)
    skew_gaussian = c2 * exp((x - centroid) / beta) * erfc(
        (x - centroid) / (sqrt(2) * sigma) + sigma / (sqrt(2) * beta))
    background = c3 * erfc((x - centroid) / (sqrt(2) * sigma))
    return gaussian + skew_gaussian + background


class SpectrumFitter:
    def __init__(self, expected_peaks=None):
        # 2.26 keV sigma for 11386 keV peak
        # 0.79 keV at 1332.5 keV peak
        # using a simple linear fit to estimate the window size we get sigma = .5918 + 0.0001465*x
        self.expected_peaks = expected_peaks
        self.R = 0.1
        self.step = 0.01
        self.sigma_guess_offset = 0.59  # keV
        self.sigma_guess_slope = 0.0001465  # keV
        self.window_factor = 8
        self.fit_values = {}

    def get_sigma_guess(self, x):
        return self.sigma_guess_offset + self.sigma_guess_slope * x

    def fit_peaks(self, spec):
        for peak_x in self.expected_peaks:
            self.fit(spec, peak_x)

    def fit(self, spec: SpectrumData, peak_x):
        peak_guess = spec.data_at_x(peak_x)
        sigma_guess = self.get_sigma_guess(peak_x)
        num_samples = int(round(self.window_factor * sigma_guess / spec.A1))
        start_ind = peak_guess - int(floor(num_samples / 2))
        stop_ind = peak_guess + int(floor(num_samples / 2)) + 1
        max_val = np.amax(spec.data[start_ind:stop_ind])
        indice = np.where(spec.data[start_ind:stop_ind] == max_val)
        if indice.shape[0] > 1:
            max_ind = int(round(np.average(indice) + start_ind))
        else:
            max_ind = start_ind + indice[0]
        centroid_guess = (max_ind + 1) * spec.A1 + spec.A0
        beta_guess = sigma_guess / 2.
        guesses = [max_val * (1 - self.R), max_val * self.R, max_val * self.step, centroid_guess, sigma_guess,
                   beta_guess]
        parameters, covariance = curve_fit(ge_peak_function, spec.get_data_x(), spec.data, p0=guesses,
                                           sigma=np.sqrt(spec.data))
        errs = np.sqrt(np.diag(covariance))
        print("fit parameters for x = {0}:".format(peak_x))
        print("guassian scale: {0} ~ {1} counts".format(parameters[0], errs[0]))
        print("skew guassian scale: {0} ~ {1} counts".format(parameters[1], errs[1]))
        print("background scale: {0} ~ {1} counts".format(parameters[2], errs[2]))
        print("centroid: {0} ~ {1} keV".format(parameters[3], errs[3]))
        print("std deviation: {0} ~ {1} keV".format(parameters[4], errs[4]))
        print("skewness: {0} ~ {1} keV".format(parameters[5], errs[5]))
        self.fit_values[peak_x] = (parameters, errs)
