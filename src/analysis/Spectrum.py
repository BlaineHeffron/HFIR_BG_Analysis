from copy import copy

import numpy as np

from src.utilities.util import rebin_data, numba_rebin


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
            self.bin_midpoints = np.zeros((self.bin_edges.shape[0] - 1,))
            for j in range(0, self.bin_edges.shape[0] - 1):
                self.bin_midpoints[j] = self.bin_edges[j] + (self.bin_edges[j + 1] - self.bin_edges[j]) / 2.

    def rebin_factor(self, factor):
        if factor <= 1:
            raise RuntimeError("rebin factor must be > 1")
        new_hist = rebin_data(self.hist[1:-1], factor)
        under = self.hist[0]
        over = self.hist[-1]
        self.hist = np.zeros((new_hist.shape[0] + 2,))
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

    def rebin(self, bin_edges=None):
        set_to_data = False
        if bin_edges is None:
            set_to_data = True
            bin_edges = np.zeros(self.data.shape[0] + 1)
            for i in range(self.data.shape[0] + 1):
                bin_edges[i] = self.A0 + self.A1 * i + self.A1 / 2.
        self.hist = np.zeros((bin_edges.shape[0] + 1,))
        self.bin_edges = bin_edges
        self.calculate_bin_midpoints()
        self.nbins = bin_edges.shape[0] - 1
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