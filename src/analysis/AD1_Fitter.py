from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from copy import copy

from src.analysis.Spectrum import gauss, dgaussdc, dgaussdR, dgaussds
from scipy.special import erfc
import numpy as np
from src.utilities.FitUtils import chisqr
from src.utilities.PlotUtils import ScatterLinePlot


def AD1_gamma_fit(x, Hg, He, c, s, c2, s2, A, B):
    return gauss(x, Hg, 0, c, s) + He*erfc((x-c2)/s2) + A + B * x

def AD1_gamma_fit_jac(x, Hg, He, c, s, c2, s2, A, B):
    dHg = gauss(x, 1, 0, c, s)
    dHe = erfc((x-c2)/s2)
    dcentroid = dgaussdc(x, Hg, 0, c, s)
    dsigma = dgaussds(x, Hg, 0, c, s)
    dc2 = 2*He*np.exp(-((x-c2)/s2)**2)/(np.sqrt(np.pi)*s2)
    ds2 = 2*He*(x-c2)*np.exp(-((x-c2)/s2)**2)/(s2**2*np.sqrt(np.pi))
    dA = np.ones(len(x))
    dB = x
    return np.hstack((dHg.reshape(-1, 1), dHe.reshape(-1, 1), dcentroid.reshape(-1, 1), dsigma.reshape(-1, 1),
                      dc2.reshape(-1, 1), ds2.reshape(-1, 1), dA.reshape(-1, 1), dB.reshape(-1, 1)))


def fit_spec(spec, e1, e2, plot_dir="", name="", use_hist=False):
    if use_hist:
        ind1 = spec.data_at_x(e1, use_hist=use_hist)
        ind2 = spec.data_at_x(e2, use_hist=use_hist)
        xs = spec.bin_midpoints
        if ind2 + 1 > len(xs):
            xs = xs[ind1:]
            ys = spec.hist[ind1:]
        else:
            xs = xs[ind1:ind2+1]
            ys = spec.hist[ind1:ind2+1]
    else:
        ind1 = spec.data_at_x(e1)
        ind2 = spec.data_at_x(e2)
        xs = spec.get_data_x()
        if ind2 + 1 > len(xs):
            xs = xs[ind1:]
            ys = spec.data[ind1:]
        else:
            xs = xs[ind1:ind2+1]
            ys = spec.data[ind1:ind2+1]
    sigma = np.sqrt(ys)
    for i in range(len(sigma)):
        if sigma[i] == 0:
            sigma[i] = 1
    guesses = [np.amax(ys), np.amax(ys)/2, xs[int(len(xs)/2)], (xs[-1]-xs[0])/4, xs[int(len(xs)/2)], (xs[-1] - xs[0])/4, 0, 0]
    lower_bounds = [0, 0, xs[0], (xs[-1]-xs[0])/20, xs[0], (xs[-1] - xs[0])/20, -np.inf, -np.inf]
    upper_bounds = [np.amax(ys), np.amax(ys), xs[-1], xs[-1]-xs[0], xs[-1], xs[-1] - xs[0], np.inf, np.inf ]
    parameters, covariance = curve_fit(AD1_gamma_fit, xs, ys, p0=guesses, sigma=sigma,
                                       bounds=(lower_bounds, upper_bounds),
                                       # absolute_sigma=True,  maxfev=100000,
                                       absolute_sigma=False, jac=AD1_gamma_fit_jac, maxfev=100000)
    errs = np.sqrt(np.diag(covariance))
    fity = AD1_gamma_fit(xs, *parameters)
    cs = chisqr(ys, fity, sigma)
    linex = np.linspace(xs[0], xs[-1], 100)
    liney = AD1_gamma_fit(linex, *parameters)
    jac = AD1_gamma_fit_jac(linex, *parameters)
    lineerr = np.sqrt(np.diag(np.matmul(jac, np.matmul(covariance, jac.T))))
    if plot_dir:
        miny = np.amin(ys)
        if miny == 0:
            nonzeros = copy(ys)
            nonzeros[nonzeros == 0] = 99999999999
            miny = np.amin(nonzeros)
        fig = ScatterLinePlot(xs, ys, sigma, linex, liney, lineerr,
                        ["best fit", r'1 $\sigma$ error', "data"], "energy [MeV]",
                        "counts", ylog=True, legend_loc='best', xmin=xs[0], xmax=xs[-1], ymin=miny)
        if name:
            plt.savefig(plot_dir + "/{2}_fit_{0}-{1}.png".format(e1,e2, name))
        else:
            plt.savefig(plot_dir + "/AD1_fit_{0}-{1}.png".format(e1, e2))
        plt.close()
    return parameters, errs, cs, len(ys) - 8
