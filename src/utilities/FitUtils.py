import numpy as np
import matplotlib.pyplot as plt
from src.utilities.PlotUtils import ScatterLinePlot
from math import sqrt
from scipy.optimize import curve_fit, minimize


def linfit(x, y, sigma, plot=None, xlabel="channel #", ylabel="Peak Energy [keV]"):
    # print("fitting x values {0} y values {1} with sigma {2}".format(x,y,sigma))
    coeff, cov = np.polyfit(x, y, 1, cov=True, w=(1 / sigma))
    xmin = min(x) * .9
    xmax = max(x) * 1.01
    if plot is not None:
        linex = np.linspace(xmin, xmax, 100)
        liney = np.array([coeff[0] * i + coeff[1] for i in linex])
        fit_y_errs = np.array([sqrt(cov[1, 1] + i ** 2 * cov[0, 0] + 2 * i * cov[0, 1]) for i in linex])
        ScatterLinePlot(x, y, sigma, linex, liney, fit_y_errs,
                        ["best fit", r'1 $\sigma$ error', "data"], xlabel, ylabel,
                        ylog=False, legend_loc='best', xmin=xmin, xmax=xmax)
        plt.tight_layout()
        plt.savefig(plot + ".png")
        plt.close()
    fity = lin_func(x, *coeff)
    chis = chisqr(y, fity, sigma)
    sig_err = ave_sig_error(y, sigma, fity)
    print("average sigma error of linear fit is {}".format(sig_err))
    return coeff, cov, chis, sig_err


def lin_func(x, A, B):
    return A * x + B


def chisqr(obs, fit, sigma):
    return np.sum(np.divide(np.square(obs - fit), np.square(sigma)))


def ave_sig_error(o, sigma, e):
    return np.average(np.divide(np.abs(o - e), sigma))


def linfit_to_calibration(coeff, cov):
    errs = np.sqrt(np.diag(cov))
    A0 = -coeff[1] / coeff[0]
    A1 = 1 / coeff[0]
    dA0 = abs(A0) * sqrt((errs[1] / coeff[1]) ** 2 + (errs[0] / coeff[0]) ** 2 + 2 * cov[0, 1] / (coeff[0] * coeff[1]))
    dA1 = abs(errs[1] * A1 ** 2)
    return (A0, A1), (dA0, dA1)


def sqrt_func(x, A, B, C):
    return np.sqrt(x) * B + A + x * C


def sqrt_func_jac(x, A, B, C):
    dA = np.ones((len(x),))
    dB = np.sqrt(x)
    dC = x
    return np.hstack((dA.reshape(-1, 1), dB.reshape(-1, 1), dC.reshape(-1, 1)))


def sqrtfit(x, y, sigma, plot=None, xlabel="channel #", ylabel="Peak Energy [eV]"):
    parameters, covariance = curve_fit(sqrt_func, x, y, sigma=sigma, absolute_sigma=False, maxfev=100000,
                                       jac=sqrt_func_jac)
    if plot is not None:
        linex = np.linspace(min(x), max(x), 100)
        liney = sqrt_func(linex, *parameters)
        jac = sqrt_func_jac(linex, *parameters)
        lineerr = np.sqrt(np.diag(np.matmul(jac, np.matmul(covariance, jac.T))))
        ScatterLinePlot(x, y * 1000, sigma * 1000, linex, liney * 1000, lineerr * 1000,
                        ["best fit", r'1 $\sigma$ error', "data"], xlabel, ylabel,
                        ylog=False, legend_loc='best')
        plt.savefig(plot + ".png")
        plt.close()
    fity = sqrt_func(x, *parameters)
    chis = chisqr(y, fity, sigma)
    print("average sigma error of sqrt fit is {}".format(ave_sig_error(y, sigma, fity)))
    return parameters, covariance, chis


def minimize_diff(data, ref, ref_errs):
    """Returns scale factor to minimize chisqr diff between data and ref"""

    def scale_min_chisqr(x):
        return chisqr(data * x, ref, ref_errs)

    initial_guess = np.array([np.sum(ref) / np.sum(data)])
    res = minimize(scale_min_chisqr, initial_guess, method='Nelder-Mead')
    if res.success:
        return res.x[0]
    else:
        raise RuntimeError(res.message)
