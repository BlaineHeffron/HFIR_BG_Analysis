import numpy as np
import matplotlib.pyplot as plt
from src.utilities.PlotUtils import ScatterLinePlot
from math import sqrt

def linfit(x, y, sigma, plot=None, xlabel="channel #", ylabel="Peak Energy [keV]"):
    # print("fitting x values {0} y values {1} with sigma {2}".format(x,y,sigma))
    coeff, cov = np.polyfit(x, y, 1, cov=True, w=(1 / sigma))
    if plot is not None:
        linex = np.linspace(x[0], x[-1], 100)
        liney = np.array([coeff[0] * i + coeff[1] for i in linex])
        fit_y_errs = [sqrt(cov[1, 1] + i ** 2 * cov[0, 0] + 2 * i * cov[0, 1]) for i in linex]
        ScatterLinePlot(x, y, sigma, linex, liney, fit_y_errs,
                        ["best linear fit", r'1 $\sigma$ error', "peak fits to data"],xlabel, ylabel,
                        ylog=False, legend_loc='lower right')
        plt.savefig(plot + ".png")
        plt.close()
    return coeff, cov


def linfit_to_calibration(coeff, cov):
    errs = np.sqrt(np.diag(cov))
    A0 = -coeff[1] / coeff[0]
    A1 = 1 / coeff[0]
    dA0 = abs(A0) * sqrt((errs[1] / coeff[1]) ** 2 + (errs[0] / coeff[0]) ** 2 + 2 * cov[0, 1] / (coeff[0] * coeff[1]))
    dA1 = abs(errs[1] * A1 ** 2)
    return (A0, A1), (dA0, dA1)
