import sys
from os.path import dirname, realpath
sys.path.insert(1, dirname(dirname(realpath(__file__))))
from os.path import join
from argparse import ArgumentParser

from src.utilities.ROOT_style import ROOT_style
import os
from src.utilities.util import get_spec_from_root, write_x_y_csv
from src.analysis.Spectrum import SpectrumFitter
from src.utilities.PlotUtils import  MultiLinePlot
import matplotlib.pyplot as plt

outdir = join(os.environ["HFIRBG_ANALYSIS"], "sim_efficiencies")

def main():
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    arg = ArgumentParser()
    ROOT_style()
    ens = [i for i in range(40,11386)]
    arg.add_argument("dir",help="path to directory containing simulated histograms", type=str)
    arg.add_argument("--collimated", "-c", action="store_true", help="if collimated")
    args = arg.parse_args()
    name = "coll_{0}_front_{1}.root"

    fitter = SpectrumFitter()
    coll = "false"
    if args.collimated:
        coll = "true"
    areas = []
    dareas = []
    for e in ens:
        rootfile = join(args.dir, name.format(coll, e))
        hist_en = get_spec_from_root(rootfile,  "GeEfficiencyPlugin/hGeEnergy", "accumulated/runtime", True, 1000., 1)
        fitter.expected_peaks = [e]
        fitter.fit_peaks(hist_en)
        a, da = fitter.fit_values[e].area()
        areas.append(a/hist_en.live)
        dareas.append(da/hist_en.live)
        fitter.fit_values = {}

    write_x_y_csv(join(outdir, "sim_efficiencies.csv"), "energy [keV]", "peak area", "peak area uncertainty", ens, areas, dareas)
    fig = MultiLinePlot(ens, [areas], [""], "energy [keV]", "peak area [hz]")
    plt.savefig(join(outdir, "sim_efficiencies.png"))


if __name__ == "__main__":
    main()
