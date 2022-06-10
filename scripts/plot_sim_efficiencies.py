import sys
from math import pi
from os.path import dirname, realpath
sys.path.insert(1, dirname(dirname(realpath(__file__))))
from os.path import join
from argparse import ArgumentParser
import numpy as np

from src.utilities.ROOT_style import ROOT_style
import os
from src.utilities.util import get_spec_from_root, write_x_y_csv, read_rows_csv
from src.analysis.Spectrum import SpectrumFitter
from src.utilities.PlotUtils import  MultiLinePlot
import matplotlib.pyplot as plt
from src.utilities.PlotUtils import MultiLinePlot

outdir = join(os.environ["HFIRBG_ANALYSIS"], "sim_efficiencies")

def compare_sim_eff():
    through = join(outdir, "sim_efficiencies_true.csv")
    coll = join(outdir, "sim_efficiencies_false_none.csv")
    through_dat = read_rows_csv(through)
    coll_dat = read_rows_csv(coll)
    xs_through = []
    xs_coll = []
    ys_through = []
    ys_coll = []
    for data in through_dat[1:]:
        if float(data[1]) > 0:
            xs_through.append(float(data[0]))
            ys_through.append(float(data[1]))
    for data in coll_dat[1:]:
        if float(data[1]) > 0:
            xs_coll.append(float(data[0]))
            ys_coll.append(float(data[1]))
    print(len(ys_coll))
    print(len(ys_through))
    ys = [[],[]]
    xs = []
    j = 0
    for i in range(len(xs_through)):
        if xs_coll[j] != xs_through[i]:
            continue
        else:
            if ys_coll[j] > ys_through[i]:
                xs.append(xs_through[i])
                ys[0].append(ys_through[i])
                ys[1].append(ys_coll[j] - ys_through[i])
            j += 1
    lead_opening_r = 6.861653
    lead_z = 180.1+43.5
    leadR = 111
    SA_ratio = lead_opening_r*lead_opening_r*pi / (2*pi*leadR*leadR + 2*pi*leadR*lead_z - lead_opening_r*lead_opening_r*pi)
    MultiLinePlot(xs, ys, ["collimated", "uncollimated"], "energy [keV], ", "rate [hz/keV]")
    plt.savefig(join(outdir, "sim_eff_compare.png"))
    y = [f[0]/(f[1]*SA_ratio) for f in ys]
    MultiLinePlot(xs, [y], ["collimated/uncollimated scaled"], "energy [keV], ", "collimated / uncollimated")
    plt.savefig(join(outdir, "sim_eff_ratio_compare.png"))





def main():
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    arg = ArgumentParser()
    ROOT_style()
    ens = [i for i in range(40,11386)]
    arg.add_argument("dir",help="path to directory containing simulated histograms", type=str)
    arg.add_argument("--collimated", "-c", action="store_true", help="if collimated")
    arg.add_argument("--direction", "-d", type=str, help="direction if not none")
    args = arg.parse_args()
    name = "coll_{0}_{1}_{2}.root"
    direction = "none"
    if args.direction:
        direction = args.direction
    fitter = SpectrumFitter()
    coll = "false"
    if args.collimated:
        coll = "true"
    areas = []
    dareas = []
    for e in ens:
        rootfile = join(args.dir, name.format(coll, direction, e))
        hist_en = get_spec_from_root(rootfile,  "GeEfficiencyPlugin/hGeEnergy", "accumulated/runtime", True, 1000., 1)
        fitter.expected_peaks = [e]
        fitter.fit_peaks(hist_en)
        if e in fitter.fit_values.keys():
            a, da = fitter.fit_values[e].area()
            areas.append(a/hist_en.live)
            dareas.append(da/hist_en.live)
        else:
            areas.append(0)
            dareas.append(0)
        fitter.fit_values = {}

    write_x_y_csv(join(outdir, "sim_efficiencies_{0}_{1}.csv".format(coll, direction)), "energy [keV]", "peak area", "peak area uncertainty", ens, areas, dareas)
    fig = MultiLinePlot(ens, [areas], [""], "energy [keV]", "peak area [hz]")
    plt.savefig(join(outdir, "sim_efficiencies_{0}_{1}.png".format(coll, direction)))


if __name__ == "__main__":
    #main()
    compare_sim_eff()
