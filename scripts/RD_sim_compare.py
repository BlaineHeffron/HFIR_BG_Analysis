import sys
import os
from os.path import dirname, realpath, join
from argparse import ArgumentParser


sys.path.insert(1, dirname(dirname(realpath(__file__))))

from src.utilities.util import get_spec_from_root, plot_multi_spectra, get_bins

outdir = join(join(os.environ["HFIRBG_ANALYSIS"], "russian_doll"), "sim")

emin = [1000 * i for i in range(12)]
emax = [1000 * (i + 1) for i in range(12)]
def main():
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    arg = ArgumentParser()
    arg.add_argument("basedir",help="path to directory containing GeRD_# directories", type=str)
    args = arg.parse_args()
    hist_dict = {"RD_" + str(i): None for i in range(6)}
    for i in range(6):
        mydir = join(args.basedir,"GeRD_{}".format(i))
        nm = "RD_" + str(i)
        for dirpath in os.listdir(mydir):
            rootfile = join(join(mydir, dirpath), "GeRD_{}.root".format(i))
            if os.path.exists(rootfile):
                hist_en = get_spec_from_root(rootfile,  "GeEnergyPlugin/hGeEnergy", "accumulated/runtime", True, 1000., 1)
                if not hist_dict[nm]:
                    hist_dict[nm] = hist_en
                else:
                    hist_dict[nm].add(hist_en)
    print(hist_dict)
    plot_name = join(outdir, "sim_compare")
    plot_multi_spectra(hist_dict, plot_name, rebin=100)
    for i in range(len(emin)):
        plot_multi_spectra(hist_dict, plot_name + "_{}".format(i), emin=emin[i], emax=emax[i], rebin=100)

if __name__ == "__main__":
    main()