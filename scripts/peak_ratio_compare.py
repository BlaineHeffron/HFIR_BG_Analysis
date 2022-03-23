import sys
from os.path import dirname, realpath
sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.database.SqliteManager import HFIRBG_DB
from src.utilities.util import get_data_dir, populate_data,  populate_data_root, \
    compare_peaks
from os.path import join, exists
import os

rundata = {"Reactor Spectrum": "MIF_BOX_REACTOR_OPTIMIZED_OVERNIGHT_LOWEST_GAIN.txt"}
rootdir = join(os.environ["HFIRBG_SIM"], "analysis/collimated")
energies = ["11386.5","9718.79","8998.63","7724.034","7693.398","7645.58","7631.18"]
outdir = join(os.environ["HFIRBG_ANALYSIS"], "peak_ratio_compare/third_root")
verify = False


def main():
    datadir = get_data_dir()
    all_energies = []
    for e in energies:
        all_energies.append(float(e))
        all_energies.append(float(e)-511)
        all_energies.append(float(e) - 2*511)
    db = HFIRBG_DB()
    data = populate_data(rundata, datadir,db)
    root_data_dict_false = {"{0}_false".format(e): join(rootdir, "coll_false_{0}.root".format(e)) for e in energies}
    root_data_dict_true = {"{0}_true".format(e): join(rootdir, "coll_true_{0}.root".format(e)) for e in energies}
    simdata = populate_data_root(root_data_dict_false, "GeEfficiencyPlugin/hGeEnergy", "accumulated/runtime", True, 1000., 10)
    simdata.update(populate_data_root(root_data_dict_true,  "GeEfficiencyPlugin/hGeEnergy", "accumulated/runtime", True, 1000.))
    if not exists(outdir):
        os.mkdir(outdir)
    compare_peaks(data, simdata, all_energies, outdir, verify, True, True)



if __name__ == "__main__":
    main()
