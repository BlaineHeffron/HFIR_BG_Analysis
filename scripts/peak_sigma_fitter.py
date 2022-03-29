import sys
from os.path import dirname, realpath

sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.database.SqliteManager import HFIRBG_DB
from src.utilities.util import get_data_dir, populate_data, fit_peak_sigmas
from os.path import join, exists
import os

#1293 is Ar-41 decay to K-41
#1274.5 is Na-22 decay to Ne-22
#1408.006 is Eu-152 decay
rundata = {"Reactor Spectrum": "MIF_BOX_REACTOR_OPTIMIZED_OVERNIGHT_LOWEST_GAIN.txt"}
energies = ["11386.5", "9718.79", "8998.63", "7724.034", "7693.398", "7645.58", "7631.18", "6809.61", "1274.5",
            "1293.64", "1408.006", "1460.8", "1332.5", "1173.2", "964.082","1112.08", "1085.841", "778.9006"]
#"511.0"]
outdir = join(os.environ["HFIRBG_ANALYSIS"], "peak_sigma_fits")
verify = False
use_sqrt_fit = True


def main():
    datadir = get_data_dir()
    all_energies = []
    for e in energies:
        all_energies.append(float(e))
        if float(e) > 2000:
            all_energies.append(float(e) - 511)
            all_energies.append(float(e) - 2 * 511)
    db = HFIRBG_DB()
    data = populate_data(rundata, datadir, db)
    if not exists(outdir):
        os.mkdir(outdir)
    fit_peak_sigmas(data, all_energies, outdir, verify, True, use_sqrt_fit=use_sqrt_fit)


if __name__ == "__main__":
    main()
