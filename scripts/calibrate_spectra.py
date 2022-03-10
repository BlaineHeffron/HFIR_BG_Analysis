import sys
from os.path import dirname, realpath
sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.database.SqliteManager import HFIRBG_DB
from src.utilities.util import get_data_dir, populate_data, calibrate_spectra
from os.path import join, exists
import os

rundata = {"Reactor Spectrum":"MIF_BOX_REACTOR_OPTIMIZED_OVERNIGHT_LOWEST_GAIN.txt"}
expected_peaks = [11386.5, 9718.79,8884.81,9656.89, 9102.1, 9297.8, 8998.63, 7938.58, 7724.034,7645.58,7631.18, 1460.8, 1332.5,1173.2,511.0]

def main():
    datadir = get_data_dir()
    data = populate_data(rundata, datadir)
    db = HFIRBG_DB()
    outdir = join(os.environ["HFIRBG_ANALYSIS"], "calibration")
    if not exists(outdir):
        os.mkdir(outdir)
    calibrate_spectra(data, expected_peaks,db, outdir, True, False, True)

if __name__ == "__main__":
    main()