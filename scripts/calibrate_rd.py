import sys
from os.path import dirname, realpath
sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.database.HFIRBG_DB import HFIRBG_DB
from src.utilities.util import get_data_dir, populate_data, populate_data_config, calibrate_spectra, write_root_with_db, \
    calibrate_nearby_runs, retrieve_data, populate_rd_data, combine_runs, populate_data_db
from os.path import join, exists
import os

config = None
#config = {"min_time": 80000, "acquisition_settings": {"coarse_gain": 2, "fine_gain": 1.02}}
#rundata = {"Reactor Spectrum": "MIF_BOX_REACTOR_OPTIMIZED_OVERNIGHT_LOWEST_GAIN.txt"}
#rundata = {"MIF": "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN.txt","HB4":"HB4_DOWN_OVERNIGHT_1.txt"}
#rundata = {"rxoff":"MIF_BOX_AT_REACTOR_RXOFF"}
rundata = {"EAST_FACE_1": "EAST_FACE_1.txt"}
#expected_peaks = [11386.5, 8884.81, 9102.1, 9297.8, 8998.63, 7724.034, 6809.61, 1460.8, 1332.5, 1274.43, 1173.2, 511.0,
#                  964.082, 1112.08, 1408.013, 778.9006, 1293.64]

#expected_peaks = [7724.034, 7645.58, 7631.18, 1460.8, 1332.5, 1293.64, 1173.2, 511.0]
#expected_peaks = [7724.034, 7645.58, 7631.18, 1460.8, 1332.5, 1293.64, 1173.2, 511.0]
#expected_peaks = [1293.64, 511.0, 374.72, 768.36, 1120.29, 1238.11, 1377.67]
expected_peaks = [1293.64, 511.0, 374.72, 1120.29]
#expected_peaks = [7724.034, 7645.58, 7631.18, 1460.8, 1332.5, 1293.64, 1173.2, 511.0, 374.72, 768.36, 1120.29, 1238.11, 1377.67]
#expected_peaks = [1460.8, 1332.5, 1274.43, 1173.2,964.082,1112.08,1408.013,778.9006]
dt = 86400*4

outdir = join(os.environ["HFIRBG_ANALYSIS"], "russian_doll")


def main():
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    db = HFIRBG_DB()
    rd_data = db.get_rd_files(True, gain_setting=['low'])
    rd_data = populate_data_db(rd_data, db)
    combine_runs(rd_data, max_interval=dt)
    if not exists(outdir):
        os.mkdir(outdir)
    for run in rd_data.keys():
        for i, spec in enumerate(rd_data[run]):
            calibrate_spectra({"{0}_interval_{1}".format(run, i): spec}, expected_peaks, db, outdir, True, True, True)


if __name__ == "__main__":
    main()
