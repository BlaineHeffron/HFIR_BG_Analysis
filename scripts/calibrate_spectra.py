import sys
from os.path import dirname, realpath
sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.database.HFIRBG_DB import HFIRBG_DB
from src.utilities.util import get_data_dir, populate_data, populate_data_config, calibrate_spectra, write_root_with_db, \
    calibrate_nearby_runs, retrieve_data
from os.path import join, exists
import os

config = None
#config = {"min_time": 80000, "acquisition_settings": {"coarse_gain": 2, "fine_gain": 1.02}}
rundata = {"Reactor Spectrum": "MIF_BOX_REACTOR_OPTIMIZED_OVERNIGHT_LOWEST_GAIN.txt"}
#rundata = {"Reactor Spectrum": "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN.txt"}
#rundata = {"MIF": "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN.txt","HB4":"HB4_DOWN_OVERNIGHT_1.txt"}
#rundata = {"rxoff":"MIF_BOX_AT_REACTOR_RXOFF"}
#rundata = {"EAST_FACE_1": "EAST_FACE_1.txt"}
expected_peaks = [11386.5, 8884.81, 9102.1, 9297.8, 8998.63, 7724.034, 6809.61, 1460.8, 1332.5, 1274.43, 1173.2, 511.0,
                  964.082, 1112.08, 1408.013, 778.9006, 1293.64]

#expected_peaks = [7724.034, 7645.58, 7631.18, 1460.8, 1332.5, 1293.64, 1173.2, 511.0]
#expected_peaks = [1460.8, 1332.5, 1274.43, 1173.2,964.082,1112.08,1408.013,778.9006]
update_nearby_runs = True
dt = 86400*2


def main():
    datadir = get_data_dir()
    db = HFIRBG_DB()
    if config:
        data = populate_data_config(config, db, comb_runs=True)
    else:
        data = populate_data(rundata, datadir, db)
    outdir = join(os.environ["HFIRBG_ANALYSIS"], "calibration")
    nearby_groups = set()
    if not exists(outdir):
        os.mkdir(outdir)
    if update_nearby_runs:
        nearby_groups = calibrate_nearby_runs(data, expected_peaks, db, outdir, True, True, True, dt)
    else:
        calibrate_spectra(data, expected_peaks, db, outdir, True, True, True)
    print("writing root files of newly calibrated spectra")
    if config:
        data = populate_data_config(config, db, comb_runs=False)
    else:
        data = populate_data(rundata, datadir, db)
    for name, spec in data.items():
        write_root_with_db(spec, name, db)
    if update_nearby_runs:
        for group_id in nearby_groups:
            files = db.retrieve_file_paths_from_calibration_group_id(group_id)
            for f in files:
                update_spec = retrieve_data(f, db)
                write_root_with_db(update_spec, update_spec.fname, db)


if __name__ == "__main__":
    main()
