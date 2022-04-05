import sys
from os.path import dirname, realpath


sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.database.HFIRBG_DB import HFIRBG_DB
from src.database.CartScanFiles import CartScanFiles
import os
from src.utilities.util import retrieve_position_scans

def fix_corrupted_files(db):
    #corrupted datafiles have start time of -3506699038 and large live times
    HB3_DOWN_18_st = 1623789683
    HB3_DOWN_lts = {}
    SE_CORNER_lts = {}
    SE_lts = {}
    NE_1_lt = 0
    position_metadata = retrieve_position_scans() # r, l, angle, live, fname
    fifteen_min_run_start = 2859
    fifteen_min_run_end = 2927
    for data in position_metadata:
        if data[-1].startswith("HB3_DOWN_"):
            n = int(data[-1].split("_")[-1])
            HB3_DOWN_lts[n] = float(data[-2])
        elif data[-1].startswith("SE_CORNER_ANGLES"):
            n = int(data[-1].split("_")[-1])
            SE_CORNER_lts[n] = float(data[-2])
        elif data[-1].startswith("SE_"):
            n = int(data[-1].split("_")[-1])
            SE_lts[n] = float(data[-2])
        elif data[-1] == "NE_1":
            NE_1_lt = float(data[-2])
    fifteen_start_estimate = HB3_DOWN_18_st
    for i in range(18, 25):
        fifteen_start_estimate += HB3_DOWN_lts[i]
    fifteen_start_estimate += 7*60 + 15*60
    SE_corner_start_estimate = fifteen_start_estimate + 60*15*(fifteen_min_run_end - fifteen_min_run_start + 1) + 30*60
    SE_start_estimate = SE_corner_start_estimate
    for i in range(1, 5):
        SE_start_estimate += SE_CORNER_lts[i] + 60
    data = db.retrieve_file_paths_in_date_range(stop=-1)
    for path in data:
        fname = os.path.basename(path)
        if fname.endswith(".txt"):
            fname = fname[0:-4]
        if fname.startswith("HB3_DOWN_"):
            n = int(fname.split("_")[-1])
            start_time_offset = 0
            for i in range(18, n):
                start_time_offset += HB3_DOWN_lts[i]
            start_estimate = (n-18)*60 + start_time_offset + HB3_DOWN_18_st
            db.update_datafile_time(fname, start_estimate, HB3_DOWN_lts[n])
        elif fname.startswith("0000"):
            num = int(fname)
            if num == fifteen_min_run_end: #cant know the livetime of the last file since it was stopped manually
                continue
            num_offset = num - fifteen_min_run_start
            start_estimate = fifteen_start_estimate + num_offset*60*15
            db.update_datafile_time(fname, start_estimate, 15*60)
        elif fname.startswith("SE_CORNER_ANGLES"):
            n = int(fname.split("_")[-1])
            start_time_offset = 0
            for i in range(1, n):
                start_time_offset += SE_CORNER_lts[i]
            start_estimate = (n-1)*60 + start_time_offset + SE_corner_start_estimate
            db.update_datafile_time(fname, start_estimate, SE_CORNER_lts[n])
        elif fname.startswith("SE_"):
            n = int(fname.split("_")[-1])
            start_time_offset = 0
            for i in range(1, n):
                start_time_offset += SE_lts[i]
            start_estimate = (n-1)*60 + start_time_offset + SE_start_estimate
            db.update_datafile_time(fname, start_estimate, SE_lts[n])
        elif fname == "NE_1":
            md = db.retrieve_file_metadata("NE_2")
            start_time = md["start_time"] = NE_1_lt - 60
            db.update_datafile_time(fname, start_time, NE_1_lt)
    db.update_datafile_time("PROSPECT_DOWN_OVERNIGHT", -1, 1) #this file has an incorrect live time, no way of knowing what it was
    db.commit()

def fix_track_pos(db):
    spectra = db.retrieve_track_spectra()
    for key, val in spectra.items():
        for data in val:
            spec, coo = data
            print("updating file {} z position".format(spec.fname))
            db.update_file_position_with_offset(spec.fname, -35.25)
    db.commit()

def main():
    db = HFIRBG_DB()
    db.sync_files()
    db.sync_db()
    db.set_calibration_groups()
    fix_corrupted_files(db)
    db.close()
    #db = CartScanFiles()
    #fix_track_pos(db)



if __name__ == "__main__":
    main()
