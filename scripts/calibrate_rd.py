import sys
from os.path import dirname, realpath


sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.database.HFIRBG_DB import HFIRBG_DB
from src.analysis.Spectrum import SpectrumData
from src.utilities.util import  calibrate_spectra,  combine_runs, populate_data_db
from os.path import join, exists
import os

#expected_peaks = [2614.533, 1293.64, 511.0, 1120.29, 609.31, 2223.245, 7367.96, 5824.6, 805.9, 651.26, 558.46]
expected_peaks = [2614.533, 511.0, 609.31, 768.36, 1120.29, 1238.11, 1764.494, 2204.21]
#expected_peaks = [7724.034, 7645.58, 7631.18, 1460.8, 1332.5, 1293.64, 1173.2, 511.0]
#expected_peaks = [1293.64, 511.0, 374.72, 768.36, 1120.29, 1238.11, 1377.67]
#expected_peaks = [7724.034, 7645.58, 7631.18, 1460.8, 1332.5, 1293.64, 1173.2, 511.0, 374.72, 768.36, 1120.29, 1238.11, 1377.67]
#expected_peaks = [1460.8, 1332.5, 1274.43, 1173.2,964.082,1112.08,1408.013,778.9006]
dt = 86400*7
dt = None

outdir = join(os.environ["HFIRBG_ANALYSIS"], "russian_doll")
calibrate_high_gain = False
gain_setting = ['low']


def main():
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    db = HFIRBG_DB()
    if calibrate_high_gain:
        rd_data = db.get_rd_files(run_grouping=True, gain_setting=['high', '90'], exclude_cal=False)
        rd_cal_data = {}
        for run in rd_data.keys():
            if "_cal" in run:
                rd_cal_data[run] = rd_data[run]
        rd_data = populate_data_db(rd_cal_data, db)
        for run in rd_data.keys():
            print("peak fitting for calibration run {}".format(run))
            for i, spec in enumerate(rd_data[run]):
                calibrate_spectra({"{0}_interval_{1}".format(run, i): spec}, [59.541], db, outdir, True,
                                      True, True, allow_undetermined=True)
    else:
        rd_data = db.get_rd_files(True, gain_setting=gain_setting, rxoff_only=True)
        rd_data = populate_data_db(rd_data, db)
        combine_runs(rd_data, max_interval=dt)
        if not exists(outdir):
            os.mkdir(outdir)
        for run in rd_data.keys():
            print("***************")
            print("calibrating run {}".format(run))
            print("***************")
            if isinstance(rd_data[run], SpectrumData):
                calibrate_spectra(rd_data, expected_peaks, db, outdir, True, True, True)
                break
            else:
                for i, spec in enumerate(rd_data[run]):
                    calibrate_spectra({"{0}_interval_{1}".format(run, i): spec}, expected_peaks, db, outdir, True, True, True)


if __name__ == "__main__":
    main()
