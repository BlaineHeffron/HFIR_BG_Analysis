import sys
import os
from os.path import dirname, realpath, join

sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.database.HFIRBG_DB import HFIRBG_DB
from src.utilities.util import populate_rd_data, plot_time_series, combine_runs, plot_spectra, populate_data, \
    populate_data_db

outdir = join(os.environ["HFIRBG_ANALYSIS"], "russian_doll")

acq_id_map = {5: "low gain", 7: "high gain", 17: "90 keV range", 16: "medium gain"}
Emin = 50
Emax = None
low_e_ranges = [20, 30, 40, 50, 60]
max_interval = 86400*7

plot_time_series_bool = True
plot_spectra_bool = False



def main():
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    db = HFIRBG_DB()
    if plot_time_series_bool:
        rd_data = db.get_rd_files(min_time=100)
        rd_data = populate_rd_data(rd_data, db)
        low_e_data = {"low_e_data": []}
        legend = []
        for shield_id in rd_data.keys():
            shield_info = db.retrieve_shield_info(shield_id)
            for acq_id in rd_data[shield_id].keys():
                low_e_data["low_e_data"] += rd_data[shield_id][acq_id]
                name = "{0}, {1}".format(shield_info[0], acq_id_map[acq_id])
                legend += [name]*len(rd_data[shield_id][acq_id])
                plot_time_series({name: rd_data[shield_id][acq_id]}, outdir, emin=Emin, emax=Emax)
        for i in range(len(low_e_ranges) - 1):
            plot_time_series(low_e_data, outdir, emin=low_e_ranges[i], emax=low_e_ranges[i + 1], legend_map={"low_e_data": legend}, ymin=.005, legend_fraction=0.3, figsize=(10,4))
        rd_data = db.get_rd_files(True, min_time=100)
        rd_data = populate_data_db(rd_data, db)
        low_e_data = {"low_e_data_runs": []}
        legend = []
        for run_name in rd_data.keys():
            low_e_data["low_e_data_runs"] += rd_data[run_name]
            legend += [run_name] * len(rd_data[run_name])
            plot_time_series({run_name: rd_data[run_name]}, outdir, emin=Emin, emax=Emax)
        for i in range(len(low_e_ranges) - 1):
            plot_time_series(low_e_data, outdir, emin=low_e_ranges[i], emax=low_e_ranges[i + 1],
                             legend_map={"low_e_data_runs": legend}, ymin=.005, legend_fraction=0.3, figsize=(16,10))
    if plot_spectra_bool:
        rd_data = db.get_rd_files(True)
        rd_data = populate_data_db(rd_data, db)
        combine_runs(rd_data, max_interval=max_interval)
        emin = [100 + 1000 * i for i in range(12)]
        emax = [100 + 1000 * (i + 1) for i in range(12)]
        for run_name in rd_data.keys():
            for j, spec in enumerate(rd_data[run_name]):
                plot_spectra([spec], join(outdir, run_name + "_interval_{}".format(j)))
                for i in range(len(emin)):
                    plot_spectra([spec], join(outdir,"{0}_interval_{1}, {2}-{3}".format(run_name, j, emin[i], emax[i])), emin=emin[i], emax=emax[i])


if __name__ == "__main__":
    main()
