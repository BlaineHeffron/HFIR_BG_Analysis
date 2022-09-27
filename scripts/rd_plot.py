import sys
import os
from os.path import dirname, realpath, join

sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.database.HFIRBG_DB import HFIRBG_DB
from src.utilities.util import populate_rd_data, plot_time_series, combine_runs, plot_spectra, \
    populate_data_db, get_bins, plot_multi_spectra, get_rd_data

outdir = join(os.environ["HFIRBG_ANALYSIS"], "russian_doll")

acq_id_map = {5: "low gain", 7: "high gain", 17: "90 keV range", 16: "medium gain"}

shield_names = {0:"no water", 3: "7 layers water", 4: "6 layers water + water underneath", 5: "7 layers water + water underneath"}

full_bins = get_bins(30, 11500, 11470)
#full_bins = get_bins(30, 3000, 2970)
med_range = get_bins(30, 2400, 2670)
low_range = get_bins(30, 60, 90)
ninety_range = get_bins(30, 60, 180)
acq_id_bin_map = {5: full_bins, 7: low_range, 17: ninety_range, 16: med_range}
Emin = 50
Emax = None
low_e_ranges = [20, 30, 40, 50, 60]
max_interval = 86400*7

plot_time_series_bool = False
plot_spectra_bool = False
plot_spectra_compare_bool = True


def print_rates(spectra, low, high):
    print("rates from {0} - {1} keV".format(low,high))
    for key, spec in spectra.items():
        rate, err = spec.integrate(low, high, True)
        print("{0} {1} ~ {2}".format(key, rate, err))


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

    if plot_spectra_compare_bool:
        rd_data = get_rd_data(db, rxon_only=True)
        rd_data_off = get_rd_data(db, rxoff_only=True)
        high_e_data = {}
        low_e_data = {}
        med_e_data = {}
        high_e_data_off = {}
        low_e_data_off = {}
        med_e_data_off = {}
        for shield_id in rd_data.keys():
            rd_shield_id = shield_id - 2
            #if rd_shield_id in shield_names.keys():
            #    rd_shield_id = shield_names[rd_shield_id]
            #else:
            #    continue
            for acq_id in rd_data[shield_id].keys():
                if acq_id == 7 or acq_id == 17:
                    low_e_data[rd_shield_id] = rd_data[shield_id][acq_id]
                    if shield_id in rd_data_off.keys():
                        if acq_id in rd_data_off[shield_id].keys():
                            low_e_data_off[rd_shield_id] = rd_data_off[shield_id][acq_id]
                elif acq_id == 5:
                    high_e_data[rd_shield_id] = rd_data[shield_id][acq_id]
                    if shield_id in rd_data_off.keys():
                        if acq_id in rd_data_off[shield_id].keys():
                            high_e_data_off[rd_shield_id] = rd_data_off[shield_id][acq_id]
                else:
                    med_e_data[rd_shield_id] = rd_data[shield_id][acq_id]
                    if shield_id in rd_data_off.keys():
                        if acq_id in rd_data_off[shield_id].keys():
                            med_e_data_off[rd_shield_id] = rd_data_off[shield_id][acq_id]
        plot_multi_spectra(high_e_data, join(outdir, "rd_high_en_shield_comparison"), rebin_edges=full_bins)
        plot_multi_spectra(low_e_data, join(outdir, "rd_low_en_shield_comparison"), rebin_edges=ninety_range, ylog=False, emin=ninety_range[0])
        plot_multi_spectra(med_e_data, join(outdir, "rd_med_en_shield_comparison"), rebin_edges=med_range)
        plot_multi_spectra(high_e_data_off, join(outdir, "rd_RxOff_high_en_shield_comparison"), rebin_edges=full_bins)
        plot_multi_spectra(low_e_data_off, join(outdir, "rd_RxOff_low_en_shield_comparison"), rebin_edges=ninety_range, ylog=False, emin=ninety_range[0])
        plot_multi_spectra(med_e_data_off, join(outdir, "rd_RxOff_med_en_shield_comparison"), rebin_edges=med_range)
        print("low gain rates rxon")
        print_rates(high_e_data, 30, 60)
        print_rates(high_e_data, 50, 11380)
        print_rates(high_e_data, None, None)
        print("med gain rates rxon")
        print_rates(med_e_data, 30, 60)
        print("high gain rates rxon")
        print_rates(low_e_data, 30, 60)
        print("low gain rates rxoff")
        print_rates(high_e_data_off, 30, 60)
        print_rates(high_e_data_off, 50, 11380)
        print_rates(high_e_data_off, None, None)
        print("med gain rates rxoff")
        print_rates(med_e_data_off, 30, 60)
        print("high gain rates rxoff")
        print_rates(low_e_data_off, 30, 60)
        emin = [3000 * i for i in range(4)]
        emax = [3000 * (i + 1) for i in range(4)]
        for i in range(len(emin)):
            plot_multi_spectra(high_e_data, join(outdir, "rd_high_en_shield_comparison_{}".format(i)),
                               rebin_edges=full_bins, emin=emin[i], emax=emax[i], ebars=False)
        for key in high_e_data_off.keys():
            for i in range(len(emin)):
                #plot_multi_spectra({"shield_" + str(key) + "_rxoff": high_e_data_off[key],
                 plot_multi_spectra({"rxoff": high_e_data_off[key],
                                "rxon": high_e_data[key]},
                               join(outdir, "rd_high_en_rxon_off_shield_{0}_comparison_{1}".format(key, i)),
                               rebin_edges=full_bins, emin=emin[i], emax=emax[i], ebars=False, figsize=(14,8))
        for key in low_e_data_off.keys():
            plot_multi_spectra({"shield_" + str(key) + "_rxoff": low_e_data_off[key],
                                "shield_" + str(key) + "_rxon": low_e_data[key]},
                               join(outdir, "rd_low_en_rxon_off_shield_{0}_comparison".format(key)),
                               rebin_edges=ninety_range, ebars=False)

if __name__ == "__main__":
    main()
