import sys
import os
from os.path import dirname, realpath, join
from argparse import ArgumentParser
sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.utilities.util import get_spec_from_root, plot_multi_spectra, get_bins, populate_rd_data, combine_runs, \
    rebin_spectra, set_indices
from src.database.HFIRBG_DB import HFIRBG_DB
from src.utilities.FitUtils import minimize_diff

outdir = join(join(os.environ["HFIRBG_ANALYSIS"], "russian_doll"), "sim")

full_bins = get_bins(30, 11500, 11470)
med_range = get_bins(30, 2400, 2670)
low_range = get_bins(30, 60, 120)
ninety_range = get_bins(30, 88, 180)
acq_id_map = {5: full_bins, 7: low_range, 17: ninety_range, 16: med_range}

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
    rebin_spectra(hist_dict, full_bins)
    plot_name = join(outdir, "sim_compare")
    plot_multi_spectra(hist_dict, plot_name)
    for i in range(len(emin)):
        plot_multi_spectra(hist_dict, plot_name + "_{}".format(i), emin=emin[i], emax=emax[i])

    db = HFIRBG_DB()
    rd_data = db.get_rd_files(min_time=1000, rxon_only=True)
    rd_data = populate_rd_data(rd_data, db)
    for key in rd_data:
        combine_runs(rd_data[key], ignore_failed_add=True)
    for shield_id in rd_data.keys():
        rd_shield_id = shield_id - 2
        for acq_id in rd_data[shield_id].keys():
            plot_multi_spectra({"sim_{}".format(rd_shield_id): hist_dict["RD_{}".format(rd_shield_id)],
                                "data_{0}_{1}".format(rd_shield_id, acq_id): rd_data[shield_id][acq_id]},
                               join(outdir, "sim_data_comparison_{0}_{1}".format(rd_shield_id, acq_id)),
                               rebin_edges=acq_id_map[acq_id])
            if acq_id == 5:
                try:
                    rd_data[shield_id][acq_id].rebin(full_bins)
                    start_index, end_index = set_indices(0, 0, 7620, 7650, rd_data[shield_id][acq_id])
                    sim = hist_dict["RD_{}".format(rd_shield_id)].get_normalized_hist()[start_index:end_index]
                    data = rd_data[shield_id][acq_id].get_normalized_hist()[start_index:end_index]
                    data_err = rd_data[shield_id][acq_id].get_normalized_err()[start_index:end_index]
                    scale = minimize_diff(sim, data, data_err)
                    print("best scale factor found for shield {0} is {1}".format(shield_id, scale))
                    hist_dict["RD_{}".format(rd_shield_id)].scale_hist(scale)
                    for i in range(len(emin)):
                        plot_multi_spectra({"sim_{}".format(rd_shield_id): hist_dict["RD_{}".format(rd_shield_id)],
                                            "data_{0}_{1}".format(rd_shield_id, acq_id): rd_data[shield_id][acq_id]},
                                           join(outdir, "sim_scaled_data_comparison_{0}_{1}_en{2}".format(rd_shield_id, acq_id, i)),
                                           emin=emin[i], emax=emax[i])
                except RuntimeError as e:
                    print(e)
                    for i in range(len(emin)):
                        plot_multi_spectra({"sim_{}".format(rd_shield_id): hist_dict["RD_{}".format(rd_shield_id)],
                                            "data_{0}_{1}".format(rd_shield_id, acq_id): rd_data[shield_id][acq_id]},
                                           join(outdir, "sim_data_comparison_{0}_{1}_en{2}".format(rd_shield_id, acq_id, i)),
                                           rebin_edges=acq_id_map[acq_id], emin=emin[i], emax=emax[i])


if __name__ == "__main__":
    main()