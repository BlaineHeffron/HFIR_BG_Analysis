import sys
import os
from os.path import dirname, realpath, join
from argparse import ArgumentParser


sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.utilities.util import get_spec_from_root, plot_multi_spectra, get_bins, populate_rd_data, combine_runs, \
    rebin_spectra, set_indices, get_rd_data, subtract_rd_data
from src.database.HFIRBG_DB import HFIRBG_DB
from src.utilities.FitUtils import minimize_diff
from src.analysis.Spectrum import SubtractSpectrum

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
    arg.add_argument("basedir", help="path to directory containing GeRD_# directories", type=str)
    arg.add_argument("--neutron", "-n", action="store_true", help="neutron sims directory")
    args = arg.parse_args()
    if args.neutron:
        hist_dict = {"RD_0": None}
        newdir = os.path.join(outdir, "neutron_sim")
        if not os.path.exists(newdir):
            os.mkdir(newdir)

    else:
        newdir = outdir
        hist_dict = {"RD_" + str(i): None for i in range(6)}
    if args.neutron:
        nm = "RD_0"
        mydir = args.basedir
        for dirpath in os.listdir(mydir):
            rootfile = join(join(mydir, dirpath), "GeRD_neutrons.root")
            if os.path.exists(rootfile):
                hist_en = get_spec_from_root(rootfile, "GeEnergyPlugin/hGeEnergy", "accumulated/runtime", True, 1000.,1)
                if not hist_dict[nm]:
                    hist_dict[nm] = hist_en
                else:
                    hist_dict[nm].add(hist_en)
    else:
        for i in range(6):
            mydir = join(args.basedir, "GeRD_{}".format(i))
            nm = "RD_" + str(i)
            for dirpath in os.listdir(mydir):
                rootfile = join(join(mydir, dirpath), "GeRD_{}.root".format(i))
                if os.path.exists(rootfile):
                    hist_en = get_spec_from_root(rootfile, "GeEnergyPlugin/hGeEnergy", "accumulated/runtime", True, 1000.,
                                                 1)
                    if not hist_dict[nm]:
                        hist_dict[nm] = hist_en
                    else:
                        hist_dict[nm].add(hist_en)
    print(hist_dict)
    for key in hist_dict:
        print(hist_dict[key].data)
    rebin_spectra(hist_dict, full_bins)
    plot_name = join(newdir, "sim_compare")
    plot_multi_spectra(hist_dict, plot_name)
    for i in range(len(emin)):
        plot_multi_spectra(hist_dict, plot_name + "_{}".format(i), emin=emin[i], emax=emax[i])

    db = HFIRBG_DB()
    rd_data = get_rd_data(db, rxon_only=True, min_time=1000)
    rd_data_off = get_rd_data(db, rxoff_only=True, min_time=1000)
    rd_sub = subtract_rd_data(rd_data, rd_data_off, acq_id_bin_edges=acq_id_map)
    rxoff_sub_sim_diff = {}
    for shield_id in rd_data.keys():
        rd_shield_id = shield_id - 2
        if args.neutron and rd_shield_id > 0:
            break
        for acq_id in rd_data[shield_id].keys():
            if shield_id in rd_sub and rd_sub[shield_id] and acq_id in rd_sub[shield_id].keys():
                plot_multi_spectra({"sim_{}".format(rd_shield_id): hist_dict["RD_{}".format(rd_shield_id)],
                                    "RxOff_sub_data_{0}_{1}".format(rd_shield_id, acq_id): rd_sub[shield_id][acq_id]},
                                   join(newdir, "sim_data_rxoff_sub_comparison_{0}_{1}".format(rd_shield_id, acq_id)),
                                   rebin_edges=acq_id_map[acq_id])

            plot_multi_spectra({"sim_{}".format(rd_shield_id): hist_dict["RD_{}".format(rd_shield_id)],
                                "data_{0}_{1}".format(rd_shield_id, acq_id): rd_data[shield_id][acq_id]},
                               join(newdir, "sim_data_comparison_{0}_{1}".format(rd_shield_id, acq_id)),
                               rebin_edges=acq_id_map[acq_id])
            if acq_id == 5:
                try:
                    if shield_id in rd_sub and rd_sub[shield_id] and acq_id in rd_sub[shield_id].keys():
                        start_index, end_index = set_indices(0, 0, 7500, 7800, rd_sub[shield_id][acq_id])
                        sim = hist_dict["RD_{}".format(rd_shield_id)].get_normalized_hist()[start_index:end_index]
                        data = rd_sub[shield_id][acq_id].get_normalized_hist()[start_index:end_index]
                        data_err = rd_sub[shield_id][acq_id].get_normalized_err()[start_index:end_index]
                        scale = minimize_diff(sim, data, data_err)
                        print("best scale factor found for shield {0} is {1}".format(shield_id, scale))
                        hist_dict["RD_{}".format(rd_shield_id)].scale_hist(scale, True)
                        rxoff_sub_sim_diff["shield {}".format(str(rd_shield_id))] = \
                            SubtractSpectrum(rd_sub[shield_id][acq_id], hist_dict["RD_{}".format(rd_shield_id)])
                        for i in range(len(emin)):
                            plot_multi_spectra({"sim_{}".format(rd_shield_id): hist_dict["RD_{}".format(rd_shield_id)],
                                                "data_{0}_{1}".format(rd_shield_id, acq_id): rd_sub[shield_id][acq_id]},
                                               join(newdir, "sim_scaled_data_rxoff_sub_comparison_{0}_{1}_en{2}".format(rd_shield_id, acq_id, i)),
                                               emin=emin[i], emax=emax[i], ebars=False)
                    else:
                        rd_data[shield_id][acq_id].rebin(full_bins)
                        start_index, end_index = set_indices(0, 0, 7500, 7800, rd_data[shield_id][acq_id])
                        sim = hist_dict["RD_{}".format(rd_shield_id)].get_normalized_hist()[start_index:end_index]
                        data = rd_data[shield_id][acq_id].get_normalized_hist()[start_index:end_index]
                        data_err = rd_data[shield_id][acq_id].get_normalized_err()[start_index:end_index]
                        scale = minimize_diff(sim, data, data_err)
                        print("best scale factor found for shield {0} is {1}".format(shield_id, scale))
                        hist_dict["RD_{}".format(rd_shield_id)].scale_hist(scale, True)
                        for i in range(len(emin)):
                            plot_multi_spectra({"sim_{}".format(rd_shield_id): hist_dict["RD_{}".format(rd_shield_id)],
                                                "data_{0}_{1}".format(rd_shield_id, acq_id): rd_data[shield_id][ acq_id]},
                                               join(newdir, "sim_scaled_data_comparison_{0}_{1}_en{2}".format(rd_shield_id, acq_id, i)),
                                               emin=emin[i], emax=emax[i], ebars=False)
                except RuntimeError as e:
                    print(e)
                    for i in range(len(emin)):
                        plot_multi_spectra({"sim_{}".format(rd_shield_id): hist_dict["RD_{}".format(rd_shield_id)],
                                            "data_{0}_{1}".format(rd_shield_id, acq_id): rd_data[shield_id][acq_id]},
                                           join(newdir,
                                                "sim_data_comparison_{0}_{1}_en{2}".format(rd_shield_id, acq_id, i)),
                                           rebin_edges=acq_id_map[acq_id], emin=emin[i], emax=emax[i], ebars=False)
    plot_multi_spectra(rxoff_sub_sim_diff, join(newdir, "rxoff_sub_data_minus_scaled_sim"))
    for i in range(len(emin)):
        plot_multi_spectra(rxoff_sub_sim_diff, join(newdir, "rxoff_sub_data_minus_scaled_sim_en{0}".format(i)),
                                                    emin=emin[i], emax=emax[i], ebars=False)


if __name__ == "__main__":
    main()
