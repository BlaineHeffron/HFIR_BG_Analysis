import sys
import os
from os.path import dirname, realpath, join
from argparse import ArgumentParser


sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.utilities.util import get_spec_from_root, plot_multi_spectra, get_bins, populate_rd_data, combine_runs, \
    rebin_spectra, set_indices, get_rd_data, subtract_rd_data, write_pyspec, load_pyspec, compare_peaks, \
    retrieve_peak_areas
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

neut_peaks = [558.46, 7367.96]
gam_peaks = [7631.18, 7645.58]

def main():
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    arg = ArgumentParser()
    arg.add_argument("basedir", help="path to directory containing RD sim files", type=str)
    args = arg.parse_args()
    newdir = os.path.join(outdir, "RD_gamma_neutron_sim")
    if not os.path.exists(newdir):
        os.mkdir(newdir)
    nm = "RD_0"
    nsim = None
    gsim = None
    if os.path.exists(join(args.basedir,"GeRD_neutrons/neutronRD.pyspec")):
        nsim = load_pyspec(join(args.basedir,"GeRD_neutrons/neutronRD.pyspec"))
    else:
        neut_name = join(args.basedir,"GeRD_neutrons/neutronRD.pyspec")
        mydir = join(args.basedir, "GeRD_neutrons")
        for dirpath in os.listdir(mydir):
            rootfile = join(join(mydir, dirpath), "GeRD_neutrons.root")
            if os.path.exists(rootfile):
                hist_en = get_spec_from_root(rootfile, "GeEnergyPlugin/hGeEnergy", "accumulated/runtime", True, 1000.,1)
                if not nsim:
                    nsim = hist_en
                else:
                    nsim.add(hist_en)
        write_pyspec(neut_name, nsim)
    gname = join(args.basedir,"GeRD_0/GeRD_0.pyspec")
    if os.path.exists(gname):
        gsim = load_pyspec(gname)
    else:
        mydir = join(args.basedir, "GeRD_0")
        for dirpath in os.listdir(mydir):
            rootfile = join(join(mydir, dirpath), "GeRD_0.root")
            if os.path.exists(rootfile):
                hist_en = get_spec_from_root(rootfile, "GeEnergyPlugin/hGeEnergy", "accumulated/runtime", True, 1000.,1)
                if not gsim:
                    gsim = hist_en
                else:
                    gsim.add(hist_en)
        write_pyspec(gname, gsim)
    nsim.rebin(full_bins)
    gsim.rebin(full_bins)
    nareas, ndareas = retrieve_peak_areas(nsim, neut_peaks)
    gareas, gdareas = retrieve_peak_areas(gsim, gam_peaks)
    print("simulated neutron peak rates: ")
    print(nareas)
    print("area uncertainties")
    print(ndareas)
    print("simulated gamma peak rates: ")
    print(gareas)
    print("area uncertainties")
    print(gdareas)

    db = HFIRBG_DB()
    rd_data = get_rd_data(db, rxon_only=True, min_time=1000)
    rd_data_off = get_rd_data(db, rxoff_only=True, min_time=1000)
    rd_sub = subtract_rd_data(rd_data, rd_data_off, acq_id_bin_edges=acq_id_map)
    for shield_id in rd_sub.keys():
        rd_shield_id = shield_id - 2
        if 5 in rd_sub[shield_id].keys():
            outname = "GammaNeutronFit_{}".format(rd_shield_id)
            areas, dareas = retrieve_peak_areas(rd_data[shield_id][5], neut_peaks)
            print("data peak rates: ")
            print(areas)
            print("area uncertainties")
            print(dareas)
            print("neut peak ratios")
            keylist = list(nareas.keys())
            p1 = keylist[0]
            p2 = keylist[1]
            print("sim: {0}\ndata: {1}".format(nareas[p1]/nareas[p2], areas[p1]/areas[p2]))

if __name__ == "__main__":
    main()
