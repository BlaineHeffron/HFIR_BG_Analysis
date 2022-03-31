import sys
from os.path import dirname, realpath
sys.path.insert(1, dirname(dirname(realpath(__file__))))

from src.utilities.util import *
from src.database.HFIRBG_DB import HFIRBG_DB

rundata = { "corner_no_shield": "CORNER_MEASUREMENT_NO_SHIELD_WALL_LOW_GAIN",
            "MIF_no_shield":"CORNER_TPS_OVERNIGHT" }
rundata = {"Reactor Spectrum":"MIF_BOX_REACTOR_OPTIMIZED_OVERNIGHT_LOWEST_GAIN.txt"}
#rundata = {"rxon":"MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN.txt","rxoff":"MIF_BOX_AT_REACTOR_RXOFF"}
#rundata = {"Reactor Spectrum":"MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN.txt"}
rundata = {"PROSPECT Spectrum":"PROSPECT_DOWN_OVERNIGHT.txt"}
bins = get_bins(0, 11500, 11500)
outdir = join(os.environ["HFIRBG_ANALYSIS"], "spectrum_plots")

def main():
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    datadir = get_data_dir()
    db = HFIRBG_DB()
    data = populate_data(rundata, datadir, db)
    combine_runs(data)
    rebin_spectra(data, bins)
    #data = background_subtract(data, "Rxoff", get_bins(100, 9400, 3100))
    emin = [100+1000*i for i in range(12)]
    emax = [100+1000*(i+1) for i in range(12)]
    for key in data.keys():
        plot_spectra([data[key]], key)
        for i in range(len(emin)):
            plot_spectra([data[key]], join(outdir,"{0}, {1}-{2}".format(key, emin[i], emax[i])), emin=emin[i], emax=emax[i])
    #plot_subtract_spectra(data, "original", "no_wall_compare_subtract", rebin=100)


if __name__ == "__main__":
    main()
