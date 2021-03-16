from src.utilities.util import *


rundata = { "corner_no_shield": "CORNER_MEASUREMENT_NO_SHIELD_WALL_LOW_GAIN",
            "MIF_no_shield":"CORNER_TPS_OVERNIGHT" }
bins = get_bins(100, 9700, 3200)

def main():
    datadir = get_data_dir()
    data = populate_data(rundata, datadir)
    combine_runs(data)
    rebin_spectra(data, bins)
    #data = background_subtract(data, "Rxoff", get_bins(100, 9400, 3100))
    emin = [800*i for i in range(12)]
    emax = [800*(i+1) for i in range(12)]
    for key in data.keys():
        plot_spectra([data[key]], key)
        for i in range(len(emin)):
            plot_spectra([data[key]], "{0}_{1}-{2}".format(key, emin[i], emax[i]), emin=emin[i], emax=emax[i])
    #plot_subtract_spectra(data, "original", "no_wall_compare_subtract", rebin=100)


if __name__ == "__main__":
    main()
