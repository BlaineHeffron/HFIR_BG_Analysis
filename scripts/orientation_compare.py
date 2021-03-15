from util import *

rundata = {   "original": [54,55], "original_cart_xzrot_90":56, "corner_no_shield":
    "CORNER_MEASUREMENT_NO_SHIELD_WALL_LOW_GAIN", "MIF_no_shield":"CORNER_TPS_OVERNIGHT" }
    #"nominal": 57, "EAST CORNER": 58,
    #"Rxoff": [i for i in range(30, 40)]}
bins = get_bins(100, 9700, 3200)


def main():
    datadir = get_data_dir()
    data = populate_data(rundata, datadir)
    combine_runs(data)
    rebin_spectra(data, bins)
    #data = background_subtract(data, "Rxoff", get_bins(100, 9400, 3100))
    plot_multi_spectra(data, "no_wall_comparison", rebin=10)
    plot_subtract_spectra(data, "original", "no_wall_compare_subtract", rebin=100)
    emin = [1000*i for i in range(9)] + [9000]
    emax = [1000*(i+1) for i in range(9)] + [9500]
    for i in range(len(emin)):
        plot_multi_spectra(data, "no_wall_{}".format(i), emin=emin[i], emax=emax[i])
        plot_subtract_spectra(data, "original", "no_wall_subtract_{}".format(i), emin=emin[i], emax=emax[i])


if __name__ == "__main__":
    main()
