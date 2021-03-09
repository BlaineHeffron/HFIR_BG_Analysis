from util import *

rundata = {   "original": [54,55], "original_cart_xzrot_90":56, "corner_no_shield": 59}
    #"nominal": 57, "EAST CORNER": 58,
    #"Rxoff": [i for i in range(30, 40)]}


def main():
    datadir = get_data_dir()
    data = populate_data(rundata, datadir)
    combine_runs(data)
    #data = background_subtract(data, "Rxoff", get_bins(100, 3600, 3500))
    plot_multi_spectra(data, "no_wall_comparison", rebin=10)
    plot_subtract_spectra(data, "original", "no_wall_compare_subtract", rebin=100)


if __name__ == "__main__":
    main()
