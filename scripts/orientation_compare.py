from util import *

rundata = {  # "original": [54,55], "original_cart_xzrot_90":56,
    "nominal": 57, "EAST CORNER": 58,
    "Rxoff": [i for i in range(30, 40)]}


def main():
    datadir = get_data_dir()
    data = populate_data(rundata, datadir)
    combine_runs(data)
    data = background_subtract(data, "Rxoff", get_bins(100, 3600, 3500))
    plot_multi_spectra(data, "nominal_compare_rxoff_subtracted", rebin=10)
    plot_subtract_spectra(data, "nominal", "nominal_compare_subtract_rxoff_subtracted", rebin=100)


if __name__ == "__main__":
    main()
