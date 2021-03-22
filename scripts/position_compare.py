from src.utilities.util import *

#rundata = {"original_1": 54, "original_2": 55, "original_cart_xzrot_90": 56,
#           "nominal": 57, "EAST CORNER": 58}
rundata = {"reactor_optimized": "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN.txt",
           "track_reactor_off_axis": "TRACK_POS55_48"}


def main():
    datadir = get_data_dir()
    data = populate_data(rundata, datadir)
    combine_runs(data)
    plot_multi_spectra(data, "reactor_vs_off_axis", rebin=10)
    plot_subtract_spectra(data, "reactor_optimized", "reactor_vs_off_axis_sub", rebin=100)


if __name__ == "__main__":
    main()
