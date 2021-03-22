from src.utilities.util import *

#rundata = {   "original": [54,55], "original_cart_xzrot_90":56, "corner_no_shield":
#    "CORNER_MEASUREMENT_NO_SHIELD_WALL_LOW_GAIN", "MIF_no_shield":"CORNER_TPS_OVERNIGHT" }
    #"nominal": 57, "EAST CORNER": 58,
    #"Rxoff": [i for i in range(30, 40)]}
#rundata = {"reactor_optimized":"MIF_BOX_REACTOR_OPTIMIZED_OVERNIGHT_LOWEST_GAIN",
#           "track_reactor_off_axis": "TRACK_POS55_48",
#           "nominal_track": "TRACK_POS132_58.5"}
rundata = {"nominal north": "NOMINAL_NORTH", "nominal east": "NOMINAL_EAST", "nominal west": "NOMINAL_WEST",
           "nominal south": "NOMINAL_SOUTH"}
plot_name = "nominal_cardinal_direction"
compare_to = "nominal north"
bins = get_bins(100, 11500, 11400)


def main():
    datadir = get_data_dir()
    data = populate_data(rundata, datadir)
    combine_runs(data)
    rebin_spectra(data, bins)
    #data = background_subtract(data, "Rxoff", get_bins(100, 9400, 3100))
    plot_multi_spectra(data, plot_name, rebin=10)
    plot_subtract_spectra(data, compare_to, plot_name + "_subtract", rebin=100)
    emin = [800*i for i in range(15)]
    emax = [800*(i+1) for i in range(15)]
    for i in range(len(emin)):
        plot_multi_spectra(data, plot_name + "_{}".format(i), emin=emin[i], emax=emax[i])
        plot_subtract_spectra(data, compare_to, plot_name + "_subtract_{}".format(i), emin=emin[i], emax=emax[i])


if __name__ == "__main__":
    main()
