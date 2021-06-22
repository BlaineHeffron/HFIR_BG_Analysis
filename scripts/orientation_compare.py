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

rundata = {"rxon":"MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN.txt","rxoff":"MIF_BOX_AT_REACTOR_RXOFF"}
plot_name = "rxon_vs_off"
compare_to = "rxoff"

rundata = {"MIF": "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN.txt","HB4":"HB4_DOWN_OVERNIGHT_1.txt","SHIELD_CENTER":"CYCLE461_DOWN_FACING_OVERNIGHT.txt"}
plot_name = "HB4_vs_Shield_Center"
compare_to = "SHIELD_CENTER"
bins = get_bins(100, 11500, 11400)

rundata = {"anomaly":"HB4_DOWN_31", "normal": "HB4_DOWN_36"}
plot_name = "high_rate_anomaly"
compare_to = "normal"
bins = get_bins(100, 11500, 11400)



def main():
    datadir = get_data_dir()
    data = populate_data(rundata, datadir)
    combine_runs(data)
    rebin_spectra(data, bins)
    #data = background_subtract(data, "Rxoff", get_bins(100, 9400, 3100))
    plot_multi_spectra(data, plot_name, rebin=10)
    plot_subtract_spectra(data, compare_to, plot_name + "_subtract", rebin=1000)
    emin = [800*i for i in range(15)]
    emax = [800*(i+1) for i in range(15)]
    #for i in range(len(emin)):
    #    plot_multi_spectra(data, plot_name + "_{}".format(i), emin=emin[i], emax=emax[i])
    #    plot_subtract_spectra(data, compare_to, plot_name + "_subtract_{}".format(i), emin=emin[i], emax=emax[i])


if __name__ == "__main__":
    main()
