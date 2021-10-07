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

#rundata = {"MIF": "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN.txt","HB4":"HB4_DOWN_OVERNIGHT_1.txt","SHIELD_CENTER":"CYCLE461_DOWN_FACING_OVERNIGHT.txt"}
rundata = {"MIF": "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN.txt","HB4":"HB4_DOWN_OVERNIGHT_1.txt"}
plot_name = "HB4_vs_MIF"
compare_to = "MIF"
bins = get_bins(100, 11500, 11400)

rundata = {"0-10%": [639,640],"10-30%":[641],"30-50%":[642],"50-70%":[643],"70-90%":[644],"90-100%":[645], "100%": [747 + i for i in range(20)]}
plot_name = "reactor_power_compare_largebin"
compare_to = "100%"
bins = get_bins(100, 11500, 11400)

rundata = {"anomaly":"HB4_DOWN_31", "normal": "HB4_DOWN_36"}
plot_name = "high_rate_anomaly"
compare_to = "normal"
bins = get_bins(100, 11500, 11400)

rundata = {"rxoff": [3830 + i for i in range(6)], "rxon": [3819 + i for i in range(10)], "rxon_lead": [4587 + i for i in range(4)], "rxon_lead_6water": [4779 + i for i in range(3)]}
plot_name = "low_energy_RD"
compare_to = "rxoff"
bins = get_bins(20., 60., 4000)

rundata = {str(i) : [i] for i in range(4400,4406)}
plot_name = "rate_increase_aug27-29"
compare_to = "4405"
bins = get_bins(100, 11500, 11400)


rundata = {"rxon_lead": [4587 + i for i in range(4)], "rxon_lead_6water": [4779 + i for i in range(3)]}
plot_name = "low_energy_RD_90kev"
compare_to = "rxon_lead"
bins = get_bins(70., 90., 2000)


#rundata = {"Russian Doll": [i for i in range(3819, 3829)], "Russian Doll + Water": [4971 + i for i in range(7)], "Collimator": [2091 + i for i in range(30)]}
rundata = {"Russian Doll - RxOn": [i for i in range(3819, 3829)], "Russian Doll + Water - RxOn": [4971 + i for i in range(7)], "Russian Doll - RxOff": [i for i in range(3830,3836)], "Collimator - RxOn": [i for i in range(2026,2088)]}
#rundata = {"RxOn": [i for i in range(3819, 3829)], "Russian Doll + Water": [4971 + i for i in range(7)], "Collimator": [2091 + i for i in range(30)]}
plot_name = "low_energy_RD_60kev"
compare_to = "Russian Doll - RxOff"
bins = get_bins(1., 60., 5900)

#rundata = {"rxoff": [i for i in range(4011, 4018)], "rxon": [i for i in range(3051, 3055)], "rxoff_lead": [i for i in range(4410, 4426)], "rxon_lead" : [i for i in range(4395,4409)] }
rundata = {"Russian Doll - RxOff": [i for i in range(4012, 4018)], "Russian Doll - RxOn": [i for i in range(3051, 3055)], "Collimator - RxOn": [54,55]}
plot_name = "full_energy_range_RD"
compare_to = "Russian Doll - RxOff"
bins = get_bins(100, 11500, 11400)


def main():
    datadir = get_data_dir()
    data = populate_data(rundata, datadir)
    combine_runs(data)
    rebin_spectra(data, bins)
    #data = background_subtract(data, "Rxoff", get_bins(100, 9400, 3100))
    plot_multi_spectra(data, plot_name, rebin=10, emin=15)
    plot_subtract_spectra(data, compare_to, plot_name + "_subtract", rebin=100, emin=15)
    emin = [800*i for i in range(15)]
    emax = [800*(i+1) for i in range(15)]
    #for i in range(len(emin)):
    #    plot_multi_spectra(data, plot_name + "_{}".format(i), emin=emin[i], emax=emax[i])
    #    plot_subtract_spectra(data, compare_to, plot_name + "_subtract_{}".format(i), emin=emin[i], emax=emax[i])


if __name__ == "__main__":
    main()
