import sys
from os.path import dirname, realpath

sys.path.insert(1, dirname(dirname(realpath(__file__))))

from src.database.HFIRBG_DB import HFIRBG_DB
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
rundata = {"MIF": "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN.txt","HB4":"hb4_down_overnight_1.txt"}
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

rundata = {"rxoff": [i for i in range(4011, 4018)], "rxon": [i for i in range(3051, 3055)], "rxoff_lead": [i for i in range(4410, 4426)], "rxon_lead" : [i for i in range(4395,4409)] }
plot_name = "full_energy_range_RD"
compare_to = "rxoff"
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

rundata = {"rxon_RD":[i  for i in range(3819,3829)], "rxoff_RD": [i for i in range(3830,3836)]}
plot_name = "high_gain_russian_doll"
compare_to = "rxoff_RD"
bins = get_bins(1, 64, 630)

rundata = {"rxon_RD_lowgain":[i for i in range(3051,3055)], "rxoff_RD_lowgain": [i for i in range(4011,4018)]}
plot_name = "low_gain_russian_doll"
compare_to = "rxoff_RD_lowgain"
bins = get_bins(100, 11500, 11400)

rundata = {"rxoff": [i for i in range(4011, 4018)], "rxon": [i for i in range(3051, 3055)], "rxoff_lead": [i for i in range(4410, 4426)], "rxon_lead" : [i for i in range(4395,4409)] }
plot_name = "full_energy_range_RD"
compare_to = "rxoff"
bins = get_bins(100, 11500, 11400)

rundata = {"rxon_lead": [4587 + i for i in range(4)], "rxon_lead_6water": [4779 + i for i in range(3)], "rxon_lead_60water": [4971 + i for i in range(7)]}
plot_name = "low_energy_RD_90kev"
compare_to = "rxon_lead"
bins = get_bins(1., 90., 8900)

rundata = {"rxon_lead": [4587 + i for i in range(4)], "rxon_lead_6water": [4779 + i for i in range(3)], "rxon_lead_60water": [4971 + i for i in range(7)], "rxon":[i  for i in range(3819,3829)], "rxoff": [i for i in range(3830,3836)]}
plot_name = "high_gain_russian_doll"
compare_to = "rxoff"
bins = get_bins(1, 64, 6300)

rundata = {"rxon_lead": [4587 + i for i in range(4)], "rxon_lead_6water": [4779 + i for i in range(3)], "rxon_lead_60water": [4971 + i for i in range(7)], "rxon":[i  for i in range(3819,3829)], "rxoff": [i for i in range(3830,3836)]}
plot_name = "high_gain_russian_doll_25"
compare_to = "rxoff"
bins = get_bins(25, 64, 6300)

rundata = {"rxoff": [i for i in range(4011, 4018)], "rxon": [i for i in range(3051, 3055)], "rxoff_lead": [i for i in range(4410, 4426)], "rxon_lead" : [i for i in range(4395,4409)] }
plot_name = "full_energy_range_RD_2.2"
compare_to = "rxoff"
bins = get_bins(2100, 2300, 200)

rundata = {"rxon":"MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN.txt","rxoff":"MIF_BOX_AT_REACTOR_RXOFF"}
plot_name = "rxon_vs_off"
compare_to = "rxoff"
bins = get_bins(2000, 3000, 1000)

rundata = {"MIF": "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN.txt","HB4":"HB4_DOWN_OVERNIGHT_1.txt"}
plot_name = "HB4_vs_MIF"
compare_to = "MIF"
bins = get_bins(4500, 5000, 500)

#rundata = {"MIF": "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN.txt","HB4":"HB4_DOWN_OVERNIGHT_1.txt"}
rundata = {"MIF": "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN.txt","HB4":"HB4_DOWN_OVERNIGHT_1.txt","SHIELD_CENTER":"CYCLE461_DOWN_FACING_OVERNIGHT.txt"}
plot_name = "HB4_vs_MIF_vs_SHIELDCENTER"
compare_to = "MIF"
bins = get_bins(100, 11500, 11400)

rundata = {"SE": "EAST_FACE_16", "NE": "EAST_FACE_2", "SHIELD_CENTER":"CYCLE461_DOWN_FACING_OVERNIGHT.txt"}
rundata = {"SE": "EAST_FACE_16", "NE": "EAST_FACE_2", "Middle":"EAST_FACE_1"}
#rundata = {"SE": "EAST_FACE_16", "NE": "EAST_FACE_6"}
plot_name = "EAST_SIDE_HIGH_RATE_VS_LOW_RATE"
compare_to = "NE"
bins = get_bins(100, 11500, 11400)

rundata = {"RD_494_ON": [4395 + i for i in range(5)], "RD_494_OFF": [4011 + i for i in range(5)]}
plot_name = "RD_ON_VS_OFF_COMPARISON"
compare_to = "RD_494_OFF"
bins = get_bins(100, 11500, 11400)

rundata = {"RD_494_OFF_HIGH_GAIN": [3830 + i for i in range(6)], "RD_494_OFF_LOW_GAIN": [4011 + i for i in range(5)],
           "RD_494_OFF_MED_GAIN": [4203, 4204]}
plot_name = "RD_494_OFF_GAIN_COMPARISON"
compare_to = "RD_494_OFF_HIGH_GAIN"
bins = get_bins(0, 100,100)

rundata = {"RD_493_ON_HIGH_GAIN": [3819 + i for i in range(10)], "RD_493_ON_LOW_GAIN": [3051 + i for i in range(4)],
           "RD_493_ON_MED_GAIN": [3435 + i for i in range(6)]}
plot_name = "RD_493_ON_GAIN_COMPARISON"
compare_to = "RD_493_ON_HIGH_GAIN"
bins = get_bins(0, 100,100)

rundata = {str(i+1): [4011+i] for i in range(7)}
plot_name = "RD_494_OFF_LOW_GAIN_DAY_COMPARISON"
compare_to = "1"
bins = None

rundata = {str(i+1): [4428+i] for i in range(3)}
plot_name = "RD_495_ON_LOW_GAIN_DAY_COMPARISON"
compare_to = "1"
bins = None


rundata = {"RD_494_ON_HIGH_GAIN": [4428 + i for i in range(2)],
           "RD_494_ON_LOW_GAIN": [4587 + i for i in range(3)]}
plot_name = "RD_494_ON_GAIN_COMPARISON"
compare_to = "RD_494_ON_LOW_GAIN"
bins = get_bins(0, 100,100)

rundata = {str(i+1): [4779+i] for i in range(2)}
plot_name = "RD_495_ON_HIGH_GAIN_DAY_COMPARISON"
compare_to = "1"
bins = None




rundata = {"RD_495_ON_LOW_GAIN": [4428 + i for i in range(2)],
           "RD_495_ON_HIGH_GAIN": [4587 + i for i in range(3)]}
plot_name = "RD_495_ON_GAIN_COMPARISON_lead"
compare_to = "RD_495_ON_HIGH_GAIN"
bins = get_bins(0, 100,100)

rundata = {"RD_495_ON_HIGH_GAIN": [4971 + i for i in range(6)],
           "RD_495_ON_LOW_GAIN": [5163 + i for i in range(9)]}
plot_name = "RD_495_ON_GAIN_COMPARISON_water"
compare_to = "RD_495_ON_HIGH_GAIN"
bins = get_bins(0, 100,100)

rundata = {str(i+1): [5356+i] for i in range(8)}
plot_name = "RD_496_OFF_HIGH_GAIN_DAY_COMPARISON"
compare_to = "1"
bins = None


rundata = {"RD_496_ON_HIGH_GAIN_60BRICKS": [5569 + i for i in range(4)],
           "RD_496_ON_HIGH_GAIN_68BRICKS": [5574 + i for i in range(4)]}
plot_name = "RD_496_ON_WATER_COMPARISON"
compare_to = "RD_496_ON_HIGH_GAIN_68BRICKS"
bins = get_bins(0, 100,100)

rundata = {str(i+1): [5574+i] for i in range(4)}
plot_name = "RD_496_ON_HIGH_GAIN_68BRICKS_DAY_COMPARISON"
compare_to = "1"
bins = get_bins(0, 88,180)
#bins = None

rundata = {"East":"EAST_FACE_18.txt","rxoff":"MIF_BOX_AT_REACTOR_RXOFF", "NE":"CYCLE461_DOWN_FACING_OVERNIGHT.txt"}
plot_name = "NE_vs_E"
compare_to = "rxoff"
bins = get_bins(500, 2800, 2300)

rundata = {"rxon":"MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN.txt","rxoff":"MIF_BOX_AT_REACTOR_RXOFF"}
plot_name = "rxon_vs_off"
compare_to = "rxoff"

#rundata = {"RD_494_ON": [4395 + i for i in range(5)], "RD_494_OFF": [4011 + i for i in range(5)]}
#plot_name = "RD_ON_VS_OFF_COMPARISON"
#compare_to = "RD_494_OFF"
#bins = get_bins(100, 520, 420)

rundata =  {"Russian Doll - RxOn": [i for i in range(3051, 3055)], "Russian Doll - RxOff": [i for i in range(4012, 4018)]}
plot_name = "full_energy_range_RD"
compare_to = "Russian Doll - RxOff"
bins = get_bins(100, 3000, 2900)




outdir = join(os.environ["HFIRBG_ANALYSIS"], "spectrum_plots")

emin = [1000 * i for i in range(12)]
emax = [1000 * (i + 1) for i in range(12)]

emin_plot = 0
rebin = 1


def main():
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    pname = join(outdir, plot_name)
    db = HFIRBG_DB()
    datadir = get_data_dir()
    data = populate_data(rundata, datadir, db)
    combine_runs(data)
    if bins is not None:
        rebin_spectra(data, bins)
    #data = background_subtract(data, "Rxoff", get_bins(100, 9400, 3100))
    #write_spectra(data, datadir, db)
    plot_multi_spectra(data, pname, rebin=rebin, emin=emin_plot, ebars=False)
    plot_subtract_spectra(data, compare_to, pname + "_subtract", rebin=rebin, emin=emin_plot)
    #for i in range(len(emin)):
    #    plot_multi_spectra(data, pname + "_{}".format(i), emin=emin[i], emax=emax[i])
    #    plot_subtract_spectra(data, compare_to, pname + "_subtract_{}".format(i), emin=emin[i], emax=emax[i])


if __name__ == "__main__":
    main()
