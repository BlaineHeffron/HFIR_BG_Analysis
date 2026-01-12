import sys
from os.path import dirname, realpath
sys.path.insert(1, dirname(dirname(realpath(__file__))))

from src.database.HFIRBG_DB import HFIRBG_DB
from src.utilities.util import *

rundata = {"original_1": 54, "original_2": 55, "original_cart_xzrot_90": 56,
           "nominal": 57, "EAST CORNER": 58}
#rundata = {"reactor_optimized": "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN.txt",
#           "track_reactor_off_axis": "TRACK_POS55_48"}

#rundata = {"reactor_optimized": "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN.txt",
#           "track_reactor_off_axis": "TRACK_POS55_48"}

rundata = {"PROSPECT_NW": "PROSPECT_DOWN_OVERNIGHT",
           "PROSPECT_NE": "CYCLE461_DOWN_FACING_OVERNIGHT"}

rundata = {"East 1": "EAST_FACE_18.txt", "East 2":"EAST_FACE_1.txt", "Shield Center":"CYCLE461_DOWN_FACING_OVERNIGHT.txt"}
compare_to = "Shield Center"
compare_name = "E_vs_NE_corner"

outdir = join(os.environ["HFIRBG_ANALYSIS"], "spectrum_plots")

def main():
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    pname = join(outdir, compare_name)
    datadir = get_data_dir()
    db = HFIRBG_DB()
    data = populate_data(rundata, datadir, db)
    combine_runs(data)
    plot_multi_spectra(data, pname, rebin=40)
    plot_subtract_spectra(data, compare_to, pname + "_sub", rebin=100)


if __name__ == "__main__":
    main()