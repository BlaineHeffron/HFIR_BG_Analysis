import sys
from os.path import dirname, realpath

sys.path.insert(1, dirname(dirname(realpath(__file__))))

from src.utilities.NumbaFunctions import integrate_lininterp_range
from src.utilities.util import *
from src.database.HFIRBG_DB import HFIRBG_DB
from src.utilities.util import plot_time_series

config = None
start = 4203
end = 4216
rundata = {"cycle494startup": [i for i in range(start, end + 1)]}
start = 4395
end = 4430
rundata["cycle494shutdown_cycle495startup"] = [i for i in range(start, end + 1)]
start = 5163
end = 5184
rundata["cycle495shutdown"] = [i for i in range(start, end + 1)]
start = 5547
end = 5573
rundata["cycle496startup"] = [i for i in range(start, end + 1)]
start = 5574
end = 5608
rundata["cycle496shutdown"] = [i for i in range(start, end + 1)]

Emin = 30  # keV
Emax = None
outdir = join(os.environ["HFIRBG_ANALYSIS"], "time_series")

config = {"detector_coordinates": { "Rx": 21.0, "Rz": 94.5, "Lx": 38.0, "Lz": 90.5, "angle": 46.5, "track":0 },
          "acquisition_settings": {"coarse_gain": 2, "fine_gain": 1.02},
         "max_date": "2021-04-20, 00:00:00", "min_date": "2021-04-11, 00:00:00"}
name = "MIF_to_reactor"

#rundata = {0: [388 + i for i in range(395-388)], 5: [638], 10: [639,640], 20: [641], 40: [642], 60: [643], 80: [644], 95: [645], 100: [747+i for i in range(45)]}
#name = "cycle 491 startup"

def main():
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    datadir = get_data_dir()
    db = HFIRBG_DB()
    if config:
        data = populate_data_config(config, db, comb_runs=False)
        mydata = {name:[]}
        for key in data.keys():
            mydata[name].append(data[key])
        plot_time_series(mydata, outdir, emin=Emin, emax=Emax)
    else:
        data = populate_data(rundata, datadir, db)
        plot_time_series(data, outdir, emin=Emin, emax=Emax)



if __name__ == "__main__":
    main()
