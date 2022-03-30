import sys
from os.path import dirname, realpath

sys.path.insert(1, dirname(dirname(realpath(__file__))))

from src.utilities.NumbaFunctions import integrate_lininterp_range
from src.utilities.util import *
from src.database.HFIRBG_DB import HFIRBG_DB
from src.utilities.util import plot_time_series

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


def main():
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    datadir = get_data_dir()
    db = HFIRBG_DB()
    data = populate_data(rundata, datadir, db)
    plot_time_series(data, outdir, emin=Emin, emax=Emax)



if __name__ == "__main__":
    main()
