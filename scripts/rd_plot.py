import sys
import os
from os.path import dirname, realpath, join
sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.database.HFIRBG_DB import HFIRBG_DB
from src.utilities.util import populate_rd_data, plot_time_series

outdir = join(os.environ["HFIRBG_ANALYSIS"], "russian_doll")

acq_id_map = {5: "low gain", 7: "high gain", 17: "90 keV range", 16: "medium gain"}
Emin = 30
Emax = None

def main():
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    db = HFIRBG_DB()
    rd_data = db.get_rd_files()
    rd_data = populate_rd_data(rd_data, db)
    for shield_id in rd_data.keys():
        for acq_id in rd_data[shield_id].keys():
            shield_info = db.retrieve_shield_info(shield_id)
            name = "{0}, {1}".format(shield_info[0], acq_id_map[acq_id])
            plot_time_series({name: rd_data[shield_id][acq_id]}, outdir, emin=Emin, emax=Emax)


if __name__=="__main__":
    main()

