import os
import sys
from os.path import dirname, realpath
sys.path.insert(1, dirname(dirname(realpath(__file__))))

from src.database.SqliteManager import get_db_dir
from src.utilities.util import get_json, read_csv_list_of_tuples, write_json


def main():
    dbdir = get_db_dir()
    runsfile = get_json(os.path.join(dbdir, "runs.json"))
    coords = read_csv_list_of_tuples(os.path.join(dbdir,"detector_coordinates.csv"))
    coord_dict = {}
    for c in coords:
        coord_dict[int(c[0])] = [float(coo) for coo in c[1:]]
    for run in runsfile["runs"]:
        run["detector_coordinates"] = coord_dict[run["detector_coordinates"]]
    write_json(os.path.join(dbdir,"runs_new.json"),runsfile, prettyprint=True)


if __name__ == "__main__":
    main()
