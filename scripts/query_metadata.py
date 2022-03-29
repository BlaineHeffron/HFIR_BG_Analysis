import sys
from os.path import dirname, realpath

sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.database.SqliteManager import HFIRBG_DB

rundata = {"Reactor Spectrum": "MIF_BOX_REACTOR_OPTIMIZED_OVERNIGHT_LOWEST_GAIN.txt"}

def main():
    db = HFIRBG_DB()
    for key, fname in rundata.items():
        print("====================================")
        print("****** metadata for {} ******".format(key))
        row = db.retrieve_file_metadata(fname)
        for nm, val in row.items():
            print("{0}: {1}".format(nm,val))

if __name__=="__main__":
    main()

