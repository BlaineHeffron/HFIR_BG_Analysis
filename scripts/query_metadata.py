import sys
from os.path import dirname, realpath

sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.database.HFIRBG_DB import HFIRBG_DB
from src.utilities.util import dt_to_string


rundata = {"Reactor Spectrum": "MIF_BOX_REACTOR_OPTIMIZED_OVERNIGHT_LOWEST_GAIN.txt"}
rundata = {"Reactor Spectrum": 2915}
#names = ["HB3_DOWN_{}".format(i) for i in range(2,28)]
#names = ["HB4_DOWN_{}".format(i) for i in range(26,29)]
#rundata = {n : n for n in names}
#rundata = {"SE_TEST_2": "SE_TEST_2"}
rundata = {"PROSPECT_DOWN_OVERNIGHT": "PROSPECT_DOWN_OVERNIGHT"}
#rundata = {"north_face": "CYCLE461_DOWN_FACING_OVERNIGHT"}

def main():
    db = HFIRBG_DB()
    for key, fname in rundata.items():
        print("====================================")
        print("****** metadata for {} ******".format(key))
        row = db.retrieve_file_metadata(fname)
        for nm, val in row.items():
            if nm == "start_time":
                print("start time: {0} = {1}".format(val, dt_to_string(val) ))
            print("{0}: {1}".format(nm,val))

if __name__=="__main__":
    main()

