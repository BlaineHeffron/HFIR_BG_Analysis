import sys
from os.path import dirname, realpath

sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.database.HFIRBG_DB import HFIRBG_DB
from src.utilities.util import get_data_dir, populate_data, fit_spectra, combine_runs

#rundata = {"Reactor Spectrum": "MIF_BOX_REACTOR_OPTIMIZED_OVERNIGHT_LOWEST_GAIN.txt"}
#rundata = {"RD 494 low gain": [4395 + i for i in range(4)]}
#rundata = {"MIF": "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN.txt","HB4":"HB4_DOWN_OVERNIGHT_1.txt"}
#rundata = {"HB4":"HB4_DOWN_OVERNIGHT_1.txt"}
#rundata = {"rxoff":"MIF_BOX_AT_REACTOR_RXOFF"}
rundata = {"PROSPECT":"EAST_FACE_1.txt"}
#energy_guesses = [964.082,1112.08,1408.013,778.9006]
#energy_guesses = [723.3,1274]
#energy_guesses = [1085,1112]
#energy_guesses = [1596,1680,1730,1770]
#energy_guesses = [1274, 1298]
#energy_guesses = [1596]
#energy_guesses = [815.7,1596]
#energy_guesses = [6018,6767+511]
#energy_guesses = [6791]
energy_guesses = [610, 768, 1120]
energy_guesses = [1238, 1377]
#energy_guesses = [595.8,867.9,608.4,1489.3]
#energy_guesses = [2614.3, 474,477,480,483]
energy_guesses = [2614.3, 1215, 1238]
energy_guesses = [7631,7645]
verify = True


def main():
    datadir = get_data_dir()
    all_energies = []
    for e in energy_guesses:
        all_energies.append(float(e))
    db = HFIRBG_DB()
    data = populate_data(rundata, datadir, db)
    combine_runs(data)
    peak_data = fit_spectra(data, all_energies, None, verify, False)
    for e, d in peak_data.items():
        print("----------------")
        print("fit for energy guess {}".format(e))
        d.display()


if __name__ == "__main__":
    main()
