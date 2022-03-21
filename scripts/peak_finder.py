import sys
from os.path import dirname, realpath

sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.database.SqliteManager import HFIRBG_DB
from src.utilities.util import get_data_dir, populate_data, fit_spectra

rundata = {"Reactor Spectrum": "MIF_BOX_REACTOR_OPTIMIZED_OVERNIGHT_LOWEST_GAIN.txt"}
energy_guesses = [1274, 1294, 1300, 1407]
verify = True


def main():
    datadir = get_data_dir()
    all_energies = []
    for e in energy_guesses:
        all_energies.append(float(e))
    db = HFIRBG_DB()
    data = populate_data(rundata, datadir, db)
    peak_data = fit_spectra(data, all_energies, None, verify, True, False)
    for e, d in peak_data.items():
        print("----------------")
        print("fit for energy guess {}".format(e))
        d.display()


if __name__ == "__main__":
    main()
