import sys
from argparse import ArgumentParser
from os.path import dirname, realpath, join
import os

sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.database.HFIRBG_DB import HFIRBG_DB
from src.utilities.util import get_data_dir, populate_data, fit_spectra, combine_runs, populate_data_config, load_pyspec

def read_md_file(path):
    d = {}
    thresh = 5
    with open(path, 'r') as f:
        cur_name = None
        for line in f.readlines():
            if line.strip().startswith("#"):
                cur_name = line.strip()[1:]
                d[cur_name] = [[],[]]
            else:
                data = line.split()
                if len(data) > 3:
                    f = float(data[2])
                    en = float(data[0])
                    if float(data[2]) > thresh:
                        d[cur_name][0].append(en)
                        d[cur_name][1].append(f)
    return d


verify = False

def main():
    arg = ArgumentParser()
    arg.add_argument("basedir", help="path to directory containing GeRD_# directories", type=str)
    args = arg.parse_args()
    if os.path.exists(join(args.basedir, "neutronRD.pyspec")):
        hist = load_pyspec(join(args.basedir, "neutronRD.pyspec"))
    else:
        raise Exception("no neutronRD.pyspec file")
    mydata = read_md_file(join(args.basedir, "ge_ncap.txt"))
    for key in mydata:
        #print(key)
        #for e, f in zip(*mydata[key]):
        #    print("{0} {1}".format(e, f))
        all_energies = mydata[key][0]
        peak_data = fit_spectra({"neut_sim": hist}, all_energies, args.basedir, verify, True, False)
        for e, d in peak_data.items():
            for en, freq in zip(*mydata[key]):
                if en == e:
                    try:
                        if abs(float(d.fit_energy_string()) - en) > 1:
                            print("** BAD FIT ** {0} {1} {2:.2f} {3:.2f} area {4:.2f}".format(key, freq, en, float(d.fit_energy_string()), d.area()[0]))
                            nearby_peaks = []
                            for mykey in mydata:
                                if mykey == key:
                                    continue
                                for myen in mydata[mykey][0]:
                                    if abs(myen - en) < 4:
                                        nearby_peaks.append(myen)
                            print("other nearby peaks are {}".format(nearby_peaks))
                        else:
                            print("{0} {1} {2:.2f} {3:.2f} area {4:.2f}".format(key, freq, en, float(d.fit_energy_string()), d.area()[0]))
                    except Exception as e:
                        print(e)
                        print("{0} {1} {2:.2f} {3:.2f}".format(key, freq, en, d.fit_energy_string()))
                    break


if __name__ == "__main__":
    main()
