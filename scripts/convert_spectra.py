import sys
from os.path import dirname, realpath

sys.path.insert(1, dirname(dirname(realpath(__file__))))

from src.database.HFIRBG_DB import HFIRBG_DB
import subprocess
from src.utilities.util import *
import shutil
import ntpath
from ROOT import TFile


def convert_to_spe(db):
    datadir = os.path.expanduser(get_data_dir())
    flist = retrieve_file_extension(datadir, ext=".txt")
    for f in flist:
        try:
            spec = retrieve_data(f, db)
        except IOError as e:
            print("SKIPPING {0}, not a valid cnf file. Error message: {1}".format(f, e))
            continue
        fdir, fname = ntpath.split(f)
        fname = fname.replace(".txt", ".spe")
        print("converting {0} to {1}".format(f, fname))
        xmax = round(spec.A0 + spec.A1 * spec.data.shape[0] + spec.A1 / 2.)
        # rebin to 0.5 keV bins
        # spec.rebin(get_bins(0, xmax, int(xmax*2)))
        write_spe(join(fdir, fname), spec.data)


def convert_to_root(db):
    datadir = os.path.expanduser(get_data_dir())
    flist = retrieve_file_extension(datadir, ext=".txt")
    for f in flist:
        try:
            spec = retrieve_data(f, db)
        except IOError as e:
            print("SKIPPING {0}, not a valid cnf file. Error message: {1}".format(f, e))
            continue
        fdir, fname = ntpath.split(f)
        fname = fname.replace(".txt", ".root")
        histname = fname.replace(".root", "")
        if os.path.exists(join(fdir, fname)):
            print("{} already exists, skipping", fname)
        print("converting {0} to {1}".format(f, fname))
        write_root(spec, histname, join(fdir, fname))


def main():
    datadir = os.path.expanduser(get_data_dir())
    os.chdir(os.path.abspath(os.path.normpath(os.path.expanduser(datadir))))
    flist = retrieve_file_extension(datadir, ext=".CNF")
    for f in flist:
        fdir, fname = ntpath.split(f)
        if ' ' in fname:
            newname = fname.replace(" ", "_")
            newf = os.path.join(fdir, newname)
            if not os.path.exists(newf):
                shutil.copyfile(f, newf)

    subprocess.run("cnf2txtall")


if __name__ == "__main__":
    main()
    db = HFIRBG_DB()
    #convert_to_spe(db)
    #convert_to_root(db)
