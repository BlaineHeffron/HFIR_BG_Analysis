import subprocess
import os
from util import *
import shutil
import ntpath


def main():
    datadir = get_data_dir()
    os.chdir(datadir)
    flist = retrieve_file_extension(datadir, ext=".CNF")
    for f in flist:
        fdir, fname = ntpath.split(f)
        if ' ' in fname:
            newname = fname.replace(" ","_")
            newf = os.path.join(fdir, newname)
            if not os.path.exists(newf):
                shutil.copyfile(f, newf)

    subprocess.run("cnf2txtall")

if __name__ == "__main__":
    main()