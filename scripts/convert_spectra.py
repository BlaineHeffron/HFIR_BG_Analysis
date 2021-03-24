import subprocess
from src.utilities.util import *
import shutil
import ntpath

def convert_to_spe():
    datadir = os.path.expanduser(get_data_dir())
    os.chdir(os.path.abspath(os.path.normpath(os.path.expanduser(datadir))))
    flist = retrieve_file_extension(datadir, ext=".txt")
    for f in flist:
        try:
            spec = retrieve_data(f)
        except IOError as e:
            print("SKIPPING {0}, not a valid cnf file. Error message: {1}".format(f, e))
            continue
        fdir, fname = ntpath.split(f)
        fname = fname.replace(".txt",".spe")
        print("converting  {0} to {1}".format(f,fname))
        write_spe(fname, spec.data)



def main():
    datadir = os.path.expanduser(get_data_dir())
    os.chdir(os.path.abspath(os.path.normpath(os.path.expanduser(datadir))))
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
    convert_to_spe()