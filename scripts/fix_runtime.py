"""
script to fix incorrect runtime for surface thrower event generator
"""

import sys
import h5py
import os
from os.path import dirname, realpath, join
from argparse import ArgumentParser
sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.utilities.util import get_runtime_from_xml, change_xml_attr



def main():
    arg = ArgumentParser()
    arg.add_argument("basedir", help="path to directory containing GeRD_# directories", type=str)
    args = arg.parse_args()
    for f in os.listdir(args.basedir):
        if f.endswith(".xml"):
            name = f[0:-4]
            rt = get_runtime_from_xml(f)
            change_xml_attr(f, "{:.3f} s ".format(rt))
            with h5py.File(join(args.basedir, name), "r+") as f:
                f['ioni'].attrs['runtime'] = rt


if __name__ == "__main__":
    main()