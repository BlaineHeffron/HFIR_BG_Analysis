"""
script to fix incorrect runtime for surface thrower event generator
"""

import h5py
import os
from os.path import join
from argparse import ArgumentParser



def get_runtime_from_xml(f, SAname="RDThrower"):
    tree = ET.parse(f)
    root = tree.getroot()
    el = root.find("./AnalysisStep/Run/PrimaryGenerator/" + SAname)
    return float(el.attrib["nAttempts"])/float(str(el.attrib["s_area"]).split(" ")[0])/1.e6

def change_xml_attr(f, value, path="./AnalysisStep/Run/PrimaryGenerator", name="time", newname=None):
    """Change the attributes with name 'name' at path 'path' of xml file 'f' to value 'value'"""
    tree = ET.parse(f)
    root = tree.getroot()
    try:
        el = root.find(path)
        el.attrib[name] = value
        if newname is not None:
            tree.write(newname)
        else:
            tree.write(f)
    except Exception as e:
        print(e)

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