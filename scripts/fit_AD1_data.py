import sys
import os
from os.path import dirname, realpath, join

sys.path.insert(1, dirname(dirname(realpath(__file__))))

from src.utilities.util import get_spec_from_root, write_rows_csv
from argparse import ArgumentParser
from src.utilities.ROOT_style import ROOT_style
from src.analysis.AD1_Fitter import fit_spec

en = [5, 6.5, 8.5, 10.5, 12]

outdir = join(os.environ["HFIRBG_ANALYSIS"], "AD1_fit")

def main():
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    arg = ArgumentParser()
    ROOT_style()
    arg.add_argument("f", help="path to root file containing AD1 hist", type=str)
    arg.add_argument("name", help="name for output plots", type=str)
    arg.add_argument("--path","-p", help="path in root file to spectrum", type=str)
    args = arg.parse_args()
    if args.path:
        spec = get_spec_from_root(args.f, "SingleSegmentsIoniPlugin/{}".format(args.path), "runtime", False)
    else:
        spec = get_spec_from_root(args.f, "SingleSegmentsIoniPlugin/hSumE", "runtime", False)
    outpath = join(outdir, args.name)
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    labels = ["guas height", "erfc height", "centroid", "std dev", "erfc loc", "A", "B"]
    mylabels = []
    for l in labels:
        mylabels.append(l)
        mylabels.append(l + " err")
    mylabels.append("chisqr")
    mylabels.append("ndf")
    mydata = []
    for i in range(len(en) - 1):
        if args.path:
            par, err, cs, ndf = fit_spec(spec, en[i], en[i + 1], outpath, name=args.path)
        else:
            par, err, cs, ndf = fit_spec(spec, en[i], en[i + 1], outpath)
        mydata.append([])
        for p, e in zip(par, err):
            mydata[-1].append(p)
            mydata[-1].append(e)
        mydata[-1].append(cs)
        mydata[-1].append(ndf)
        #print("parameters for energy range {0} - {1}".format(en[i], en[i + 1]))
        #print(par)
        #print('errors:')
        #print(err)
        print("centroid: {0:.3f} ~ {3:.3f}, std dev: {1:.3f} ~ {4:.3f}, erfc location: {2:.3f} ~ {5:.3f}, guass height: {8:.0f}, erfc height: {9:.0f}, chisqr / ndf: {6:.0f} / {7},".format(par[2], par[3], par[4], err[2], err[3], err[4], cs, ndf, par[0], par[1]))
    if args.path:
        outf = join(outpath, "{}_fit.csv".format(args.path))
    else:
        outf = join(outpath, "hSumE_fit.csv")
    f = open(outf, 'w')
    write_rows_csv(f, mylabels, mydata, delimiter=',')


if __name__ == "__main__":
    main()
