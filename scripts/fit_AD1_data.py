import sys
import os
from os.path import dirname, realpath, join


sys.path.insert(1, dirname(dirname(realpath(__file__))))

from src.analysis.Spectrum import SubtractSpectrum
from src.utilities.FitUtils import minimize_diff
from src.utilities.util import get_spec_from_root, write_rows_csv, set_indices, plot_multi_spectra, get_bins
from argparse import ArgumentParser
from src.utilities.ROOT_style import ROOT_style
from src.analysis.AD1_Fitter import fit_spec

en = [5, 6.5, 8.5, 10, 12]

outdir = join(os.environ["HFIRBG_ANALYSIS"], "AD1_fit")
bin_edges = get_bins(0.1, 12, 12*4)

def main():
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    arg = ArgumentParser()
    ROOT_style()
    arg.add_argument("f", help="path to root file containing AD1 hist", type=str)
    arg.add_argument("name", help="name for output plots", type=str)
    arg.add_argument("--path","-p", help="path in root file to spectrum", type=str)
    arg.add_argument("--bg","-b", help="path to background root file", type=str)
    arg.add_argument("--px", help="use projection x on data root file", action="store_true")
    arg.add_argument("--nopluginpath", help="dont use SingleSegmentsIoniPlugin/ as a prefix to path", action="store_true")
    arg.add_argument("--sim","-s", help="path to sim root file for comparison", type=str)
    args = arg.parse_args()
    spec_path = "SingleSegmentsIoniPlugin/hSumE"
    if args.path:
        if args.nopluginpath:
            spec_path = args.path
        else:
            spec_path = "SingleSegmentsIoniPlugin/{}".format(args.path)
    spec = get_spec_from_root(args.f, spec_path, "runtime", False, projectionX=args.px, has_live=not args.nopluginpath)
    if args.bg:
        bgspec = get_spec_from_root(args.bg, spec_path, "runtime", False, projectionX=args.px, has_live=not args.nopluginpath)
        spec = SubtractSpectrum(spec, bgspec)
    simspec = None
    if args.sim:
        simspec = get_spec_from_root(args.sim, "SingleSegmentsIoniPlugin/hSumE", "runtime", False)
        start_index, end_index = set_indices(0, 0, 0.1, 10, simspec)
        simspec.rebin(bin_edges)
        spec.rebin(bin_edges)
        sim = simspec.get_normalized_hist()[start_index:end_index]
        data = spec.get_normalized_hist()[start_index:end_index]
        data_err = spec.get_normalized_err()[start_index:end_index]/100.
        scale = minimize_diff(sim, data, data_err)
        simspec.scale_hist(scale, True)
        print("scale is {}".format(scale))
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
            par, err, cs, ndf = fit_spec(spec, en[i], en[i + 1], outpath, name=args.path, use_hist=(args.bg is not None))
        else:
            par, err, cs, ndf = fit_spec(spec, en[i], en[i + 1], outpath, use_hist=(args.bg is not None))
        mydata.append([])
        for p, e in zip(par, err):
            mydata[-1].append(p)
            mydata[-1].append(e)
        mydata[-1].append(cs)
        mydata[-1].append(ndf)
        print("centroid: {0:.3f} ~ {3:.3f}, std dev: {1:.3f} ~ {4:.3f}, erfc location: {2:.3f} ~ {5:.3f}, guass height: {8:.0f}, erfc height: {9:.0f}, chisqr / ndf: {6:.0f} / {7},".format(par[2], par[3], par[4], err[2], err[3], err[4], cs, ndf, par[0], par[1]))
    if args.path:
        outf = join(outpath, "{}_fit.csv".format(args.path))
    else:
        outf = join(outpath, "hSumE_fit.csv")
    f = open(outf, 'w')
    write_rows_csv(f, mylabels, mydata, delimiter=',')
    if simspec is not None:
        if args.path:
            plot_multi_spectra({"sim": simspec, "data": spec}, join(outpath, "{}sim_scaled_data_AD1_comparison".format(args.path)), emin=0.1, emax=12, ebars=False)
        else:
            plot_multi_spectra({"sim": simspec, "data": spec}, join(outpath, "{}_sim_scaled_data_AD1_comparison".format("hSumE")), emin=0.1, emax=12, ebars=False)

if __name__ == "__main__":
    main()
