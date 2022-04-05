import sys
from os.path import dirname, realpath
sys.path.insert(1, dirname(dirname(realpath(__file__))))
from argparse import ArgumentParser
from ROOT import TFile, TCanvas, TLegend
from src.utilities.ROOT_style import ROOT_style
import os
from src.utilities.util import scale_to_bin_width

nIncrements = 10

def chisqr(obs_hist, exp_hist):
    chisqr = 0
    for i in range(obs_hist.GetNbinsX()):
        if exp_hist.GetBinContent(i+1) == 0:
            if obs_hist.GetBinContent(i+1) == 0:
                chisqr += 0
            else:
                chisqr += obs_hist.GetBinContent(i+1)
        else:
            chisqr += (obs_hist.GetBinContent(i+1) - exp_hist.GetBinContent(i+1))**2 / exp_hist.GetBinContent(i+1)
    return chisqr

def main():
    arg = ArgumentParser()
    ROOT_style()
    arg.add_argument("f",help="path to root file containing reconstructed histogram", type=str)
    arg.add_argument("foriginal",help="path to root file containing original histogram", type=str)
    arg.add_argument("outpath",help="path to output graphs", type=str)
    args = arg.parse_args()
    f = TFile(args.f,"READ")
    hist_recon = f.Get("FoldedBack")
    f2 = TFile(args.foriginal,"READ")
    hist_measure = f2.Get("GeDataHist")
    hist_live = f2.Get("LiveTime")[0]
    print(hist_recon)
    print(hist_measure)
    hist_measure.Scale(1./hist_live)
    scale_to_bin_width(hist_measure)
    #hist_copy = hist_measure.Clone()
    #hist_copy.Add(hist_recon,-1)
    print("chisqr is {}".format(chisqr(hist_measure, hist_recon)))
    canvas = TCanvas("canvas")
    canvas.cd()
    canvas.SetLogy(True)
    canvas.Print(os.path.join(args.outpath, "unfoldplots.pdf["))
    xlow = hist_recon.GetXaxis().GetBinLowEdge(1)
    xhigh = hist_recon.GetXaxis().GetBinLowEdge(hist_recon.GetNbinsX()+1)
    xlow2 = hist_measure.GetXaxis().GetBinLowEdge(1)
    xhigh2 = hist_measure.GetXaxis().GetBinLowEdge(hist_measure.GetNbinsX()+1)
    if xhigh > xhigh2:
        xhigh = xhigh2
    if xlow < xlow2:
        xlow = xlow2
    hist_recon.Rebin(3)
    hist_measure.Rebin(3)
    hist_recon.GetXaxis().SetRangeUser(xlow, xhigh)
    hist_measure.GetXaxis().SetRangeUser(xlow, xhigh)
    hist_measure.Draw("L")
    hist_recon.Draw("L SAME")
    legend = TLegend(0.7, 0.85, .9, 0.95)
    legend.AddEntry(hist_measure, "Measured", "f")
    legend.AddEntry(hist_recon, "reconstructed", "f")
    legend.Draw()
    canvas.Print(os.path.join(args.outpath, "unfoldplots.pdf"))
    width = (xhigh - xlow) / nIncrements
    for i in range(nIncrements):
        hist_recon.GetXaxis().SetRangeUser(xlow + i*width, xlow + (i+1)*width)
        hist_measure.GetXaxis().SetRangeUser(xlow + i*width, xlow + (i+1)*width)
        hist_measure.Draw("HIST C")
        hist_recon.Draw("HIST SAME C")
        legend.Draw()
        canvas.Print(os.path.join(args.outpath, "unfoldplots.pdf"))

    #canvas.Print(os.path.join(args.outpath, "unfoldplots.pdf"))
    #hist_copy.Draw()
    canvas.Print(os.path.join(args.outpath, "unfoldplots.pdf]"))

if __name__ == "__main__":
    main()