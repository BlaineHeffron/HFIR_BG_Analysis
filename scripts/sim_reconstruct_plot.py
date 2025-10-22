import sys
from os.path import dirname, realpath
sys.path.insert(1, dirname(dirname(realpath(__file__))))
from argparse import ArgumentParser
from ROOT import TFile
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import os

nIncrements = 10

def hist_to_arrays(hist):
    """Convert ROOT histogram to numpy arrays"""
    n_bins = hist.GetNbinsX()
    x = np.array([hist.GetBinCenter(i+1) for i in range(n_bins)])
    y = np.array([hist.GetBinContent(i+1) for i in range(n_bins)])
    yerr = np.array([hist.GetBinError(i+1) for i in range(n_bins)])
    edges = np.array([hist.GetBinLowEdge(i+1) for i in range(n_bins+1)])
    return x, y, yerr, edges

def chisqr(obs_hist, exp_hist):
    chisqr = 0
    for i in range(exp_hist.GetNbinsX()):
        x = exp_hist.GetBinCenter(i+1)
        bin_num = obs_hist.GetXaxis().FindBin(x)
        if obs_hist.GetBinError(bin_num) == 0:
            chisqr += exp_hist.GetBinContent(i+1)
        else:
            chisqr += (obs_hist.GetBinContent(bin_num) - exp_hist.GetBinContent(i+1))**2 / obs_hist.GetBinError(bin_num)**2
    return chisqr

def main():
    arg = ArgumentParser()
    arg.add_argument("f",help="path to root file containing reconstructed histogram", type=str)
    arg.add_argument("foriginal",help="path to root file containing original histogram", type=str)
    arg.add_argument("outpath",help="path to output graphs", type=str)
    args = arg.parse_args()
    
    f = TFile(args.f,"READ")
    hist_recon = f.Get("FoldedBack")
    hist_measure = f.Get("Measured")
    hist_en = f.Get("UnfoldedEnergy")
    hist_unf = f.Get("Unfolded")
    n_vectors = hist_unf.GetNbinsX()
    
    # Convert to numpy arrays
    x_recon, y_recon, yerr_recon, edges_recon = hist_to_arrays(hist_recon)
    x_measure, y_measure, yerr_measure, edges_measure = hist_to_arrays(hist_measure)
    x_en, y_en, yerr_en, edges_en = hist_to_arrays(hist_en)
    
    print("chisqr is {}".format(chisqr(hist_measure, hist_recon)))
    print("ndf = {}".format(hist_measure.GetNbinsX() - n_vectors))
    
    # Determine x-axis range
    xlow = max(edges_recon[0], edges_measure[0])
    xhigh = min(edges_recon[-1], edges_measure[-1])
    
    # Create PDF
    from matplotlib.backends.backend_pdf import PdfPages
    pdf_path = os.path.join(args.outpath, "unfoldplots.pdf")
    
    with PdfPages(pdf_path) as pdf:
        # Main plot with measured and reconstructed
        fig, ax = plt.subplots(figsize=(10, 8))
        # Use very distinct colors with RGB values
        ax.step(x_measure, y_measure, where='mid', label='Measured',
                color='#FF0000', linewidth=3, linestyle='-')  # Red
        ax.step(x_recon, y_recon, where='mid', label='Reconstructed',
                color='#00FF00', linewidth=3, linestyle='--')  # Green
        ax.set_xlim(xlow, xhigh)
        ax.set_yscale('log')
        ax.set_xlabel('Energy', fontsize=14)
        ax.set_ylabel('Counts', fontsize=14)
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig, dpi=150)
        plt.close()
        
        # Unfolded energy plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.step(x_en, y_en, where='mid', label='Unfolded Energy',
                color='#0000FF', linewidth=3)  # Blue
        ax.set_xlim(xlow, xhigh)
        ax.set_yscale('log')
        ax.set_xlabel('Energy', fontsize=14)
        ax.set_ylabel('Counts', fontsize=14)
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig, dpi=150)
        plt.close()
        
        # Zoomed plots
        width = (xhigh - xlow) / nIncrements
        for i in range(nIncrements):
            xlow_zoom = xlow + i*width
            xhigh_zoom = xlow + (i+1)*width
            
            # Measured vs reconstructed zoomed
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.step(x_measure, y_measure, where='mid', label='Measured',
                    color='#FF0000', linewidth=3, linestyle='-')  # Red
            ax.step(x_recon, y_recon, where='mid', label='Reconstructed',
                    color='#00FF00', linewidth=3, linestyle='--')  # Green
            ax.set_xlim(xlow_zoom, xhigh_zoom)
            ax.set_yscale('log')
            ax.set_xlabel('Energy', fontsize=14)
            ax.set_ylabel('Counts', fontsize=14)
            ax.legend(loc='upper right', fontsize=12)
            ax.grid(True, alpha=0.3)
            pdf.savefig(fig, dpi=150)
            plt.close()
            
            # Unfolded energy zoomed
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.step(x_en, y_en, where='mid', label='Unfolded Energy',
                    color='#0000FF', linewidth=3)  # Blue
            ax.set_xlim(xlow_zoom, xhigh_zoom)
            ax.set_yscale('log')
            ax.set_xlabel('Energy', fontsize=14)
            ax.set_ylabel('Counts', fontsize=14)
            ax.legend(loc='upper right', fontsize=12)
            ax.grid(True, alpha=0.3)
            pdf.savefig(fig, dpi=150)
            plt.close()
    
    print(f"Saved plots to {pdf_path}")

if __name__ == "__main__":
    main()