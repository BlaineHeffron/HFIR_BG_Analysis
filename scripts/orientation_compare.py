from util import *

rundata = {#"original": [54,55], "original_cart_xzrot_90":56,
    "nominal":57, "EAST CORNER": 58, "Rxoff": [i for i in range(30,40)]}
datadir = '/home/bheffron/projects/HFIR_BG/data/'

def main():
    data = populate_data(rundata, datadir)
    combine_runs(data)
    data = background_subtract(data, "Rxoff")
    plot_multi_spectra(data, "nominal_compare_lowgain", rebin=10)
    plot_subtract_spectra(data, "nominal", "nominal_compare_subtract", rebin=100)

if __name__ == "__main__":
    main()
