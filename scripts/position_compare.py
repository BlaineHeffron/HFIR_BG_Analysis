from util import *

rundata = {#"original_1": 54, "original_2": 55, "original_cart_xzrot_90":56,
           "nominal":57, "EAST CORNER": 58}
datadir = '/home/bheffron/projects/HFIR_BG/data/'

def main():
    data = populate_data(rundata, datadir)
    plot_multi_spectra(data, "nominal_compare_lowgain", rebin=10)
    plot_subtract_spectra(data, "nominal", "nominal_compare_subtract", rebin=100)

if __name__ == "__main__":
    main()