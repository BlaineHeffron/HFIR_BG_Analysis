import sys
from os.path import dirname, realpath
sys.path.insert(1, dirname(dirname(realpath(__file__))))
from src.database.HFIRBG_DB import HFIRBG_DB
from src.utilities.PlotUtils import scatter_plot
from src.utilities.util import retrieve_position_scans, get_bins, get_data_dir, \
    retrieve_spectra, retrieve_file_extension
import matplotlib.pyplot as plt


def parse_coord(s):
    data = s.split(",")
    x = float(data[0].strip())
    y = float(data[1].strip())
    return x, y

def get_x_y(r, l):
    rx, ry = parse_coord(r)
    lx, ly = parse_coord(l)
    #todo add offset based on angle
    return (rx + lx) / 2. , (ry + ly) / 2.


def file_scan_main():
    datadir = get_data_dir()
    db = HFIRBG_DB()
    fs = retrieve_file_extension(datadir, ".txt")
    position_metadata = retrieve_position_scans() # r, l, angle, fname
    name = "down_facing_scan"
    rates_to_3 = []
    rates_3_up = []
    x = []
    y = []
    bins_low = get_bins(100, 3000, 290)
    bins_up = get_bins(3000, 11500, 850)
    for data in position_metadata:
        if float(data[2]) == 0:
            yc, xc = get_x_y(data[0], data[1])
            spec = retrieve_spectra(data[-1], fs, db)
            if spec is None:
                print(data[-1])
                continue
            spec.rebin(bins_low)
            rates_to_3.append(spec.sum_hist_rate())
            spec.rebin(bins_up)
            rates_3_up.append(spec.sum_hist_rate())
            x.append(xc)
            y.append(yc)
    fig = scatter_plot(x, y, rates_to_3, "z", "x", "rate [hz]", "Down Facing Scan, 0.1 to 3.0 MeV", xmin=40, ymin=0, xmax=420, ymax=160, invert_y=True)
    plt.savefig("{}.png".format(name + "_to_3.png"))
    plt.close(fig)
    fig = scatter_plot(x, y, rates_3_up, "z", "x", "rate [hz]", "Down Facing Scan, 3.0 to 11.5 MeV", xmin=40, ymin=0, xmax=420, ymax=160, invert_y=True)
    plt.savefig("{}.png".format(name + "_3_up.png"))
    plt.close(fig)

def main():
    db = HFIRBG_DB()




if __name__ == "__main__":
    main()