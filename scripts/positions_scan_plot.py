import sys
from os.path import dirname, realpath
import matplotlib.pyplot as plt

sys.path.insert(1, dirname(dirname(realpath(__file__))))
import os
from os.path import join
from src.database.HFIRBG_DB import HFIRBG_DB
from src.utilities.PlotUtils import scatter_plot, HFIR_scatter_plot
from src.utilities.util import retrieve_position_scans, get_bins, get_data_dir, \
    retrieve_spectra, retrieve_file_extension
from src.database.CartScanFiles import CartScanFiles, parse_orientation_key

energy_cutoffs = [50, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11400]
#energy_cutoffs = [50, 11400]
outdir = join(os.environ["HFIRBG_ANALYSIS"], "position_scan")


def parse_coord(s):
    data = s.split(",")
    x = float(data[0].strip())
    y = float(data[1].strip())
    return x, y


def get_x_y(r, l):
    rx, ry = parse_coord(r)
    lx, ly = parse_coord(l)
    return (rx + lx) / 2., (ry + ly) / 2.


def file_scan_main():
    datadir = get_data_dir()
    db = HFIRBG_DB()
    fs = retrieve_file_extension(datadir, ".txt")
    position_metadata = retrieve_position_scans()  # r, l, angle, live, fname
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
    fig = scatter_plot(x, y, rates_to_3, "z", "x", "rate [hz]", "Down Facing Scan, 0.1 to 3.0 MeV", xmin=40, ymin=0,
                       xmax=420, ymax=160, invert_y=True)
    plt.savefig("{}.png".format(name + "_to_3.png"))
    plt.close(fig)
    fig = scatter_plot(x, y, rates_3_up, "z", "x", "rate [hz]", "Down Facing Scan, 3.0 to 11.5 MeV", xmin=40, ymin=0,
                       xmax=420, ymax=160, invert_y=True)
    plt.savefig("{}.png".format(name + "_3_up.png"))
    plt.close(fig)


def plot_top_down_rates(scan_spec, energy_ranges, labels):
    zero_data = []
    for key, val in scan_spec.items():
        angle, phi = parse_orientation_key(key)
        if angle == 0:
            zero_data += val
    for key, val in scan_spec.items():
        x = []
        y = []
        rates = []
        angle, phi = parse_orientation_key(key)
        if angle == 0:
            continue
        for j, data in enumerate(val):
            spec, coo = data
            x.append(coo[1])
            y.append(coo[0])
            for i in range(len(energy_ranges)):
                if j == 0:
                    rates.append([])
                r, dr = spec.integrate(energy_ranges[i][0], energy_ranges[i][1], True)
                if r > 1:
                    print("abnormally large rate for file {}".format(spec.fname))
                rates[i].append(r)
        for i in range(len(energy_ranges)):
            if len(rates[i]) < 3:
                continue
            # fig = scatter_plot(x, y, rates[i], "z", "x", "rate [hz/keV]", "det angle = {0}, cart angle = {1}, {2}".format(angle, phi, labels[i]), xmin=40, ymin=0, xmax=420, ymax=160, invert_y=True)
            fig = HFIR_scatter_plot(x, y, rates[i], "z [in]", "x [in]", "rate [hz/keV]",
                                    "det angle = {0}, cart angle = {1}, {2}".format(angle, phi, labels[i]),
                                    invert_y=True)
            plot_name = "det_{0}_cart_{1}_{2}_to_{3}.png".format(angle, phi, energy_ranges[i][0], energy_ranges[i][1])
            plt.savefig(join(outdir, plot_name), bbox_inches='tight')
            plt.close(fig)
    x = []
    y = []
    rates = []
    for j, data in enumerate(zero_data):
        spec, coo = data
        x.append(coo[1])
        y.append(coo[0])
        for i in range(len(energy_ranges)):
            if j == 0:
                rates.append([])
            r, dr = spec.integrate(energy_ranges[i][0], energy_ranges[i][1], True)
            if r > 1:
                print("abnormally large rate for file {}".format(spec.fname))
            rates[i].append(r)
    for i in range(len(energy_ranges)):
        if len(rates[i]) < 3:
            continue
        # fig = scatter_plot(x, y, rates[i], "z", "x", "rate [hz/keV]", "det angle = {0}, cart angle = {1}, {2}".format(angle, phi, labels[i]), xmin=40, ymin=0, xmax=420, ymax=160, invert_y=True)
        fig = HFIR_scatter_plot(x, y, rates[i], "z [in]", "x [in]", "rate [hz/keV]",
                                "down facing, {}".format(labels[i]),
                                invert_y=True)
        plot_name = "down_facing_{0}_to_{1}.png".format(energy_ranges[i][0], energy_ranges[i][1])
        plt.savefig(join(outdir, plot_name), bbox_inches='tight')
        plt.close(fig)



def plot_track_rates(scan_spec, energy_ranges):
    x = []
    y = []
    rates = []
    for i in range(len(energy_ranges)):
        rates.append([])
        for key, val in scan_spec.items():
            angle, phi = parse_orientation_key(key)
            for j, data in enumerate(val):
                spec, coo = data
                if i == 0:
                    x.append(coo[1])
                    y.append(angle)
                r, dr = spec.integrate(energy_ranges[i][0], energy_ranges[i][1], True)
                if r > 1:
                    print("abnormally large rate for file {}".format(spec.fname))
                rates[i].append(r)
    for i in range(len(energy_ranges)):
        if len(rates[i]) < 3:
            continue
        # fig = scatter_plot(x, y, rates[i], "z", "x", "rate [hz/keV]", "det angle = {0}, cart angle = {1}, {2}".format(angle, phi, labels[i]), xmin=40, ymin=0, xmax=420, ymax=160, invert_y=True)
        fig = scatter_plot(x, y, rates[i], "z [in]", "angle [deg]", "rate [hz/keV]",
                           "{0} - {1} keV".format(energy_ranges[i][0], energy_ranges[i][1]), invert_y=False)
        plot_name = "track_scan_{0}_to_{1}.png".format(energy_ranges[i][0], energy_ranges[i][1])
        plt.savefig(join(outdir, plot_name))
        plt.close(fig)


def main():
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    energy_ranges = []
    labels = []
    for i in range(len(energy_cutoffs) - 1):
        energy_ranges.append([energy_cutoffs[i], energy_cutoffs[i + 1]])
        labels.append("{0} - {1} keV".format(energy_cutoffs[i], energy_cutoffs[i + 1]))
    #energy_ranges.append([energy_cutoffs[0], energy_cutoffs[-1]])
    #labels.append("{0} - {1} keV".format(energy_cutoffs[0], energy_cutoffs[-1]))
    db = CartScanFiles()
    scan_spec = db.retrieve_position_spectra(11400)
    plot_top_down_rates(scan_spec, energy_ranges, labels)
    del scan_spec
    track_spec = db.retrieve_track_spectra()
    plot_track_rates(track_spec, energy_ranges)


if __name__ == "__main__":
    main()
