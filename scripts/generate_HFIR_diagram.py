import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.ticker import AutoMinorLocator
from math import pi
import numpy as np
import os

outdir = os.path.join(os.environ["HFIRBG_ANALYSIS"], "diagrams")


def HFIR_diagram():
    xmin = -5
    ymin = -160
    xmax = 300
    ymax = 180
    corez = 141.15
    corex = -148.5
    corer = 8.66 #in
    # rcParams.update({'font.size': 14})
    ratio = abs((ymax - ymin) / (xmax - xmin))# * 4. / 5
    fig = plt.figure(figsize=(7, 7 * ratio))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel("z [in]")
    ax1.set_ylabel("x [in]")
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymax, ymin)
    #core
    ax1.add_patch(Circle((corez, corex), corer,
                         edgecolor='black',
                         fill=False,
                         lw=2))
    # HB4
    HB4line = [(240.56 + np.tan(30 * pi / 180) * x, x) for x in np.linspace(corex -10, ymax, 20)]
    # HB3
    HB3line = [(67.9 - np.tan(30 * pi / 180) * x, x) for x in np.linspace(corex +10, ymax, 20)]
    # "shutter drive rod sleeve" coming off hb3
    # drline = [(60.35 + np.sin(30*pi/180)*x, 13.0 + np.cos(30*pi/180)*x) for x in range(0, 113+10, 12)]
    ax1.plot([a[0] for a in HB4line], [a[1] for a in HB4line], linestyle='dashed', color='grey')
    ax1.plot([a[0] + 4 for a in HB4line], [a[1] for a in HB4line], linestyle='dashed', color='grey')
    ax1.plot([a[0] for a in HB3line], [a[1] for a in HB3line], linestyle='dashed', color='grey')
    ax1.plot([a[0] + 4 for a in HB3line], [a[1] for a in HB3line], linestyle='dashed', color='grey')
    ax1.text(90, -80, 'HB3', fontsize=15, color='black')
    ax1.text(190, -100, 'HB4', fontsize=15, color='black')
    ax1.text(50, -145, 'Reactor Core', fontsize=15, color='black')
    # ax1.plot([a[0] for a in drline], [a[1] for a in drline], linestyle='dashed', color='grey')
    # ax1.plot([a[0] + 4 for a in drline], [a[1] for a in drline], linestyle='dashed', color='grey')
    # next lets plot the monolith, passes through z = 141.5, x = 111, .1293 x/z
    monolith_line1 = [(141.5 + a, 111 - a * .1293) for a in [-44.7, 67.1]]
    monolith_line2 = [(monolith_line1[0][0] - a, monolith_line1[0][1] - a * .43) for a in [0, 100]]
    monolith_line3 = [(monolith_line1[1][0] + a, monolith_line1[1][1] - a * .58) for a in [0, 150]]
    ax1.plot([a[0] for a in monolith_line1], [a[1] for a in monolith_line1], linestyle='solid', color='grey')
    ax1.plot([a[0] for a in monolith_line2], [a[1] for a in monolith_line2], linestyle='solid', color='grey')
    ax1.plot([a[0] for a in monolith_line3], [a[1] for a in monolith_line3], linestyle='solid', color='grey')
    ax1.text(30, 130, 'Monolith Boundary', fontsize=15, color='black')
    #pool wall
    ax1.plot([0,xmax],[0, 0], linestyle='solid', color='black')
    ax1.plot([0,0],[0, -200], linestyle='solid', color='black')
    ax1.text(110, -3, 'Reactor Pool Wall', fontsize=15, color='black')
    # add box for prospect
    ax1.add_patch(Rectangle((165, 128), 46.25, -83.4,
                            edgecolor='cornflowerblue',
                            fill=False,
                            lw=2))
    ax1.text(99, 95, 'PROSPECT\nScintillator\nVolume', fontsize=15, color='black')
    # lead shield wall
    ax1.add_patch(Rectangle((125, 21.5), 286.5 - 155, -14,
                            edgecolor='black',
                            facecolor='black',
                            fill=False,
                            lw=1))
    ax1.add_patch(Rectangle((10, 21.5), 54, -14,
                            edgecolor='black',
                            facecolor='black',
                            fill=False,
                            lw=1))
    ax1.add_patch(Rectangle((70, 10), 30, -10,
                            edgecolor='red',
                            fill=False,
                            lw=1))
    ax1.text(76, 10, 'MIF', color='black', fontsize=15)
    ax1.text(145, 18, 'Pb Wall', color='black', fontsize=15)
    ax1.text(20, 18, 'Pb Wall', color='black', fontsize=15)
    # russian doll
    ax1.add_patch(Circle((200, 21.5 + 12), 12,
                         edgecolor='green',
                         fill=False,
                         lw=2))
    ax1.text(72, 38, 'Russian Doll Shield', fontsize=15, color='black')
    # ax1.set_title(title)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis="x", direction="in", length=16, width=1)
    ax1.tick_params(axis="y", direction="in", length=16, width=1)
    ax1.tick_params(axis="x", which="minor", direction="in", length=8, width=1)
    ax1.tick_params(axis="y", which="minor", direction="in", length=8, width=1)
    # plt.subplots_adjust(bottom=-.1)
    # plt.tight_layout()
    return fig


if __name__ == "__main__":
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    fig = HFIR_diagram()
    plt.savefig(os.path.join(outdir,"HFIR_diagram.png"), bbox_inches='tight')