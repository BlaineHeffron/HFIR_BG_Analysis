import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import PathPatch, Rectangle, Circle
from matplotlib.path import Path
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
from matplotlib import rcParams
import matplotlib.dates as mdate
from pytz import timezone
from math import ceil, floor
from collections import OrderedDict

#from util import safe_divide, write_x_y_csv

#mpl.use('Agg')
plt.rcParams['font.size'] = '12'
TITLE_SIZE = 16

# initialize globals
cmaps = OrderedDict()
tab_colors = ['tab:blue', 'tab:red', 'tab:brown', 'tab:purple', 'black', 'tab:green', 'tab:grey', 'tab:olive',
              'tab:cyan', 'tab:pink', 'tab:orange']
# see markers list here https://matplotlib.org/3.2.1/api/markers_api.html
category_markers = ['.', '^', 'o', 'v', 's', 'P', 'x', '*', 'd', 'h', '8', 'D', '|', '1', 'p', '<', 'H', '4']
category_styles = ['-', '--', '--', '-', ':']

# ================================================================================== #
# color maps taken from https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
# ================================================================================== #

cmaps['Sequential'] = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis']
cmaps['Sequential2'] = [
    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
cmaps['SequentialBanding'] = [
    'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
    'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
    'hot', 'afmhot', 'gist_heat', 'copper']
cmaps['Diverging'] = [
    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
    'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
cmaps['Cyclic'] = ['twilight', 'twilight_shifted', 'hsv']
cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']
cmaps['Miscellaneous'] = [
    'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
    'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
    'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 22
plt.rcParams['axes.labelsize'] = 22
#plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 22
plt.rcParams['figure.titlesize'] = 24

def plot_z_acc_matrix(cm, nx, ny, title, zlabel="mean average error [mm]", cmap=plt.cm.viridis):
    fontsize = 12
    fig = plt.figure(figsize=(9, 6.5))
    min = 10000000
    for i in range(nx):
        for j in range(ny):
            if min > cm[i, j] > 0:
                min = cm[i,j]
    cm = cm.transpose()
    cm = np.ma.masked_where(cm == 0, cm)
    cmap.set_bad(color='black')
    plt.imshow(cm, interpolation='nearest', cmap=cmap, origin="lower")
    if title != '':
        plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label(zlabel, labelpad=18)
    tick_x = np.arange(nx)
    tick_y = np.arange(ny)
    tick_labelx = np.arange(1, nx + 1)
    tick_labely = np.arange(1, ny + 1)
    plt.xticks(tick_x, tick_labelx)
    plt.yticks(tick_y, tick_labely)
    fmt = '.0f'
    thresh = (cm.max() + min) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] != 0:
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "white", fontsize=fontsize)

    plt.tight_layout()
    plt.ylabel('y segment')
    plt.xlabel('x segment')
    fig.subplots_adjust(left=0.1)
    return fig


def plot_confusion_matrix(cm, classes, normalize=False, title='', cmap=plt.cm.Blues):
    fontsize = 12
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fontsize = 16
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if title != '':
        plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=fontsize)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.subplots_adjust(bottom=0.20)
    return fig


def plot_n_contour(X, Y, Z, xlabel, ylabel, title, suptitle=None, cm=plt.cm.cividis):
    n_categories = len(title)
    nrows = ceil((n_categories - 1) / 3)
    fig_height = 4.0
    if (n_categories < 3):
        fig, axes = plt.subplots(ceil(n_categories / 3), n_categories, figsize=(fig_height * 3.9, fig_height * nrows))
    else:
        fig, axes = plt.subplots(ceil(n_categories / 3), 3, figsize=(fig_height * 3.9, fig_height * nrows))
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=TITLE_SIZE)
    for z, t, i in zip(Z, title, range(n_categories)):
        z = np.transpose(z)
        # axes[int(floor(i / 3)), i % 3].clabel(CS, inline=True)
        if n_categories < 4:
            CS = axes[i].contourf(X, Y, z, cmap=cm)
            axes[i].set_title(t, fontsize=TITLE_SIZE)
            if i == 0:
                axes[i].set_ylabel(ylabel)
            else:
                axes[i].tick_params(axis='y', labelcolor='w')
            axes[i].set_xlabel(xlabel)
            plt.colorbar(CS, ax=axes[i])
        else:
            CS = axes[int(floor(i / 3)), i % 3].contourf(X, Y, z, cmap=cm)
            axes[int(floor(i / 3)), i % 3].set_title(t, fontsize=TITLE_SIZE)
            if i % 3 == 0:
                axes[int(floor(i / 3)), i % 3].set_ylabel(ylabel)
            else:
                axes[int(floor(i / 3)), i % 3].tick_params(axis='y', labelcolor='w')
            if floor(i / 3) == floor((n_categories - 1) / 3):
                axes[int(floor(i / 3)), i % 3].set_xlabel(xlabel)
            else:
                axes[int(floor(i / 3)), i % 3].tick_params(axis='x', labelcolor='w')
            plt.colorbar(CS, ax=axes[floor(i / 3), i % 3])
    i = 0
    for ax in fig.get_axes():
        if i == n_categories:
            break
        ax.label_outer()
        i += 1
    if n_categories < 4:
        fig.subplots_adjust(bottom=0.18)
    return fig


def plot_contour(X, Y, Z, xlabel, ylabel, title, filled=True, cm=plt.cm.cividis):
    Z = np.transpose(Z)
    fig, ax = plt.subplots()
    if filled:
        CS = ax.contourf(X, Y, Z, cmap=cm)
        plt.colorbar(CS, ax=ax)
    else:
        CS = ax.contour(X, Y, Z, cmap=cm)
        ax.clabel(CS, inline=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=TITLE_SIZE)
    return fig


def plot_bar(X, Y, xlabel, ylabel):
    fig, ax = plt.subplots()
    plt.bar(X, Y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def plot_n_hist1d(xedges, vals, title, xlabel, ylabel, suptitle=None, norm_to_bin_width=True, logy=True):
    n_categories = len(title)
    nrows = ceil((n_categories - 1) / 3)
    fig_height = 4.0
    if (n_categories < 3):
        fig, axes = plt.subplots(ceil(n_categories / 3), n_categories, figsize=(fig_height * 3.9, fig_height * nrows))
    else:
        fig, axes = plt.subplots(ceil(n_categories / 3), 3, figsize=(fig_height * 3.9, fig_height * nrows))
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=TITLE_SIZE)
    xwidth = xedges[1] - xedges[0]
    for m in range(n_categories):
        if norm_to_bin_width:
            vals[m] = vals[m].astype(np.float32)
            vals[m] *= (1. / xwidth)
        tot = vals[m].shape[0]
        w = np.zeros((tot,))
        xs = np.zeros((tot,))
        n = 0
        for i in range(len(xedges) - 1):
            x = xwidth * i + xwidth / 2.
            w[n] = vals[m][i]
            xs[n] = x
            n += 1
        if (n_categories < 4):
            axes[m].hist(xs, bins=xedges, weights=w)
            axes[m].set_xlabel(xlabel)
            if m == 0:
                axes[m].set_ylabel(ylabel)
            axes[m].set_title(title[m], fontsize=TITLE_SIZE)
        else:
            axes[floor(m / 3), m % 3].hist(xs, bins=xedges, weights=w)
            if floor(m / 3) == floor((n_categories - 1) / 3):
                axes[floor(m / 3), m % 3].set_xlabel(xlabel)
            else:
                axes[floor(m / 3), m % 3].tick_params(axis='x', labelcolor='w')
            if m % 3 == 0:
                axes[floor(m / 3), m % 3].set_ylabel(ylabel)
            axes[floor(m / 3), m % 3].set_title(title[m], fontsize=TITLE_SIZE)
    i = 0
    if logy:
        for ax in fig.get_axes():
            if i == n_categories:
                break
            # ax.label_outer()
            ax.set_yscale('log')
            i += 1
    if n_categories < 4:
        fig.subplots_adjust(bottom=0.18)
    # cb.set_label(zlabel, rotation=270)
    return fig


def plot_n_hist2d(xedges, yedges, vals, title, xlabel, ylabel, suptitle=None, norm_to_bin_width=True, logz=True,
                  cm=plt.cm.cividis):
    n_categories = len(title)
    nrows = ceil((n_categories - 1) / 3)
    fig_height = 4.0
    if n_categories < 3:
        fig, axes = plt.subplots(ceil(n_categories / 3), n_categories, figsize=(fig_height * 3.9, fig_height * nrows))
    else:
        fig, axes = plt.subplots(ceil(n_categories / 3), 3, figsize=(fig_height * 3.9, fig_height * nrows))
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=TITLE_SIZE)
    xwidth = xedges[1] - xedges[0]
    ywidth = yedges[1] - yedges[0]
    for m in range(n_categories):
        if norm_to_bin_width:
            vals[m] = vals[m].astype(np.float32)
            vals[m] *= 1. / (xwidth * ywidth)
        tot = vals[m].shape[0] * vals[m].shape[1]
        w = np.zeros((tot,))
        xs = np.zeros((tot,))
        ys = np.zeros((tot,))
        n = 0
        for i in range(len(xedges) - 1):
            x = xedges[0] + xwidth * i + xwidth / 2.
            for j in range(len(yedges) - 1):
                y = yedges[0] + ywidth * j + ywidth / 2.
                if vals[m][i, j] <= 0 and logz:
                    w[n] = 1. / (xwidth * ywidth) if norm_to_bin_width else 1.
                else:
                    w[n] = vals[m][i, j]
                xs[n] = x
                ys[n] = y
                n += 1
        if n_categories < 4:
            if logz:
                h = axes[m].hist2d(xs, ys, bins=[xedges, yedges], weights=w, cmap=cm, norm=LogNorm())
            else:
                h = axes[m].hist2d(xs, ys, bins=[xedges, yedges], weights=w, cmap=cm)
            if floor(m / 3) == floor((n_categories - 1) / 3):
                axes[m].set_xlabel(xlabel)
            else:
                axes[m].tick_params(axis='x', labelcolor='w')
            if m == 0:
                axes[m].set_ylabel(ylabel)
            else:
                axes[m].tick_params(axis='y', labelcolor='w')
            axes[m].set_title(title[m], fontsize=TITLE_SIZE)
            plt.colorbar(h[3], ax=axes[m])
        else:
            if logz:
                h = axes[floor(m / 3), m % 3].hist2d(xs, ys, bins=[xedges, yedges], weights=w, cmap=cm, norm=LogNorm())
            else:
                h = axes[floor(m / 3), m % 3].hist2d(xs, ys, bins=[xedges, yedges], weights=w, cmap=cm)
            if floor(m / 3) == floor((n_categories - 1) / 3):
                axes[floor(m / 3), m % 3].set_xlabel(xlabel)
            else:
                axes[floor(m / 3), m % 3].tick_params(axis='x', labelcolor='w')
            if m % 3 == 0:
                axes[floor(m / 3), m % 3].set_ylabel(ylabel)
            else:
                axes[floor(m / 3), m % 3].tick_params(axis='y', labelcolor='w')
            axes[floor(m / 3), m % 3].set_title(title[m], fontsize=TITLE_SIZE)
            plt.colorbar(h[3], ax=axes[floor(m / 3), m % 3])
    i = 0
    for ax in fig.get_axes():
        if i == n_categories:
            break
        ax.label_outer()
        i += 1
    # cb.set_label(zlabel, rotation=270)
    if n_categories < 4:
        fig.subplots_adjust(bottom=0.18)
    return fig


def plot_hist2d(xedges, yedges, vals, title, xlabel, ylabel, zlabel, norm_to_bin_width=True, logz=True,
                cm=plt.cm.cividis):
    fig, ax = plt.subplots()
    xwidth = xedges[1] - xedges[0]
    ywidth = yedges[1] - yedges[0]
    if norm_to_bin_width:
        vals = vals.astype(np.float32)
        vals *= 1. / (xwidth * ywidth)
    tot = vals.shape[0] * vals.shape[1]
    w = np.zeros((tot,))
    xs = np.zeros((tot,))
    ys = np.zeros((tot,))
    n = 0
    for i in range(len(xedges) - 1):
        x = xedges[0] + xwidth * i + xwidth / 2.
        for j in range(len(yedges) - 1):
            y = yedges[0] + ywidth * j + ywidth / 2.
            w[n] = vals[i, j]
            xs[n] = x
            ys[n] = y
            n += 1
    if logz:
        h = plt.hist2d(xs, ys, bins=[xedges, yedges], weights=w, cmap=cm, norm=LogNorm())
    else:
        h = plt.hist2d(xs, ys, bins=[xedges, yedges], weights=w, cmap=cm)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=TITLE_SIZE)
    cb = plt.colorbar(h[3])
    if(zlabel):
        cb.set_label(zlabel, rotation=270, labelpad=20)
    return fig


def plot_hist1d(xedges, vals, title, xlabel, ylabel, norm_to_bin_width=True):
    fig, ax = plt.subplots()
    xwidth = xedges[1] - xedges[0]
    if norm_to_bin_width:
        vals = vals.astype(np.float32)
        vals /= xwidth
    tot = vals.shape[0]
    xs = np.zeros((tot,))
    n = 0
    for i in range(len(xedges) - 1):
        x = xedges[0] + xwidth * i + xwidth / 2.
        xs[n] = x
        n += 1
    h = plt.hist(xs, bins=xedges, weights=vals)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis="x", direction="in", length=16, width=1)
    ax.tick_params(axis="y", direction="in", length=16, width=1)
    ax.tick_params(axis="x", which="minor", direction="in", length=8, width=1)
    ax.tick_params(axis="y", which="minor", direction="in", length=8, width=1)
    ax.set_title(title, fontsize=TITLE_SIZE)
    return fig


def plot_roc(data, class_names):
    # Plot all ROC curves
    lw = 4
    fig, ax = plt.subplots()
    for i, classd in enumerate(data):
        plt.plot(classd[0], classd[1],
                 label=class_names[i],
                 color=tab_colors[i % 10],
                 # marker=category_markers[i % len(category_markers)],
                 ls=category_styles[i % len(category_styles)],
                 linewidth=lw)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    return fig


def plot_pr(data, class_names):
    # Plot all ROC curves
    lw = 4
    fig, ax = plt.subplots()
    for i, classd in enumerate(data):
        plt.plot(classd[1], classd[0],
                 label=class_names[i],
                 color=tab_colors[i % 10],
                 # marker=category_markers[i % len(category_markers)],
                 ls=category_styles[i % len(category_styles)],
                 linewidth=lw)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    plt.legend(loc="lower right")
    return fig



def GetMPLStyles():
    style_list = []
    for i in range(20):
        style_list.append(tab_colors[i%10] + category_styles[i%len(category_styles)])
    return style_list


def MultiLinePlot(xaxis, yvals, line_labels, xlabel, ylabel,
                  colors=None, styles=None,
                  xmax=-1, ymax=-1, ymin=None, xmin=None, ylog=True, xdates=False,
                  vertlines=None, vlinelabel=None, xlog=False, title=None, figsize=(12, 9)):
    rcParams.update({'font.size': 22})
    if colors is None:
        colors = []
    if styles is None:
        styles = []
    if(xdates):
        xaxis = mdate.epoch2num(xaxis)
        if(vertlines):
            vertlines = mdate.epoch2num(vertlines)
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    if(ymin is None):
        if(ylog):
            ymin = min([min(y)  for y in yvals])*0.5
            if(ymin <= 0):
                print("error: ymin is ", ymin, " on a log-y axis. Defaulting to 1e-5")
                ymin = 1e-5
        else:
            ymin = min([min(y)  for y in yvals])
            if ymin < 0: ymin *= 1.05
            else: ymin *= .95
    if(xmin is None):
        xmin = min(xaxis)
    if(xlog):
        ax1.set_xscale('log')
    else:
        ax1.set_xscale('linear')
    if(ylog):
        ax1.set_yscale('log')
    else:
        ax1.set_yscale('linear')
    if(ymax == -1):
        if(ylog):
            ax1.set_ylim(ymin,max([max(y) for y in yvals])*1.5)
        else:
            ax1.set_ylim(ymin,max([max(y) for y in yvals])*1.05)
    else:
        ax1.set_ylim(ymin,ymax)
    if(xmax == -1):
        ax1.set_xlim(xmin,max(xaxis))
    else:
        ax1.set_xlim(xmin,xmax)
    #for i, y in enumerate(yvals):
    if not colors:
        colors = tab_colors
    if not styles:
        styles = category_styles
    for i in range(len(yvals)):
        ax1.plot(xaxis,yvals[i],color=colors[i%10],linestyle=styles[i%len(styles)])
    if(vertlines is not None):
        for v in vertlines:
            ax1.axvline(v,color='k',linestyle='-')#,label=vlinelabel)
    if(xdates):
        #date_fmt = '%y-%m-%d %H:%M:%S'
        date_fmt = '%y-%m-%d'
        date_formatter = mdate.DateFormatter(date_fmt,tz=timezone('US/Eastern'))
        ax1.xaxis.set_major_formatter(date_formatter)
        fig.autofmt_xdate()
    else:
        pass
        #start,end = ax1.get_xlim()
        #diff = (end - start) / 8.
        #ax1.xaxis.set_ticks(np.arange(start,end,diff))
        #ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        #plt.setp(ax1.get_xticklabels(), rotation=30,\
        #horizontalalignment='right')
    box = ax1.get_position()
    if(len(yvals) == 1):
        if(not title):
            ax1.set_title(line_labels[0])
        else:
            ax1.set_title(title)
    else:
        if(title):
            ax1.set_title(title)
        ax1.set_position([box.x0,box.y0,box.width,box.height])
        ax1.legend(line_labels,loc='center left', \
                   bbox_to_anchor=(0.20,0.22),ncol=1)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    if not ylog:
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis="x", direction="in", length=16, width=1)
    ax1.tick_params(axis="y", direction="in", length=16, width=1)
    ax1.tick_params(axis="x", which="minor", direction="in", length=8, width=1)
    ax1.tick_params(axis="y", which="minor", direction="in", length=8, width=1)
    #plt.gcf().subplots_adjust(left=0.16)
    #plt.gcf().subplots_adjust(bottom=0.22)
    #plt.gcf().subplots_adjust(right=0.05)
    #plt.savefig(outname)
    #plt.close()
    return fig

def ScatterDifferencePlot(xaxis, ref, ref_err, yvals, errors, yval_labels, xlabel, ylabel,
                    colors=None, styles=None,
                    xmax=-1, ymax=-1, ymin=None, xmin=None, xdates=False,
                    vertlines=None, xlog=False, title=None,  figsize=(12, 9),
                    legend_loc='upper right'):
    if colors is None:
        colors = []
    if styles is None:
        styles = []
    if(xdates):
        xaxis = mdate.epoch2num(xaxis)
        if(vertlines):
            vertlines = mdate.epoch2num(vertlines)
    rcParams.update({'font.size': 18})
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    yvals -= ref
    if(ymin is None):
        ymin = np.amin(yvals)*1.1
        errmin = np.amin(-1*ref_err)*1.1
        if errmin < ymin:
            ymin = errmin
    if(xmin is None):
        xmin = np.amin(xaxis)*.98
    if(xlog):
        ax1.set_xscale('log')
    else:
        ax1.set_xscale('linear')
    if(ymax == -1):
        if np.amax(yvals) > np.amax(ref_err):
            ax1.set_ylim(ymin,np.amax(yvals)*1.05)
        else:
            ax1.set_ylim(ymin, np.amax(ref_err) * 1.05)
    else:
        ax1.set_ylim(ymin,ymax)
    if(xmax == -1):
        ax1.set_xlim(xmin,np.amax(xaxis) + np.amin(xaxis)*.02)
    else:
        ax1.set_xlim(xmin,xmax)
    if not colors:
        colors = tab_colors
    if not styles:
        styles = category_styles
    for i in range(yvals.shape[0]):
        ax1.errorbar(xaxis,yvals[i],yerr=errors[i], fmt='o')
    ax1.fill_between(xaxis, ref_err, -ref_err, alpha=0.2)
    if(vertlines is not None):
        for v in vertlines:
            ax1.axvline(v,color='k',linestyle='-')#,label=vlinelabel)
    if(xdates):
        #date_fmt = '%y-%m-%d %H:%M:%S'
        date_fmt = '%y-%m-%d'
        date_formatter = mdate.DateFormatter(date_fmt,tz=timezone('US/Eastern'))
        ax1.xaxis.set_major_formatter(date_formatter)
        fig.autofmt_xdate()
    else:
        pass
        #start,end = ax1.get_xlim()
        #diff = (end - start) / 8.
        #ax1.xaxis.set_ticks(np.arange(start,end,diff))
        #ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        #plt.setp(ax1.get_xticklabels(), rotation=30,\
        #horizontalalignment='right')
    if title:
        ax1.set_title(title)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width, box.height])
    ax1.legend([r"$1\sigma$ error band"] + yval_labels, loc=legend_loc)
    rcParams.update({'font.size': 14})
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis="x", direction="in", length=16, width=1)
    ax1.tick_params(axis="y", direction="in", length=16, width=1)
    ax1.tick_params(axis="x", which="minor", direction="in", length=8, width=1)
    ax1.tick_params(axis="y", which="minor", direction="in", length=8, width=1)
    #plt.gcf().subplots_adjust(left=0.16)
    #plt.gcf().subplots_adjust(bottom=0.22)
    #plt.gcf().subplots_adjust(right=0.05)
    #plt.savefig(outname)
    #plt.close()
    return fig

def ScatterLinePlot(xaxis, yvals, errors, linex, liney, lineerr, line_labels, xlabel, ylabel,
                     colors=None, styles=None,
                     xmax=-1, ymax=-1, ymin=None, xmin=None, ylog=True, xdates=False,
                     vertlines=None, vlinelabel=None, xlog=False, title=None,  figsize=(12, 9),
                    legend_loc='center left'):
    if colors is None:
        colors = []
    if styles is None:
        styles = []
    if(xdates):
        xaxis = mdate.epoch2num(xaxis)
        if(vertlines):
            vertlines = mdate.epoch2num(vertlines)
    rcParams.update({'font.size': 18})
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    if(ymin is None):
        if(ylog):
            ymin = min(yvals)*0.5
            if(ymin <= 0):
                print("error: ymin is ", ymin, " on a log-y axis. Defaulting to 1e-5")
                ymin = 1e-5
        else:
            ymin = min(yvals)
            if ymin < 0: ymin *= 1.05
            else: ymin *= .95
    if(xmin is None):
        xmin = min(xaxis)
    if(xlog):
        ax1.set_xscale('log')
    else:
        ax1.set_xscale('linear')
    if(ylog):
        ax1.set_yscale('log')
    else:
        ax1.set_yscale('linear')
    if(ymax == -1):
        if(ylog):
            ax1.set_ylim(ymin,max(yvals)*1.5)
        else:
            ax1.set_ylim(ymin,max(yvals)*1.05)
    else:
        ax1.set_ylim(ymin,ymax)
    if(xmax == -1):
        ax1.set_xlim(xmin,max(xaxis))
    else:
        ax1.set_xlim(xmin,xmax)
    if not colors:
        colors = tab_colors
    if not styles:
        styles = category_styles
    ax1.plot(linex, liney)
    ax1.errorbar(xaxis,yvals,yerr=errors, fmt='o')
    ax1.fill_between(linex, liney - lineerr, liney + lineerr, alpha=0.2)
    if(vertlines is not None):
        for v in vertlines:
            ax1.axvline(v,color='k',linestyle='-')#,label=vlinelabel)
    if(xdates):
        #date_fmt = '%y-%m-%d %H:%M:%S'
        date_fmt = '%y-%m-%d'
        date_formatter = mdate.DateFormatter(date_fmt,tz=timezone('US/Eastern'))
        ax1.xaxis.set_major_formatter(date_formatter)
        fig.autofmt_xdate()
    else:
        pass
        #start,end = ax1.get_xlim()
        #diff = (end - start) / 8.
        #ax1.xaxis.set_ticks(np.arange(start,end,diff))
        #ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        #plt.setp(ax1.get_xticklabels(), rotation=30,\
        #horizontalalignment='right')
    if(not title):
        ax1.set_title(line_labels[0])
    else:
        ax1.set_title(title)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width, box.height])
    #ax1.legend(line_labels, loc=legend_loc, \
    #           bbox_to_anchor=(0.75, 0.85), ncol=1)
    ax1.legend(line_labels, loc=legend_loc)
    rcParams.update({'font.size': 14})
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    if not ylog:
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis="x", direction="in", length=16, width=1)
    ax1.tick_params(axis="y", direction="in", length=16, width=1)
    ax1.tick_params(axis="x", which="minor", direction="in", length=8, width=1)
    ax1.tick_params(axis="y", which="minor", direction="in", length=8, width=1)
    #plt.gcf().subplots_adjust(left=0.16)
    #plt.gcf().subplots_adjust(bottom=0.22)
    #plt.gcf().subplots_adjust(right=0.05)
    #plt.savefig(outname)
    #plt.close()
    return fig

def MultiScatterPlot(xaxis, yvals, errors, line_labels, xlabel, ylabel,
                  colors=None, styles=None,
                  xmax=-1, ymax=-1, ymin=None, xmin=None, ylog=True, xdates=False,
                  vertlines=None, vlinelabel=None, xlog=False, title=None,  figsize=(12, 9)):
    if colors is None:
        colors = []
    if styles is None:
        styles = []
    if(xdates):
        xaxis = mdate.epoch2num(xaxis)
        if(vertlines):
            vertlines = mdate.epoch2num(vertlines)
    rcParams.update({'font.size': 18})
    fig = plt.figure(figsize=(12, 6.5))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    if(ymin is None):
        if(ylog):
            ymin = min([min(y)  for y in yvals])*0.5
            if(ymin <= 0):
                print("error: ymin is ", ymin, " on a log-y axis. Defaulting to 1e-5")
                ymin = 1e-5
        else:
            ymin = min([min(y)  for y in yvals])
            if ymin < 0: ymin *= 1.05
            else: ymin *= .95
    if(xmin is None):
        xmin = min(xaxis) - (max(xaxis) - min(xaxis))*.01
    if(xlog):
        ax1.set_xscale('log')
    else:
        ax1.set_xscale('linear')
    if(ylog):
        ax1.set_yscale('log')
    else:
        ax1.set_yscale('linear')
    if(ymax == -1):
        if(ylog):
            ax1.set_ylim(ymin,max([max(y) for y in yvals])*1.5)
        else:
            ax1.set_ylim(ymin,max([max(y) for y in yvals])*1.05)
    else:
        ax1.set_ylim(ymin,ymax)
    if(xmax == -1):
        ax1.set_xlim(xmin,max(xaxis) + (max(xaxis) - min(xaxis))*.01)
    else:
        ax1.set_xlim(xmin,xmax)
    #for i, y in enumerate(yvals):
    if not colors:
        colors = tab_colors
    if not styles:
        styles = category_styles
    for i in range(len(yvals)):
        ax1.errorbar(xaxis,yvals[i],yerr=errors[i],color=colors[i%10],fmt=category_markers[i%len(category_markers)], capsize=3)
    if(vertlines is not None):
        for v in vertlines:
            ax1.axvline(v,color='k',linestyle='-')#,label=vlinelabel)
    if(xdates):
        #date_fmt = '%y-%m-%d %H:%M:%S'
        date_fmt = '%y-%m-%d'
        date_formatter = mdate.DateFormatter(date_fmt,tz=timezone('US/Eastern'))
        ax1.xaxis.set_major_formatter(date_formatter)
        fig.autofmt_xdate()
    else:
        pass
        #start,end = ax1.get_xlim()
        #diff = (end - start) / 8.
        #ax1.xaxis.set_ticks(np.arange(start,end,diff))
        #ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        #plt.setp(ax1.get_xticklabels(), rotation=30,\
        #horizontalalignment='right')
    box = ax1.get_position()
    if(len(yvals) == 1):
        if(not title):
            ax1.set_title(line_labels[0])
        else:
            ax1.set_title(title)
    else:
        if(title):
            ax1.set_title(title)
        ax1.set_position([box.x0,box.y0,box.width,box.height])
        ax1.legend(line_labels,loc='center left', \
                   bbox_to_anchor=(0.5,0.85),ncol=1)
        rcParams.update({'font.size':14})
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    if not ylog:
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis="x", direction="in", length=16, width=1)
    ax1.tick_params(axis="y", direction="in", length=16, width=1)
    ax1.tick_params(axis="x", which="minor", direction="in", length=8, width=1)
    ax1.tick_params(axis="y", which="minor", direction="in", length=8, width=1)
    #plt.gcf().subplots_adjust(left=0.16)
    #plt.gcf().subplots_adjust(bottom=0.22)
    #plt.gcf().subplots_adjust(right=0.05)
    #plt.savefig(outname)
    #plt.close()
    return fig

def scatter_plot(x, y, c, xlabel, ylabel, zlabel, title, ymin=None, ymax=None, xmin=None, xmax=None, invert_y=False, invert_x=False, xdates=False):
    rcParams.update({'font.size': 18})
    if xmin is not None and xmax is not None and ymin is not None and ymax is not None:
        ratio = abs((ymax - ymin) / (xmax - xmin))*4./5
        fig = plt.figure(figsize=(12, 12*ratio))
    else:
        fig = plt.figure(figsize=(12, 6.5))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    if(xdates):
        x = mdate.epoch2num(x)
    if ymin is None:
        ymin = min(y)
        if ymin < 0:
            ymin *= 1.05
        else:
            ymin *= .95
    if xmin is None:
        xmin = min(x) * .95
    if ymax is None:
        if invert_y:
            ax1.set_ylim(max(y)*1.05, ymin)
        else:
            ax1.set_ylim(ymin, max(y) * 1.05)
    else:
        if invert_y:
            ax1.set_ylim(ymax,ymin)
        else:
            ax1.set_ylim(ymin, ymax)
    if xmax is None:
        if invert_x:
            ax1.set_xlim(max(x)*1.05, xmin)
        else:
            ax1.set_xlim(xmin, max(x) * 1.05)
    else:
        if invert_x:
            ax1.set_xlim(xmax,xmin)
        else:
            ax1.set_xlim(xmin, xmax)
    if(xdates):
        #date_fmt = '%y-%m-%d %H:%M:%S'
        date_fmt = '%y-%m-%d'
        date_formatter = mdate.DateFormatter(date_fmt,tz=timezone('US/Eastern'))
        ax1.xaxis.set_major_formatter(date_formatter)
        fig.autofmt_xdate()
    h = ax1.scatter(x,y,c=c, cmap='viridis')
    ax1.set_title(title)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis="x", direction="in", length=16, width=1)
    ax1.tick_params(axis="y", direction="in", length=16, width=1)
    ax1.tick_params(axis="x", which="minor", direction="in", length=8, width=1)
    ax1.tick_params(axis="y", which="minor", direction="in", length=8, width=1)
    cb = plt.colorbar(h)
    cb.set_label(zlabel, rotation=270, labelpad=20)
    return fig


def HFIR_scatter_plot(x, y, c, xlabel, ylabel, zlabel, title, invert_y=False, invert_x=False, xdates=False):
    xmin = 40
    ymin = 0
    xmax = 420
    ymax = 180
    rcParams.update({'font.size': 18})
    ratio = abs((ymax - ymin) / (xmax - xmin))*4./5
    fig = plt.figure(figsize=(14, 14*ratio))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    if(xdates):
        x = mdate.epoch2num(x)
    if ymin is None:
        ymin = min(y)
        if ymin < 0:
            ymin *= 1.05
        else:
            ymin *= .95
    if xmin is None:
        xmin = min(x) * .95
    if ymax is None:
        if invert_y:
            ax1.set_ylim(max(y)*1.05, ymin)
        else:
            ax1.set_ylim(ymin, max(y) * 1.05)
    else:
        if invert_y:
            ax1.set_ylim(ymax,ymin)
        else:
            ax1.set_ylim(ymin, ymax)
    if xmax is None:
        if invert_x:
            ax1.set_xlim(max(x)*1.05, xmin)
        else:
            ax1.set_xlim(xmin, max(x) * 1.05)
    else:
        if invert_x:
            ax1.set_xlim(xmax,xmin)
        else:
            ax1.set_xlim(xmin, xmax)
    # add box for prospect
    ax1.add_patch(Rectangle((165, 128), 46.25, -83.4,
                               edgecolor='cornflowerblue',
                               fill=False,
                               lw=2))
    # lead shield wall
    ax1.add_patch(Rectangle((125, 21.5), 286.5-155, -14,
                            edgecolor='black',
                            facecolor='black',
                            fill=True,
                            lw=1))
    # russian doll
    ax1.add_patch(Circle((200, 21.5+12), 12,
                            edgecolor='green',
                            fill=False,
                            lw=2))
    h = ax1.scatter(x,y,c=c, cmap='viridis')
    ax1.set_title(title)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis="x", direction="in", length=16, width=1)
    ax1.tick_params(axis="y", direction="in", length=16, width=1)
    ax1.tick_params(axis="x", which="minor", direction="in", length=8, width=1)
    ax1.tick_params(axis="y", which="minor", direction="in", length=8, width=1)
    cb = plt.colorbar(h)
    cb.set_label(zlabel, rotation=270, labelpad=20)
    #plt.subplots_adjust(bottom=-.1)
    #plt.tight_layout()
    return fig

def draw_error_band(ax, x, y, err, **kwargs):
    # Calculate normals via centered finite differences (except the first point
    # which uses a forward difference and the last point which uses a backward
    # difference).
    dx = np.concatenate([[x[1] - x[0]], x[2:] - x[:-2], [x[-1] - x[-2]]])
    dy = np.concatenate([[y[1] - y[0]], y[2:] - y[:-2], [y[-1] - y[-2]]])
    l = np.hypot(dx, dy)
    nx = dy / l
    ny = -dx / l

    # end points of errors
    xp = x + nx * err
    yp = y + ny * err
    xn = x - nx * err
    yn = y - ny * err

    vertices = np.block([[xp, xn[::-1]],
                         [yp, yn[::-1]]]).T
    codes = np.full(len(vertices), Path.LINETO)
    codes[0] = codes[len(xp)] = Path.MOVETO
    path = Path(vertices, codes)
    ax.add_patch(PathPatch(path, **kwargs))