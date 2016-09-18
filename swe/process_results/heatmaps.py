import numpy as np
import glob
import csv
import ast
from sklearn import preprocessing
from swe_pareto_tools import get_gp_filter_location_relevance, get_gpesa_location_relevance
from gather_hill_climber_results import plot_wrapper

import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import matplotlib.patheffects as PathEffects
import matplotlib.gridspec as gridspec


def pipeline(year, predictor_level, response_level, scale=False):
    predictors_train = glob.glob("/Volumes/sam/train/{0}/predictors*{1}.npy".format(predictor_level, year))
    response_train = glob.glob("/Volumes/sam/train/{0}/response*{1}.npy".format(response_level, year))
    predictors_test = glob.glob("/Volumes/sam/test/{0}/predictors*{1}.npy".format(predictor_level, year))
    response_test = glob.glob("/Volumes/sam/test/{0}/response*{1}.npy".format(response_level, year))
    p_train, r_train = np.load(predictors_train[0]), np.load(response_train[0])
    p_test, r_test = np.load(predictors_test[0]), np.load(response_test[0])

    r_train, r_test = np.log(r_train), np.log(r_test)

    if scale:  # never scale in preprocessing for circle method
        scale = preprocessing.StandardScaler()
        p_train = scale.fit_transform(p_train)
        p_test = scale.transform(p_test)
    return p_train, r_train, p_test, r_test


def get_nonzero_frequency_mask():
    nonzero_frequency_mask = np.zeros((3, 113*113))
    for year in range(2003, 2012):
        _predictors, _, _, _ = pipeline(year, "pixel_level", "daily_total", scale=False)
        predictors = np.array([_predictors[:, np.arange(_predictors.shape[1]) % 3 == n].T for n in range(3)])
        current_nonzeros = np.sum(predictors, axis=2) != 0.0
        nonzero_frequency_mask[current_nonzeros] += 1
    # nonzero_frequency_mask = np.sum(nonzero_frequency_mask, axis=0).reshape((113, 113))
    return np.array([np.array(feature).reshape(113, 113)[:, ::-1].T for feature in nonzero_frequency_mask])


def get_circle_grid(new_resolution, old_resolution):
    tile_lengths = np.repeat(old_resolution / float(new_resolution), new_resolution)
    radiuses = tile_lengths * (2 ** 0.5) / 2
    centroids = tile_lengths / 2 + np.pad(np.cumsum(tile_lengths), (1, 0), mode='constant')[:-1]
    return centroids, radiuses


def plot_single_circle_grid(centroids, radiuses, ax, intensities, grid=True, alpha=0.75):
    # intensities = np.ma.masked_equal(abs(np.array(intensities)), .0)
    patches = []
    count = 0
    if grid:
        for n, x in enumerate(centroids):
            for y, r in zip(centroids, radiuses):
                # ax.text(x, y, count)
                count += 1
                circle = Circle((x, y), r)
                patches.append(circle)
    else:
        for xy, r in zip(centroids, radiuses):
            count += 1
            circle = Circle(xy, r)
            patches.append(circle)

    sorted_index = [idx for (intensity, idx) in sorted(zip(intensities, range(len(intensities))))]
    patches = [patches[idx] for idx in sorted_index]
    intensities = [intensities[idx] for idx in sorted_index]

    norm = mpl.colors.Normalize(vmin=0.0, vmax=max(intensities))
    cm.jet.set_bad(color='white', alpha=0.0)
    colors = [('white')] + [(cm.jet(i)) for i in xrange(1, 256)]
    new_map = mpl.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256)

    p = PatchCollection(patches, cmap=new_map, alpha=alpha, norm=norm, linewidth=0)
    p.set_array(np.array(intensities))
    ax.add_collection(p)

    ax.annotate(int(np.sqrt(count)), xy=(2, 90), fontsize=30,
                path_effects=[PathEffects.withStroke(linewidth=3, foreground="w")])


def get_importance(resolution, model_type="Ridge"):
    file_patterns = glob.glob("../data/SweData/linear_regression/*/*{0}*resolution_{1}.csv".format(model_type, resolution))
    coefficients = np.zeros((3 * resolution ** 2,))
    for file_name in file_patterns:
        with open(file_name, "rb") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                coefficients += abs(np.array(ast.literal_eval(row[6])))

    coefficients = coefficients.reshape(3, resolution ** 2)
    coefficients = np.max(coefficients, axis=0)
    coefficients = coefficients.flatten()
    index = range(resolution ** 2) * 3  # three features with the same circle
    max_importance = [0] * resolution ** 2
    for circle, coef in zip(index, coefficients):
        # if abs(coef) > max_importance[circle]:
        #     max_importance[circle] = abs(coef)
        max_importance[circle] += abs(coef) / 3.0
    return max_importance


def plot_multiple_grids(new_resolutions, rows, old_resolution=113):
    nonzero_frequency_mask = get_nonzero_frequency_mask()
    hfont = {'fontname': 'Times'}

    global_rows = 3
    global_columns = 2
    height = 30.0
    spacing = 0.10
    ratio = (10 - (global_columns-1)*spacing/10. - 0.0001) / (12.0 + (global_rows-1)*spacing)  #0.0002 too wide; 0.0001 is just a tad bit too tall
    width = height * ratio
    fig = plt.figure(figsize=(width, height))

    gs0 = gridspec.GridSpec(global_rows, global_columns)
    gs0.update(wspace=spacing/10., hspace=spacing)

    # Filtered Ridge
    gs00 = gridspec.GridSpecFromSubplotSpec(rows, len(new_resolutions)/rows, subplot_spec=gs0[0], wspace=0.0, hspace=0.0)
    ax1 = plt.Subplot(fig, gs00[:, :])
    fig.add_subplot(ax1)
    plt.title('Filtered Ridge (FR)', fontsize=30, **hfont)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.annotate('a', xy=(0, 1), xycoords='axes fraction', fontsize=40, xytext=(0, 40),
                 textcoords='offset points', ha='right', va='top')
    ax = []
    for i in range(len(new_resolutions)):
        ax1 = plt.Subplot(fig, gs00[i/(rows+1), i % (rows+1)])
        fig.add_subplot(ax1)
        ax.append(ax1)

    for n, res in enumerate(new_resolutions):
        centroids, radiuses = get_circle_grid(res, old_resolution)
        intensities = get_importance(res, "Ridge")
        plot_single_circle_grid(centroids, radiuses, ax[n], intensities)
        ax[n].set_xlim([0, 112])
        ax[n].set_ylim([0, 112])
        ax[n].set_xticks([])
        ax[n].set_yticks([])
        if n > -1:
            yticks = ax[n].yaxis.get_major_ticks()
            xticks = ax[n].xaxis.get_major_ticks()
            for x, y in zip(xticks, yticks):
                y.label1.set_visible(False)
                x.label1.set_visible(False)

    # Filtered Lasso
    gs02 = gridspec.GridSpecFromSubplotSpec(rows, len(new_resolutions)/rows, subplot_spec=gs0[2], wspace=0.0, hspace=0.0)
    ax1 = plt.Subplot(fig, gs02[:, :])
    fig.add_subplot(ax1)
    plt.title('Filtered Lasso (FL)', fontsize=30, **hfont)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.annotate('c', xy=(0, 1), xycoords='axes fraction', fontsize=40, xytext=(0, 40),
                 textcoords='offset points', ha='right', va='top')

    ax = []
    for i in range(len(new_resolutions)):
        ax1 = plt.Subplot(fig, gs02[i/(rows+1), i % (rows+1)])
        fig.add_subplot(ax1)
        ax.append(ax1)

    for n, res in enumerate(new_resolutions):
        centroids, radiuses = get_circle_grid(res, old_resolution)
        intensities = get_importance(res, "Lasso")
        plot_single_circle_grid(centroids, radiuses, ax[n], intensities)
        ax[n].set_xlim([0, 112])
        ax[n].set_ylim([0, 112])
        ax[n].set_xticks([])
        ax[n].set_yticks([])
        if n > -1:
            yticks = ax[n].yaxis.get_major_ticks()
            xticks = ax[n].xaxis.get_major_ticks()
            for x, y in zip(xticks, yticks):
                y.label1.set_visible(False)
                x.label1.set_visible(False)

    # Filtered GP
    gs04 = gridspec.GridSpecFromSubplotSpec(rows, len(new_resolutions)/rows, subplot_spec=gs0[4], wspace=0.0, hspace=0.0)
    ax1 = plt.Subplot(fig, gs04[:, :])
    fig.add_subplot(ax1)
    plt.title('Filtered GP (FGP)', fontsize=30, **hfont)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.annotate('e', xy=(0, 1), xycoords='axes fraction', fontsize=40, xytext=(0, 40),
                 textcoords='offset points', ha='right', va='top')
    axes = []
    for i in range(len(new_resolutions)):
        ax1 = plt.Subplot(fig, gs04[i/(rows+1), i % (rows+1)])
        fig.add_subplot(ax1)
        axes.append(ax1)

    for n, res in enumerate(new_resolutions):
        centroids, radiuses = get_circle_grid(res, old_resolution)
        intensities = get_gp_filter_location_relevance(res)
        plot_single_circle_grid(centroids, radiuses, axes[n], intensities)
        axes[n].set_xlim([0, 112])
        axes[n].set_ylim([0, 112])
        axes[n].set_xticks([])
        axes[n].set_yticks([])
        if n > -1:
            yticks = axes[n].yaxis.get_major_ticks()
            xticks = axes[n].xaxis.get_major_ticks()
            for x, y in zip(xticks, yticks):
                y.label1.set_visible(False)
                x.label1.set_visible(False)

    # Wrapped Ridge
    gs01 = gridspec.GridSpecFromSubplotSpec(48, 60, subplot_spec=gs0[1], wspace=0.0, hspace=0.0)
    ax1 = plt.Subplot(fig, gs01[:, :48])
    fig.add_subplot(ax1)
    plt.title('Wrapped Ridge (WR)', fontsize=30, **hfont)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.annotate('b', xy=(1, 1), xycoords='axes fraction', fontsize=40, xytext=(0, 40),
                 textcoords='offset points', ha='left', va='top')

    axes = []
    for i in range(4):
        ax1 = plt.Subplot(fig, gs01[24*int(i/2):24*int(i/2+1), 24*(i%2):24*(i%2+1)])
        fig.add_subplot(ax1)
        axes.append(ax1)

    for n, res in enumerate(range(1, 5)):
        wrapped = plot_wrapper("ridge", res, axes[n], nonzero_frequency_mask)
        axes[n].set_xlim([0, 112])
        axes[n].set_ylim([0, 112])
        axes[n].set_xticks([])
        axes[n].set_yticks([])
        if n > -1:
            yticks = axes[n].yaxis.get_major_ticks()
            xticks = axes[n].xaxis.get_major_ticks()
            for x, y in zip(xticks, yticks):
                y.label1.set_visible(False)
                x.label1.set_visible(False)

    ax2 = plt.Subplot(fig, gs01[6:42, 49:53])
    fig.add_subplot(ax2)
    cb = plt.colorbar(wrapped, ticks=[], cax=ax2)
    cb.ax.text(0.5, -0.01, 'Min', transform=cb.ax.transAxes, va='top', ha='center', size=25, **hfont)
    cb.ax.text(0.5, 1.0, 'Max', transform=cb.ax.transAxes, va='bottom', ha='center', size=25, **hfont)
    cb.set_label("Importance", size=30, rotation=270, labelpad=-8, **hfont)

    # Wrapped Lasso
    gs03 = gridspec.GridSpecFromSubplotSpec(48, 60, subplot_spec=gs0[3], wspace=0.0, hspace=0.0)
    ax1 = plt.Subplot(fig, gs03[:, :48])
    fig.add_subplot(ax1)
    plt.title('Wrapped Lasso (WL)', fontsize=30, **hfont)
    ax1.annotate('d', xy=(1, 1), xycoords='axes fraction', fontsize=40, xytext=(0, 40),
                 textcoords='offset points', ha='left', va='top')
    ax1.set_xticks([])
    ax1.set_yticks([])
    axes = []
    for i in range(4):
        ax1 = plt.Subplot(fig, gs03[24*int(i/2):24*int(i/2+1), 24*(i%2):24*(i%2+1)])
        fig.add_subplot(ax1)
        axes.append(ax1)

    for n, res in enumerate(range(1, 5)):
        wrapped = plot_wrapper("lasso", res, axes[n], nonzero_frequency_mask)
        axes[n].set_xlim([0, 112])
        axes[n].set_ylim([0, 112])
        axes[n].set_xticks([])
        axes[n].set_yticks([])
        if n > -1:
            yticks = axes[n].yaxis.get_major_ticks()
            xticks = axes[n].xaxis.get_major_ticks()
            for x, y in zip(xticks, yticks):
                y.label1.set_visible(False)
                x.label1.set_visible(False)

    ax2 = plt.Subplot(fig, gs03[6:42, 49:53])
    fig.add_subplot(ax2)
    cb = plt.colorbar(wrapped, ticks=[], cax=ax2)
    cb.ax.text(0.5, -0.01, 'Min', transform=cb.ax.transAxes, va='top', ha='center', size=25, **hfont)
    cb.ax.text(0.5, 1.0, 'Max', transform=cb.ax.transAxes, va='bottom', ha='center', size=25, **hfont)
    cb.set_label("Importance", size=30, rotation=270, labelpad=-8, **hfont)

    # Embedded GP
    gs05 = gridspec.GridSpecFromSubplotSpec(48, 60, subplot_spec=gs0[5], wspace=0.0, hspace=0.0)
    ax1 = plt.Subplot(fig, gs05[:, :48])
    fig.add_subplot(ax1)
    plt.title('Embedded (GPESA)', fontsize=30, **hfont)
    embedded = get_gpesa_location_relevance(ax1, nonzero_frequency_mask)
    ax1.annotate('f', xy=(1, 1), xycoords='axes fraction', fontsize=40, xytext=(0, 40),
                 textcoords='offset points', ha='left', va='top')

    ax2 = plt.Subplot(fig, gs05[6:42, 49:53])
    fig.add_subplot(ax2)
    cb = plt.colorbar(embedded, ticks=[], cax=ax2)
    cb.ax.text(0.5, -0.01, 'Min', transform=cb.ax.transAxes, va='top', ha='center', size=25, **hfont)
    cb.ax.text(0.5, 1.0, 'Max', transform=cb.ax.transAxes, va='bottom', ha='center', size=25, **hfont)
    cb.set_label("Importance", size=30, rotation=270, labelpad=-8, **hfont)

    #
    plt.savefig("/Users/mecl/gp_mecl/exp/swe/all_heatmaps.pdf")
    plt.show()


if __name__ == "__main__":
    plot_multiple_grids(range(1, 21), rows=4)


