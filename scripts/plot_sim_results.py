#! /usr/bin/env python

import sys
import os
import re
import errno
import math
import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
_LOG = logging.getLogger(os.path.basename(__file__))

import pycoevolity
import project_util

import matplotlib as mpl

# Use TrueType (42) fonts rather than Type 3 fonts
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
tex_font_settings = {
        "text.usetex": True,
        "font.family": "sans-serif",
        "text.latex.preamble" : [
                "\\usepackage[T1]{fontenc}",
                "\\usepackage[cm]{sfmath}",
                ]
}

mpl.rcParams.update(tex_font_settings)

import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy.stats
import numpy


class BoxData(object):
    def __init__(self, values = [], labels = []):
        assert len(values) == len(labels)
        self._values = values
        self._labels = labels

    @classmethod
    def init(cls, results, parameters, labels = None):
        if labels:
            assert len(parameters) == len(labels)
        bd = cls()
        bd._values = [[] for i in range(len(parameters))]
        if labels:
            bd._labels = list(labels)
        else:
            bd._labels = list(parameters)
        for i, parameter_str in enumerate(parameters):
            x = [float(e) for e in results["{0}".format(parameter_str)]]
            bd._d[i].append(x)
        return bd

    def _get_values(self):
        return self._values

    values = property(_get_values)

    def _get_labels(self):
        return self._labels

    labels = property(_get_labels)

    def _get_number_of_categories(self):
        return len(self._values)

    number_of_categories = property(_get_number_of_categories)


    @classmethod
    def init_time_v_sharing(cls, results,
            estimator_prefix = "mean"):

        bd_abs_error = cls()
        bd_ci_width = cls()

        nreplicates = len(results["true_model"])
        ncomparisons = len(results["true_model"][0])

        labels = list(range(ncomparisons))
        err_vals = [[] for l in labels]
        ci_vals = [[] for l in labels]

        for sim_index in range(nreplicates):
            true_model = [int(x) for x in list(results["true_model"][sim_index])]
            assert len(true_model) == ncomparisons
            # Only use first comparison so as not to multply count the same
            # parameter estimates
            comparison_index = 0
            # for comparison_index in range(ncomparisons):
            num_shared = true_model.count(true_model[comparison_index]) - 1
            height_key_suffix = "root_height_c{0}sp1".format(comparison_index + 1)
            true_height = float(results["true_{0}".format(height_key_suffix)][sim_index])
            est_height = float(results["{0}_{1}".format(estimator_prefix, height_key_suffix)][sim_index])
            ci_lower = float(results["eti_95_lower_{0}".format(height_key_suffix)][sim_index])
            ci_upper = float(results["eti_95_upper_{0}".format(height_key_suffix)][sim_index])
            ci_width = ci_upper - ci_lower
            abs_error = math.fabs(true_height - est_height)

            err_vals[num_shared].append(abs_error)
            ci_vals[num_shared].append(ci_width)

        bd_abs_error._values = err_vals
        bd_ci_width._values = ci_vals
        bd_abs_error._labels = labels
        bd_ci_width._labels = labels
        return bd_abs_error, bd_ci_width


class HistogramData(object):
    def __init__(self, x = []):
        self._x = x

    @classmethod
    def init(cls, results, parameters, parameter_is_discrete):
        d = cls()
        d._x = []
        for parameter_str in parameters:
            if parameter_is_discrete:
                d._x.extend(int(x) for x in results["{0}".format(parameter_str)])
            else:
                d._x.extend(float(x) for x in results["{0}".format(parameter_str)])
        return d

    def _get_x(self):
        return self._x
    x = property(_get_x)


class ScatterData(object):
    def __init__(self,
            x = [],
            y = [],
            y_lower = [],
            y_upper = [],
            highlight_values = [],
            highlight_threshold = None,
            highlight_greater_than = True):
        self._x = x
        self._y = y
        self._y_lower = y_lower
        self._y_upper = y_upper
        self._highlight_values = highlight_values
        self._highlight_threshold = highlight_threshold
        self._highlight_greater_than = highlight_greater_than
        self._vet_data()
        self._highlight_indices = []
        self._populate_highlight_indices()
        self.highlight_color = (184 / 255.0, 90 / 255.0, 13 / 255.0) # pauburn

    @classmethod
    def init(cls, results, parameters,
            highlight_parameter_prefix = None,
            highlight_threshold = None,
            highlight_greater_than = True):
        d = cls()
        d._x = []
        d._y = []
        d._y_lower = []
        d._y_upper = []
        d._highlight_threshold = highlight_threshold
        d._highlight_values = []
        d._highlight_indices = []
        for parameter_str in parameters:
            d._x.extend(float(x) for x in results["true_{0}".format(parameter_str)])
            d._y.extend(float(x) for x in results["mean_{0}".format(parameter_str)])
            d._y_lower.extend(float(x) for x in results["eti_95_lower_{0}".format(parameter_str)])
            d._y_upper.extend(float(x) for x in results["eti_95_upper_{0}".format(parameter_str)])
            if highlight_parameter_prefix:
                d._highlight_values.extend(float(x) for x in results["{0}_{1}".format(
                        highlight_parameter_prefix,
                        parameter_str)])
        d._vet_data()
        d._populate_highlight_indices()
        return d

    @classmethod
    def init_time_v_sharing(cls, results,
            estimator_prefix = "mean",
            highlight_parameter_prefix = "psrf",
            highlight_threshold = 1.2,
            highlight_greater_than = True):
        d_abs_error = cls()
        d_abs_error._x = []
        d_abs_error._y = []
        d_abs_error._y_lower = []
        d_abs_error._y_upper = []
        d_abs_error._highlight_threshold = highlight_threshold
        d_abs_error._highlight_values = []
        d_abs_error._highlight_indices = []
        d_ci_width = cls()
        d_ci_width._x = []
        d_ci_width._y = []
        d_ci_width._y_lower = []
        d_ci_width._y_upper = []
        d_ci_width._highlight_threshold = highlight_threshold
        d_ci_width._highlight_values = []
        d_ci_width._highlight_indices = []
        nreplicates = len(results["true_model"])
        ncomparisons = len(results["true_model"][0])
        for sim_index in range(nreplicates):
            true_model = [int(x) for x in list(results["true_model"][sim_index])]
            assert len(true_model) == ncomparisons
            # Only use first comparison so as not to multply count the same
            # parameter estimates
            comparison_index = 0
            # for comparison_index in range(ncomparisons):
            num_shared = true_model.count(true_model[comparison_index])
            height_key_suffix = "root_height_c{0}sp1".format(comparison_index + 1)
            true_height = float(results["true_{0}".format(height_key_suffix)][sim_index])
            est_height = float(results["{0}_{1}".format(estimator_prefix, height_key_suffix)][sim_index])
            ci_lower = float(results["eti_95_lower_{0}".format(height_key_suffix)][sim_index])
            ci_upper = float(results["eti_95_upper_{0}".format(height_key_suffix)][sim_index])
            ci_width = ci_upper - ci_lower
            abs_error = math.fabs(true_height - est_height)
            d_abs_error._x.append(num_shared)
            d_ci_width._x.append(num_shared)
            d_abs_error._y.append(abs_error)
            d_ci_width._y.append(ci_width)
            if highlight_parameter_prefix:
                d_abs_error._highlight_values.append(float(results["{0}_{1}".format(
                        highlight_parameter_prefix,
                        height_key_suffix)][sim_index]))
                d_ci_width._highlight_values.append(float(results["{0}_{1}".format(
                        highlight_parameter_prefix,
                        height_key_suffix)][sim_index]))
        d_abs_error._vet_data()
        d_ci_width._vet_data()
        d_abs_error._populate_highlight_indices()
        d_ci_width._populate_highlight_indices()
        return d_abs_error, d_ci_width

    def _vet_data(self):
        assert len(self._x) == len(self._y)
        if self._y_lower:
            assert len(self._x) == len(self._y_lower)
        if self._y_upper:
            assert len(self._x) == len(self._y_upper)
        if self._highlight_values:
            assert len(self._x) == len(self._highlight_values)

    def _populate_highlight_indices(self):
        if (self._highlight_values) and (self._highlight_threshold is not None):
            for i in range(len(self._x)):
                if self.highlight(i):
                    self._highlight_indices.append(i)

    def has_y_ci(self):
        return bool(self.y_lower) and bool(self.y_upper)

    def has_highlights(self):
        return bool(self._highlight_indices)

    def _get_x(self):
        return self._x
    x = property(_get_x)

    def _get_y(self):
        return self._y
    y = property(_get_y)

    def _get_y_lower(self):
        return self._y_lower
    y_lower = property(_get_y_lower)

    def _get_y_upper(self):
        return self._y_upper
    y_upper = property(_get_y_upper)

    def _get_highlight_indices(self):
        return self._highlight_indices
    highlight_indices = property(_get_highlight_indices)

    def _get_highlight_x(self):
        return [self._x[i] for i in self._highlight_indices]
    highlight_x = property(_get_highlight_x)

    def _get_highlight_y(self):
        return [self._y[i] for i in self._highlight_indices]
    highlight_y = property(_get_highlight_y)

    def _get_highlight_y_lower(self):
        return [self._y_lower[i] for i in self._highlight_indices]
    highlight_y_lower = property(_get_highlight_y_lower)

    def _get_highlight_y_upper(self):
        return [self._y_upper[i] for i in self._highlight_indices]
    highlight_y_upper = property(_get_highlight_y_upper)

    def highlight(self, index):
        if (not self._highlight_values) or (self._highlight_threshold is None):
            return False

        if self._highlight_greater_than:
            if self._highlight_values[index] > self._highlight_threshold:
                return True
            else:
                return False
        else:
            if self._highlight_values[index] < self._highlight_threshold:
                return True
            else:
                return False
        return False


def get_abs_error(true, estimate):
    return math.fabs(true - estimate)

def get_relative_estimate(true, estimate):
    return estimate / float(true)

def get_relative_error(true, estimate):
    return math.fabs(true - estimate) / true

def plot_gamma(shape = 1.0,
        scale = 1.0,
        offset = 0.0,
        x_min = 0.0,
        x_max = None,
        number_of_points = 1000,
        x_label = "Relative root size",
        include_x_label = True,
        include_y_label = True,
        include_title = True,
        curve_line_width = 1.5,
        curve_line_style = '-',
        curve_line_color = (57 / 255.0, 115 / 255.0, 124 / 255.0),
        one_line_width = 1.0,
        one_line_style = '--',
        one_line_color = (184 / 255.0, 90 / 255.0, 13 / 255.0),
        plot_width = 3.5,
        plot_height = 3.0,
        xy_label_size = 16.0,
        title_size = 16.0,
        pad_left = 0.2,
        pad_right = 0.99,
        pad_bottom = 0.18,
        pad_top = 0.9,
        plot_file_prefix = None,
        plot_dir = project_util.PLOT_DIR,
        ):
    if x_max is None:
        x_max = scipy.stats.gamma.ppf(0.999, shape, scale = scale)
        x_max_plot = x_max + offset
    else:
        x_max_plot = x_max
    x = numpy.linspace(x_min, x_max, number_of_points)
    d = scipy.stats.gamma.pdf(x, shape, scale = scale)

    x_plot = [v + offset for v in x]

    if offset > 0.0:
        x_gap = numpy.linspace(x_min, offset, 100)
        d_gap = [0.0 for i in range(100)]
        x_plot = list(x_gap) + list(x_plot)
        d = d_gap + list(d)

    plt.close('all')
    fig = plt.figure(figsize = (plot_width, plot_height))
    gs = gridspec.GridSpec(1, 1,
            wspace = 0.0,
            hspace = 0.0)
    ax = plt.subplot(gs[0, 0])
    line = ax.plot(x_plot, d)
    ax.set_xlim(x_min, x_max_plot)
    plt.setp(line,
            color = curve_line_color,
            linestyle = curve_line_style,
            linewidth = curve_line_width,
            marker = '',
            zorder = 100)
    ax.axvline(x = 1.0,
            color = one_line_color,
            linestyle = one_line_style,
            linewidth = one_line_width,
            marker = '',
            zorder = 0)
    if include_x_label:
        ax.set_xlabel(
                "{0}".format(x_label),
                fontsize = xy_label_size)
    if include_y_label:
        ax.set_ylabel(
                "Density",
                fontsize = xy_label_size)
    if include_title:
        if offset == 0.0:
            col_header = "$\\textrm{{\\sffamily Gamma}}({0:.0f}, {1})$\n$\\textrm{{\\sffamily mean}} = {2}$".format(shape, scale, (shape * scale) + offset)
        else:
            col_header = "$\\textrm{{\\sffamily Gamma}}({0:.0f}, {1})$\n$\\textrm{{\\sffamily offset}} = {2}$ $\\textrm{{\\sffamily mean}} = {3}$".format(shape, scale, offset, (shape * scale) + offset)
        ax.set_title(col_header,
                fontsize = title_size)

    gs.update(
            left = pad_left,
            right = pad_right,
            bottom = pad_bottom,
            top = pad_top)

    plot_path = os.path.join(plot_dir,
            "{0}-gamma.pdf".format(plot_file_prefix))
    plt.savefig(plot_path)
    _LOG.info("Plot written to {0!r}".format(plot_path))

def get_sequence_iter(start = 0.0, stop = 1.0, n = 10):
    assert(stop > start)
    step = (stop - start) / float(n - 1)
    return ((start + (i * step)) for i in range(n))

def truncate_color_map(cmap, min_val = 0.0, max_val = 10, n = 100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(
                    n = cmap.name,
                    a = min_val,
                    b = max_val),
            cmap(list(get_sequence_iter(min_val, max_val, n))))
    return new_cmap

def get_errors(values, lowers, uppers):
    n = len(values)
    assert(n == len(lowers))
    assert(n == len(uppers))
    return [[values[i] - lowers[i] for i in range(n)],
            [uppers[i] - values[i] for i in range(n)]]

def ci_width_iter(results, parameter_str):
    n = len(results["eti_95_upper_{0}".format(parameter_str)])
    for i in range(n):
        upper = float(results["eti_95_upper_{0}".format(parameter_str)][i])
        lower = float(results["eti_95_lower_{0}".format(parameter_str)][i])
        yield upper - lower

def absolute_error_iter(results, parameter_str):
    n = len(results["true_{0}".format(parameter_str)])
    for i in range(n):
        t = float(results["true_{0}".format(parameter_str)][i])
        e = float(results["mean_{0}".format(parameter_str)][i])
        yield math.fabs(t - e)

def generate_scatter_plot_grid(
        data_grid,
        plot_file_prefix,
        parameter_symbol = "t",
        column_labels = None,
        row_labels = None,
        plot_width = 1.9,
        plot_height = 1.8,
        pad_left = 0.1,
        pad_right = 0.98,
        pad_bottom = 0.12,
        pad_top = 0.92,
        x_label = None,
        x_label_size = 18.0,
        y_label = None,
        y_label_size = 18.0,
        force_shared_x_range = True,
        force_shared_y_range = True,
        force_shared_xy_ranges = True,
        force_shared_spines = True,
        include_coverage = True,
        include_rmse = True,
        include_identity_line = True,
        include_error_bars = True,
        plot_dir = project_util.PLOT_DIR
        ):
    if force_shared_spines or force_shared_xy_ranges:
        force_shared_x_range = True
        force_shared_y_range = True

    if row_labels:
        assert len(row_labels) ==  len(data_grid)
    if column_labels:
        assert len(column_labels) == len(data_grid[0])

    nrows = len(data_grid)
    ncols = len(data_grid[0])

    x_min = float('inf')
    x_max = float('-inf')
    y_min = float('inf')
    y_max = float('-inf')
    for row_index, data_grid_row in enumerate(data_grid):
        for column_index, data in enumerate(data_grid_row):
            x_min = min(x_min, min(data.x))
            x_max = max(x_max, max(data.x))
            y_min = min(y_min, min(data.y))
            y_max = max(y_max, max(data.y))
    if force_shared_xy_ranges:
        mn = min(x_min, y_min)
        mx = max(x_max, y_max)
        x_min = mn
        y_min = mn
        x_max = mx
        y_max = mx
    x_buffer = math.fabs(x_max - x_min) * 0.05
    x_axis_min = x_min - x_buffer
    x_axis_max = x_max + x_buffer
    y_buffer = math.fabs(y_max - y_min) * 0.05
    y_axis_min = y_min - y_buffer
    y_axis_max = y_max + y_buffer


    plt.close('all')
    w = plot_width
    h = plot_height
    fig_width = (ncols * w)
    fig_height = (nrows * h)
    fig = plt.figure(figsize = (fig_width, fig_height))
    if force_shared_spines:
        gs = gridspec.GridSpec(nrows, ncols,
                wspace = 0.0,
                hspace = 0.0)
    else:
        gs = gridspec.GridSpec(nrows, ncols)

    for row_index, data_grid_row in enumerate(data_grid):
        for column_index, data in enumerate(data_grid_row):
            proportion_within_ci = 0.0
            if include_coverage and data.has_y_ci():
                proportion_within_ci = pycoevolity.stats.get_proportion_of_values_within_intervals(
                        data.x,
                        data.y_lower,
                        data.y_upper)
            rmse = 0.0
            if include_rmse:
                rmse = pycoevolity.stats.root_mean_square_error(data.x, data.y)
            ax = plt.subplot(gs[row_index, column_index])
            if include_error_bars and data.has_y_ci():
                line = ax.errorbar(
                        x = data.x,
                        y = data.y,
                        yerr = get_errors(data.y, data.y_lower, data.y_upper),
                        ecolor = '0.65',
                        elinewidth = 0.5,
                        capsize = 0.8,
                        barsabove = False,
                        marker = 'o',
                        linestyle = '',
                        markerfacecolor = 'none',
                        markeredgecolor = '0.35',
                        markeredgewidth = 0.7,
                        markersize = 2.5,
                        zorder = 100,
                        rasterized = True)
                if data.has_highlights():
                    line, = ax.plot(data.highlight_x, data.highlight_y)
                    plt.setp(line,
                            marker = 'o',
                            linestyle = '',
                            markerfacecolor = 'none',
                            markeredgecolor = data.highlight_color,
                            markeredgewidth = 0.7,
                            markersize = 2.5,
                            zorder = 200,
                            rasterized = True)
            else:
                line, = ax.plot(data.x, data.y)
                plt.setp(line,
                        marker = 'o',
                        linestyle = '',
                        markerfacecolor = 'none',
                        markeredgecolor = '0.35',
                        markeredgewidth = 0.7,
                        markersize = 2.5,
                        zorder = 100,
                        rasterized = True)
                if data.has_highlights():
                    line, = ax.plot(data.highlight_x, data.highlight_y)
                    plt.setp(line,
                            marker = 'o',
                            linestyle = '',
                            markerfacecolor = 'none',
                            markeredgecolor = data.highlight_color,
                            markeredgewidth = 0.7,
                            markersize = 2.5,
                            zorder = 200,
                            rasterized = True)
            if force_shared_x_range:
                ax.set_xlim(x_axis_min, x_axis_max)
            else:
                ax.set_xlim(min(data.x), max(data.x))
            if force_shared_y_range:
                ax.set_ylim(y_axis_min, y_axis_max)
            else:
                ax.set_ylim(min(data.y), max(data.y))
            if include_identity_line:
                identity_line, = ax.plot(
                        [x_axis_min, x_axis_max],
                        [y_axis_min, y_axis_max])
                plt.setp(identity_line,
                        color = '0.7',
                        linestyle = '-',
                        linewidth = 1.0,
                        marker = '',
                        zorder = 0)
            if include_coverage:
                ax.text(0.02, 0.97,
                        "\\scriptsize\\noindent$p({0:s} \\in \\textrm{{\\sffamily CI}}) = {1:.3f}$".format(
                                parameter_symbol,
                                proportion_within_ci),
                        horizontalalignment = "left",
                        verticalalignment = "top",
                        transform = ax.transAxes,
                        size = 6.0,
                        zorder = 300,
                        bbox = {
                            'facecolor': 'white',
                            'edgecolor': 'white',
                            'pad': 2}
                        )
            if include_rmse:
                text_y = 0.97
                if include_coverage:
                    text_y = 0.87
                ax.text(0.02, text_y,
                        "\\scriptsize\\noindent RMSE = {0:.2e}".format(
                                rmse),
                        horizontalalignment = "left",
                        verticalalignment = "top",
                        transform = ax.transAxes,
                        size = 6.0,
                        zorder = 300,
                        bbox = {
                            'facecolor': 'white',
                            'edgecolor': 'white',
                            'pad': 2}
                        )
            if column_labels and (row_index == 0):
                col_header = column_labels[column_index]
                ax.text(0.5, 1.015,
                        col_header,
                        horizontalalignment = "center",
                        verticalalignment = "bottom",
                        transform = ax.transAxes)
            if row_labels and (column_index == (ncols - 1)):
                row_label = row_labels[row_index]
                ax.text(1.015, 0.5,
                        row_label,
                        horizontalalignment = "left",
                        verticalalignment = "center",
                        rotation = 270.0,
                        transform = ax.transAxes)

    if force_shared_spines:
        # show only the outside ticks
        all_axes = fig.get_axes()
        for ax in all_axes:
            if not ax.is_last_row():
                ax.set_xticks([])
            if not ax.is_first_col():
                ax.set_yticks([])

        # show tick labels only for lower-left plot 
        all_axes = fig.get_axes()
        for ax in all_axes:
            if ax.is_last_row() and ax.is_first_col():
                continue
            xtick_labels = ["" for item in ax.get_xticklabels()]
            ytick_labels = ["" for item in ax.get_yticklabels()]
            ax.set_xticklabels(xtick_labels)
            ax.set_yticklabels(ytick_labels)

        # avoid doubled spines
        all_axes = fig.get_axes()
        for ax in all_axes:
            for sp in ax.spines.values():
                sp.set_visible(False)
                sp.set_linewidth(2)
            if ax.is_first_row():
                ax.spines['top'].set_visible(True)
                ax.spines['bottom'].set_visible(True)
            else:
                ax.spines['bottom'].set_visible(True)
            if ax.is_first_col():
                ax.spines['left'].set_visible(True)
                ax.spines['right'].set_visible(True)
            else:
                ax.spines['right'].set_visible(True)

    if x_label:
        fig.text(0.5, 0.001,
                x_label,
                horizontalalignment = "center",
                verticalalignment = "bottom",
                size = x_label_size)
    if y_label:
        fig.text(0.005, 0.5,
                y_label,
                horizontalalignment = "left",
                verticalalignment = "center",
                rotation = "vertical",
                size = y_label_size)

    gs.update(left = pad_left,
            right = pad_right,
            bottom = pad_bottom,
            top = pad_top)

    plot_path = os.path.join(plot_dir,
            "{0}-scatter.pdf".format(plot_file_prefix))
    plt.savefig(plot_path, dpi=600)
    _LOG.info("Plots written to {0!r}".format(plot_path))

def generate_scatter_plot(
        data,
        plot_file_prefix,
        parameter_symbol = "t",
        title = None,
        title_size = 16.0,
        x_label = None,
        x_label_size = 16.0,
        y_label = None,
        y_label_size = 16.0,
        plot_width = 3.5,
        plot_height = 3.0,
        pad_left = 0.2,
        pad_right = 0.99,
        pad_bottom = 0.18,
        pad_top = 0.9,
        force_shared_xy_ranges = True,
        xy_limits = None,
        include_coverage = True,
        include_rmse = True,
        include_identity_line = True,
        include_error_bars = True,
        plot_dir = project_util.PLOT_DIR):

    if xy_limits:
        x_axis_min, x_axis_max, y_axis_min, y_axis_max = xy_limits
    else:
        x_min = min(data.x)
        x_max = max(data.x)
        y_min = min(data.y)
        y_max = max(data.y)
        if force_shared_xy_ranges:
            mn = min(x_min, y_min)
            mx = max(x_max, y_max)
            x_min = mn
            y_min = mn
            x_max = mx
            y_max = mx
        x_buffer = math.fabs(x_max - x_min) * 0.05
        x_axis_min = x_min - x_buffer
        x_axis_max = x_max + x_buffer
        y_buffer = math.fabs(y_max - y_min) * 0.05
        y_axis_min = y_min - y_buffer
        y_axis_max = y_max + y_buffer

    plt.close('all')
    fig = plt.figure(figsize = (plot_width, plot_height))
    gs = gridspec.GridSpec(1, 1,
            wspace = 0.0,
            hspace = 0.0)

    proportion_within_ci = 0.0
    if include_coverage and data.has_y_ci():
        proportion_within_ci = pycoevolity.stats.get_proportion_of_values_within_intervals(
                data.x,
                data.y_lower,
                data.y_upper)
    rmse = 0.0
    if include_rmse:
        rmse = pycoevolity.stats.root_mean_square_error(data.x, data.y)
    ax = plt.subplot(gs[0, 0])
    if include_error_bars and data.has_y_ci():
        line = ax.errorbar(
                x = data.x,
                y = data.y,
                yerr = get_errors(data.y, data.y_lower, data.y_upper),
                ecolor = '0.65',
                elinewidth = 0.5,
                capsize = 0.8,
                barsabove = False,
                marker = 'o',
                linestyle = '',
                markerfacecolor = 'none',
                markeredgecolor = '0.35',
                markeredgewidth = 0.7,
                markersize = 2.5,
                zorder = 100,
                rasterized = True)
        if data.has_highlights():
            # line = ax.errorbar(
            #         x = data.highlight_x,
            #         y = data.highlight_y,
            #         yerr = get_errors(data.highlight_y,
            #                 data.highlight_y_lower,
            #                 data.highlight_y_upper),
            #         ecolor = data.highlight_color,
            #         elinewidth = 0.5,
            #         capsize = 0.8,
            #         barsabove = False,
            #         marker = 'o',
            #         linestyle = '',
            #         markerfacecolor = 'none',
            #         markeredgecolor = data.highlight_color,
            #         markeredgewidth = 0.7,
            #         markersize = 2.5,
            #         zorder = 200,
            #         rasterized = True)
            line, = ax.plot(data.highlight_x, data.highlight_y)
            plt.setp(line,
                    marker = 'o',
                    linestyle = '',
                    markerfacecolor = 'none',
                    markeredgecolor = data.highlight_color,
                    markeredgewidth = 0.7,
                    markersize = 2.5,
                    zorder = 200,
                    rasterized = True)
    else:
        line, = ax.plot(data.x, data.y)
        plt.setp(line,
                marker = 'o',
                linestyle = '',
                markerfacecolor = 'none',
                markeredgecolor = '0.35',
                markeredgewidth = 0.7,
                markersize = 2.5,
                zorder = 100,
                rasterized = True)
        if data.has_highlights():
            line, = ax.plot(x = data.highlight_x, y = data.highlight_y)
            plt.setp(line,
                    marker = 'o',
                    linestyle = '',
                    markerfacecolor = 'none',
                    markeredgecolor = data.highlight_color,
                    markeredgewidth = 0.7,
                    markersize = 2.5,
                    zorder = 200,
                    rasterized = True)
    ax.set_xlim(x_axis_min, x_axis_max)
    ax.set_ylim(y_axis_min, y_axis_max)
    if include_identity_line:
        identity_line, = ax.plot(
                [x_axis_min, x_axis_max],
                [y_axis_min, y_axis_max])
        plt.setp(identity_line,
                color = '0.7',
                linestyle = '-',
                linewidth = 1.0,
                marker = '',
                zorder = 0)
    if include_coverage:
        ax.text(0.02, 0.97,
                "\\normalsize\\noindent$p({0:s} \\in \\textrm{{\\sffamily CI}}) = {1:.3f}$".format(
                        parameter_symbol,
                        proportion_within_ci),
                horizontalalignment = "left",
                verticalalignment = "top",
                transform = ax.transAxes,
                size = 8.0,
                zorder = 300,
                bbox = {
                    'facecolor': 'white',
                    'edgecolor': 'white',
                    'pad': 2}
                )
    if include_rmse:
        text_y = 0.97
        if include_coverage:
            text_y = 0.87
        ax.text(0.02, text_y,
                "\\normalsize\\noindent RMSE = {0:.2e}".format(
                        rmse),
                horizontalalignment = "left",
                verticalalignment = "top",
                transform = ax.transAxes,
                size = 8.0,
                zorder = 300,
                bbox = {
                    'facecolor': 'white',
                    'edgecolor': 'white',
                    'pad': 2}
                )
    if x_label is not None:
        ax.set_xlabel(
                x_label,
                fontsize = x_label_size)
    if y_label is not None:
        ax.set_ylabel(
                y_label,
                fontsize = y_label_size)
    if title is not None:
        ax.set_title(plot_title,
                fontsize = title_size)

    gs.update(
            left = pad_left,
            right = pad_right,
            bottom = pad_bottom,
            top = pad_top)

    plot_path = os.path.join(plot_dir,
            "{0}-scatter.pdf".format(plot_file_prefix))
    plt.savefig(plot_path, dpi=600)
    _LOG.info("Plots written to {0!r}".format(plot_path))


def generate_box_plot(
        data,
        plot_file_prefix,
        title = None,
        title_size = 16.0,
        x_label = None,
        x_label_size = 16.0,
        y_label = None,
        y_label_size = 16.0,
        plot_width = 3.5,
        plot_height = 3.0,
        pad_left = 0.2,
        pad_right = 0.99,
        pad_bottom = 0.18,
        pad_top = 0.9,
        jitter = 0.01,
        alpha = 0.4,
        rasterized = False,
        plot_dir = project_util.PLOT_DIR):

    plt.close('all')
    fig = plt.figure(figsize = (plot_width, plot_height))
    gs = gridspec.GridSpec(1, 1,
            wspace = 0.0,
            hspace = 0.0)

    ax = plt.subplot(gs[0, 0])
    box_dict = ax.boxplot(data.values,
            labels = data.labels,
            notch = False,
            sym = '',
            vert = True,
            showfliers = False,
            whis = 'range',
            zorder = 500)
    # Change median line from default color to black
    plt.setp(box_dict["medians"], color = "black")
    for i in range(data.number_of_categories):
        vals = data.values[i]
        x = numpy.random.uniform(low = i + 1 - jitter, high = i + 1 + jitter, size = len(vals))
        ax.plot(x, vals,
                marker = 'o',
                linestyle = '',
                markerfacecolor = 'none',
                markeredgecolor = '0.35',
                markeredgewidth = 0.7,
                alpha = alpha,
                markersize = 2.5,
                zorder = 100,
                rasterized = rasterized)
    if x_label is not None:
        ax.set_xlabel(
                x_label,
                fontsize = x_label_size)
    if y_label is not None:
        ax.set_ylabel(
                y_label,
                fontsize = y_label_size)
    if title is not None:
        ax.set_title(plot_title,
                fontsize = title_size)

    gs.update(
            left = pad_left,
            right = pad_right,
            bottom = pad_bottom,
            top = pad_top)

    plot_path = os.path.join(plot_dir,
            "{0}-box.pdf".format(plot_file_prefix))
    if rasterized:
        plt.savefig(plot_path, dpi=600)
    else:
        plt.savefig(plot_path)
    _LOG.info("Box plot written to {0!r}".format(plot_path))


def generate_histogram_grid(
        data_grid,
        plot_file_prefix,
        column_labels = None,
        row_labels = None,
        parameter_label = "Number of variable sites",
        range_key = "range",
        center_key = "mean",
        number_of_digits = 0,
        plot_width = 1.9,
        plot_height = 1.8,
        pad_left = 0.1,
        pad_right = 0.98,
        pad_bottom = 0.12,
        pad_top = 0.92,
        force_shared_x_range = True,
        force_shared_bins = True,
        force_shared_y_range = True,
        force_shared_spines = True,
        plot_dir = project_util.PLOT_DIR
        ):
    if force_shared_spines:
        force_shared_x_range = True
        force_shared_y_range = True

    if row_labels:
        assert len(row_labels) ==  len(data_grid)
    if column_labels:
        assert len(column_labels) == len(data_grid[0])

    nrows = len(data_grid)
    ncols = len(data_grid[0])

    x_min = float('inf')
    x_max = float('-inf')
    for row_index, data_grid_row in enumerate(data_grid):
        for column_index, data in enumerate(data_grid_row):
            x_min = min(x_min, min(data.x))
            x_max = max(x_max, max(data.x))

    axis_buffer = math.fabs(x_max - x_min) * 0.05
    axis_min = x_min - axis_buffer
    axis_max = x_max + axis_buffer

    plt.close('all')
    w = plot_width
    h = plot_height
    fig_width = (ncols * w)
    fig_height = (nrows * h)
    fig = plt.figure(figsize = (fig_width, fig_height))
    if force_shared_spines:
        gs = gridspec.GridSpec(nrows, ncols,
                wspace = 0.0,
                hspace = 0.0)
    else:
        gs = gridspec.GridSpec(nrows, ncols)

    hist_bins = None
    x_range = None
    if force_shared_x_range:
        x_range = (x_min, x_max)
    for row_index, data_grid_row in enumerate(data_grid):
        for column_index, data in enumerate(data_grid_row):
            summary = pycoevolity.stats.get_summary(data.x)
            _LOG.info("0.025, 0.975 quantiles: {0:.2f}, {1:.2f}".format(
                    summary["qi_95"][0],
                    summary["qi_95"][1]))

            ax = plt.subplot(gs[row_index, column_index])
            n, bins, patches = ax.hist(data.x,
                    weights = [1.0 / float(len(data.x))] * len(data.x),
                    bins = hist_bins,
                    range = x_range,
                    cumulative = False,
                    histtype = 'bar',
                    align = 'mid',
                    orientation = 'vertical',
                    rwidth = None,
                    log = False,
                    color = None,
                    edgecolor = '0.5',
                    facecolor = '0.5',
                    fill = True,
                    hatch = None,
                    label = None,
                    linestyle = None,
                    linewidth = None,
                    zorder = 10,
                    )
            if (hist_bins is None) and force_shared_bins:
                hist_bins = bins
            ax.text(0.98, 0.98,
                    "\\scriptsize {mean:,.{ndigits}f} ({lower:,.{ndigits}f}--{upper:,.{ndigits}f})".format(
                            mean = summary[center_key],
                            lower = summary[range_key][0],
                            upper = summary[range_key][1],
                            ndigits = number_of_digits),
                    horizontalalignment = "right",
                    verticalalignment = "top",
                    transform = ax.transAxes,
                    zorder = 200,
                    bbox = {
                        'facecolor': 'white',
                        'edgecolor': 'white',
                        'pad': 2}
                    )

            if column_labels and (row_index == 0):
                col_header = column_labels[column_index]
                ax.text(0.5, 1.015,
                        col_header,
                        horizontalalignment = "center",
                        verticalalignment = "bottom",
                        transform = ax.transAxes)
            if row_labels and (column_index == (ncols - 1)):
                row_label = row_labels[row_index]
                ax.text(1.015, 0.5,
                        row_label,
                        horizontalalignment = "left",
                        verticalalignment = "center",
                        rotation = 270.0,
                        transform = ax.transAxes)

    if force_shared_y_range:
        all_axes = fig.get_axes()
        # y_max = float('-inf')
        # for ax in all_axes:
        #     ymn, ymx = ax.get_ylim()
        #     y_max = max(y_max, ymx)
        for ax in all_axes:
            ax.set_ylim(0.0, 1.0)

    if force_shared_spines:
        # show only the outside ticks
        all_axes = fig.get_axes()
        for ax in all_axes:
            if not ax.is_last_row():
                ax.set_xticks([])
            if not ax.is_first_col():
                ax.set_yticks([])

        # show tick labels only for lower-left plot 
        all_axes = fig.get_axes()
        for ax in all_axes:
            if ax.is_last_row() and ax.is_first_col():
                continue
            xtick_labels = ["" for item in ax.get_xticklabels()]
            ytick_labels = ["" for item in ax.get_yticklabels()]
            ax.set_xticklabels(xtick_labels)
            ax.set_yticklabels(ytick_labels)

        # avoid doubled spines
        all_axes = fig.get_axes()
        for ax in all_axes:
            for sp in ax.spines.values():
                sp.set_visible(False)
                sp.set_linewidth(2)
            if ax.is_first_row():
                ax.spines['top'].set_visible(True)
                ax.spines['bottom'].set_visible(True)
            else:
                ax.spines['bottom'].set_visible(True)
            if ax.is_first_col():
                ax.spines['left'].set_visible(True)
                ax.spines['right'].set_visible(True)
            else:
                ax.spines['right'].set_visible(True)

    fig.text(0.5, 0.001,
            parameter_label,
            horizontalalignment = "center",
            verticalalignment = "bottom",
            size = 18.0)
    fig.text(0.005, 0.5,
            # "Density",
            "Frequency",
            horizontalalignment = "left",
            verticalalignment = "center",
            rotation = "vertical",
            size = 18.0)

    gs.update(left = pad_left,
            right = pad_right,
            bottom = pad_bottom,
            top = pad_top)

    plot_path = os.path.join(plot_dir,
            "{0}-histograms.pdf".format(plot_file_prefix))
    plt.savefig(plot_path)
    _LOG.info("Plots written to {0!r}".format(plot_path))


def generate_histogram(
        data,
        plot_file_prefix,
        title = None,
        title_size = 16.0,
        x_label = None,
        x_label_size = 16.0,
        y_label = None,
        y_label_size = 16.0,
        plot_width = 3.5,
        plot_height = 3.0,
        pad_left = 0.2,
        pad_right = 0.99,
        pad_bottom = 0.18,
        pad_top = 0.9,
        bins = None,
        x_range = None,
        range_key = "range",
        center_key = "mean",
        number_of_digits = 0,
        plot_dir = project_util.PLOT_DIR
        ):

    plt.close('all')
    fig = plt.figure(figsize = (plot_width, plot_height))
    gs = gridspec.GridSpec(1, 1,
            wspace = 0.0,
            hspace = 0.0)

    summary = pycoevolity.stats.get_summary(data.x)
    _LOG.info("0.025, 0.975 quantiles: {0:.2f}, {1:.2f}".format(
            summary["qi_95"][0],
            summary["qi_95"][1]))

    ax = plt.subplot(gs[0, 0])
    n, b, patches = ax.hist(data.x,
            weights = [1.0 / float(len(data.x))] * len(data.x),
            bins = bins,
            range = x_range,
            cumulative = False,
            histtype = 'bar',
            align = 'mid',
            orientation = 'vertical',
            rwidth = None,
            log = False,
            color = None,
            edgecolor = '0.5',
            facecolor = '0.5',
            fill = True,
            hatch = None,
            label = None,
            linestyle = None,
            linewidth = None,
            zorder = 10,
            )

    ax.text(0.98, 0.98,
            "{mean:,.{ndigits}f} ({lower:,.{ndigits}f}--{upper:,.{ndigits}f})".format(
                    mean = summary[center_key],
                    lower = summary[range_key][0],
                    upper = summary[range_key][1],
                    ndigits = number_of_digits),
            horizontalalignment = "right",
            verticalalignment = "top",
            transform = ax.transAxes,
            zorder = 200,
            bbox = {
                'facecolor': 'white',
                'edgecolor': 'white',
                'pad': 2}
            )


    if x_label is not None:
        ax.set_xlabel(
                x_label,
                fontsize = x_label_size)
    if y_label is not None:
        ax.set_ylabel(
                y_label,
                fontsize = y_label_size)
    if title is not None:
        ax.set_title(plot_title,
                fontsize = title_size)

    gs.update(
            left = pad_left,
            right = pad_right,
            bottom = pad_bottom,
            top = pad_top)

    plot_path = os.path.join(plot_dir,
            "{0}-histogram.pdf".format(plot_file_prefix))
    plt.savefig(plot_path)
    _LOG.info("Plots written to {0!r}".format(plot_path))


def generate_model_plot_grid(
        results_grid,
        column_labels = None,
        row_labels = None,
        number_of_comparisons = 3,
        plot_width = 1.6,
        plot_height = 1.5,
        pad_left = 0.1,
        pad_right = 0.98,
        pad_bottom = 0.12,
        pad_top = 0.92,
        y_label_size = 18.0,
        y_label = None,
        number_font_size = 12.0,
        force_shared_spines = True,
        plot_as_histogram = False,
        histogram_correct_values = [],
        plot_file_prefix = None,
        plot_dir = project_util.PLOT_DIR
        ):
    _LOG.info("Generating model plots...")

    cmap = truncate_color_map(plt.cm.binary, 0.0, 0.65, 100)

    if row_labels:
        assert len(row_labels) ==  len(results_grid)
    if column_labels:
        assert len(column_labels) == len(results_grid[0])

    nrows = len(results_grid)
    ncols = len(results_grid[0])

    plt.close('all')
    w = plot_width
    h = plot_height
    fig_width = (ncols * w)
    fig_height = (nrows * h)
    fig = plt.figure(figsize = (fig_width, fig_height))
    if force_shared_spines:
        gs = gridspec.GridSpec(nrows, ncols,
                wspace = 0.0,
                hspace = 0.0)
    else:
        gs = gridspec.GridSpec(nrows, ncols,
                wspace = 0.0,
                hspace = 0.37)

    for row_index, results_grid_row in enumerate(results_grid):
        for column_index, results in enumerate(results_grid_row):
            true_map_nevents = []
            true_map_nevents_probs = []
            for i in range(number_of_comparisons):
                true_map_nevents.append([0 for i in range(number_of_comparisons)])
                true_map_nevents_probs.append([[] for i in range(number_of_comparisons)])
            true_nevents = tuple(int(x) for x in results["true_num_events"])
            map_nevents = tuple(int(x) for x in results["map_num_events"])
            true_nevents_cred_levels = tuple(float(x) for x in results["true_num_events_cred_level"])
            # true_model_cred_levels = tuple(float(x) for x in results["true_model_cred_level"])
            assert(len(true_nevents) == len(map_nevents))
            assert(len(true_nevents) == len(true_nevents_cred_levels))
            # assert(len(true_nevents) == len(true_model_cred_levels))

            true_nevents_probs = []
            map_nevents_probs = []
            for i in range(len(true_nevents)):
                true_nevents_probs.append(float(
                    results["num_events_{0}_p".format(true_nevents[i])][i]))
                map_nevents_probs.append(float(
                    results["num_events_{0}_p".format(map_nevents[i])][i]))
            assert(len(true_nevents) == len(true_nevents_probs))
            assert(len(true_nevents) == len(map_nevents_probs))

            mean_true_nevents_prob = sum(true_nevents_probs) / len(true_nevents_probs)
            median_true_nevents_prob = pycoevolity.stats.median(true_nevents_probs)

            nevents_within_95_cred = 0
            # model_within_95_cred = 0
            ncorrect = 0
            for i in range(len(true_nevents)):
                true_map_nevents[map_nevents[i] - 1][true_nevents[i] - 1] += 1
                true_map_nevents_probs[map_nevents[i] - 1][true_nevents[i] - 1].append(map_nevents_probs[i])
                if true_nevents_cred_levels[i] <= 0.95:
                    nevents_within_95_cred += 1
                # if true_model_cred_levels[i] <= 0.95:
                #     model_within_95_cred += 1
                if true_nevents[i] == map_nevents[i]:
                    ncorrect += 1
            p_nevents_within_95_cred = nevents_within_95_cred / float(len(true_nevents))
            # p_model_within_95_cred = model_within_95_cred / float(len(true_nevents))
            p_correct = ncorrect / float(len(true_nevents))

            _LOG.info("p(nevents within CS) = {0:.4f}".format(p_nevents_within_95_cred))
            # _LOG.info("p(model within CS) = {0:.4f}".format(p_model_within_95_cred))
            ax = plt.subplot(gs[row_index, column_index])

            if plot_as_histogram:
                total_nevent_estimates = len(map_nevents)
                nevents_indices = [float(x) for x in range(number_of_comparisons)]
                nevents_counts = [0 for x in nevents_indices]
                for k in map_nevents:
                    nevents_counts[k - 1] += 1
                nevents_freqs = [
                        (x / float(total_nevent_estimates)) for x in nevents_counts
                        ]
                assert len(nevents_indices) == len(nevents_freqs)
                bar_width = 0.9
                bar_color = "0.5"
                bars_posterior = ax.bar(
                        nevents_indices,
                        nevents_freqs,
                        bar_width,
                        color = bar_color,
                        label = "MAP")
                x_tick_labels = [str(i + 1) for i in range(number_of_comparisons)]
                plt.xticks(
                        nevents_indices,
                        x_tick_labels
                        )
                if histogram_correct_values:
                    correct_val = histogram_correct_values[row_index][column_index]
                    correct_line, = ax.plot(
                            [correct_val - 1, correct_val - 1],
                            [0.0, 1.0])
                    plt.setp(correct_line,
                            color = '0.7',
                            linestyle = '--',
                            linewidth = 1.0,
                            marker = '',
                            zorder = 200)
            else:
                ax.imshow(true_map_nevents,
                        origin = 'lower',
                        cmap = cmap,
                        interpolation = 'none',
                        aspect = 'auto'
                        # extent = [0.5, 3.5, 0.5, 3.5]
                        )
                for i, row_list in enumerate(true_map_nevents):
                    for j, num_events in enumerate(row_list):
                        ax.text(j, i,
                                str(num_events),
                                horizontalalignment = "center",
                                verticalalignment = "center",
                                size = number_font_size)

            upper_text_y = 1.02
            lower_text_y = 1.11
            upper_text_valign = "bottom"
            lower_text_valign = "bottom"
            if force_shared_spines:
                upper_text_y = 0.98
                lower_text_y = 0.02
                upper_text_valign = "top"
                lower_text_valign = "bottom"

            if force_shared_spines:
                ax.text(0.98, lower_text_y,
                        "\\scriptsize$p(k \\in \\textrm{{\\sffamily CS}}) = {0:.3f}$".format(
                                p_nevents_within_95_cred),
                        horizontalalignment = "right",
                        verticalalignment = lower_text_valign,
                        transform = ax.transAxes,
                        zorder = 300,
                        bbox = {
                            'facecolor': 'white',
                            'edgecolor': 'white',
                            'pad': 2}
                        )
                ax.text(0.02, upper_text_y,
                        "\\scriptsize$p(\\hat{{k}} = k) = {0:.3f}$".format(
                                p_correct),
                        horizontalalignment = "left",
                        verticalalignment = upper_text_valign,
                        transform = ax.transAxes,
                        zorder = 300,
                        bbox = {
                            'facecolor': 'white',
                            'edgecolor': 'white',
                            'pad': 2}
                        )
                ax.text(0.98, upper_text_y,
                        "\\scriptsize$\\widetilde{{p(k|\\mathbf{{D}})}} = {0:.3f}$".format(
                                median_true_nevents_prob),
                        horizontalalignment = "right",
                        verticalalignment = upper_text_valign,
                        transform = ax.transAxes,
                        zorder = 300,
                        bbox = {
                            'facecolor': 'white',
                            'edgecolor': 'white',
                            'pad': 2}
                        )
            else:
                ax.text(0.02, lower_text_y,
                        "\\scriptsize$p(k \\in \\textrm{{\\sffamily CS}}) = {0:.3f}$".format(
                                p_nevents_within_95_cred),
                        horizontalalignment = "left",
                        verticalalignment = lower_text_valign,
                        transform = ax.transAxes,
                        zorder = 300,
                        # bbox = {
                        #     'facecolor': 'white',
                        #     'edgecolor': 'white',
                        #     'pad': 0}
                        )
                ax.text(0.02, upper_text_y,
                        "\\scriptsize$p(\\hat{{k}} = k) = {0:.3f}$".format(
                                p_correct),
                        horizontalalignment = "left",
                        verticalalignment = upper_text_valign,
                        transform = ax.transAxes,
                        zorder = 300,
                        # bbox = {
                        #     'facecolor': 'white',
                        #     'edgecolor': 'white',
                        #     'pad': 0}
                        )
                ax.text(0.98, upper_text_y,
                        "\\scriptsize$\\widetilde{{p(k|\\mathbf{{D}})}} = {0:.3f}$".format(
                                median_true_nevents_prob),
                        horizontalalignment = "right",
                        verticalalignment = upper_text_valign,
                        transform = ax.transAxes,
                        zorder = 300,
                        # bbox = {
                        #     'facecolor': 'white',
                        #     'edgecolor': 'white',
                        #     'pad': 0}
                        )

            col_text_y = 1.2
            if force_shared_spines:
                col_text_y = 1.015
            if column_labels and (row_index == 0):
                col_header = column_labels[column_index]
                ax.text(0.5, col_text_y,
                        col_header,
                        horizontalalignment = "center",
                        verticalalignment = "bottom",
                        transform = ax.transAxes)
            if row_labels and (column_index == (ncols - 1)):
                row_label = row_labels[row_index]
                ax.text(1.015, 0.5,
                        row_label,
                        horizontalalignment = "left",
                        verticalalignment = "center",
                        rotation = 270.0,
                        transform = ax.transAxes)

    all_axes = fig.get_axes()
    if plot_as_histogram:
        for ax in all_axes:
            ax.set_ylim(0.0, 1.0)
            # Make sure ticks correspond only with number of events
            # ax.xaxis.set_ticks(range(number_of_comparisons))
            # xtick_labels = [item for item in ax.get_xticklabels()]
            # for i in range(len(xtick_labels)):
            #     xtick_labels[i].set_text(str(i + 1))
            # ax.set_xticklabels(xtick_labels)

    if force_shared_spines:
        # show only the outside ticks
        for ax in all_axes:
            if not ax.is_last_row():
                ax.set_xticks([])
            if not ax.is_first_col():
                ax.set_yticks([])

        # show tick labels only for lower-left plot 
        for ax in all_axes:
            # Make sure ticks correspond only with number of events
            if not plot_as_histogram:
                ax.xaxis.set_ticks(range(number_of_comparisons))
                ax.yaxis.set_ticks(range(number_of_comparisons))
            if ax.is_last_row() and ax.is_first_col():
                if not plot_as_histogram:
                    xtick_labels = [item for item in ax.get_xticklabels()]
                    for i in range(len(xtick_labels)):
                        xtick_labels[i].set_text(str(i + 1))
                    ax.set_xticklabels(xtick_labels)
                    ytick_labels = [item for item in ax.get_yticklabels()]
                    for i in range(len(ytick_labels)):
                        ytick_labels[i].set_text(str(i + 1))
                    ax.set_yticklabels(ytick_labels)
            else:
                xtick_labels = ["" for item in ax.get_xticklabels()]
                ytick_labels = ["" for item in ax.get_yticklabels()]
                ax.set_xticklabels(xtick_labels)
                ax.set_yticklabels(ytick_labels)

        # avoid doubled spines
        for ax in all_axes:
            for sp in ax.spines.values():
                sp.set_visible(False)
                sp.set_linewidth(1.2)
            if ax.is_first_row():
                ax.spines['top'].set_visible(True)
                ax.spines['bottom'].set_visible(True)
            else:
                ax.spines['bottom'].set_visible(True)
            if ax.is_first_col():
                ax.spines['left'].set_visible(True)
                ax.spines['right'].set_visible(True)
            else:
                ax.spines['right'].set_visible(True)
    else:
        # show tick labels only for left plots
        for ax in all_axes:
            # Make sure ticks correspond only with number of events
            if not plot_as_histogram:
                ax.xaxis.set_ticks(range(number_of_comparisons))
                ax.yaxis.set_ticks(range(number_of_comparisons))
                xtick_labels = [item for item in ax.get_xticklabels()]
                for i in range(len(xtick_labels)):
                    xtick_labels[i].set_text(str(i + 1))
                ax.set_xticklabels(xtick_labels)
            if ax.is_first_col():
                if not plot_as_histogram:
                    ytick_labels = [item for item in ax.get_yticklabels()]
                    for i in range(len(ytick_labels)):
                        ytick_labels[i].set_text(str(i + 1))
                    ax.set_yticklabels(ytick_labels)
            else:
                ytick_labels = ["" for item in ax.get_yticklabels()]
                ax.set_yticklabels(ytick_labels)

        # show only the outside ticks
        for ax in all_axes:
            if not ax.is_first_col():
                ax.set_yticks([])

        # avoid doubled spines
        for ax in all_axes:
            for sp in ax.spines.values():
                sp.set_visible(False)
                sp.set_linewidth(1.2)
            ax.spines['top'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            if ax.is_first_col():
                ax.spines['left'].set_visible(True)
                ax.spines['right'].set_visible(True)
            else:
                ax.spines['right'].set_visible(True)


    if plot_as_histogram:
        fig.text(0.5, 0.001,
                "Estimated number of events ($\\hat{{k}}$)",
                horizontalalignment = "center",
                verticalalignment = "bottom",
                size = 18.0)
    else:
        fig.text(0.5, 0.001,
                "True number of events ($k$)",
                horizontalalignment = "center",
                verticalalignment = "bottom",
                size = 18.0)
    if y_label is None:
        if plot_as_histogram:
            y_label = "Frequency"
        else:
            y_label = "Estimated number of events ($\\hat{{k}}$)"
    fig.text(0.005, 0.5,
            y_label,
            horizontalalignment = "left",
            verticalalignment = "center",
            rotation = "vertical",
            size = 18.0)

    gs.update(left = pad_left,
            right = pad_right,
            bottom = pad_bottom,
            top = pad_top)

    if plot_file_prefix:
        plot_path = os.path.join(plot_dir,
                "{0}-nevents.pdf".format(plot_file_prefix))
    else:
        plot_path = os.path.join(plot_dir,
                "nevents.pdf")
    plt.savefig(plot_path)
    _LOG.info("Plots written to {0!r}\n".format(plot_path))

def generate_model_plot(
        results,
        number_of_comparisons = 3,
        show_all_models = False,
        plot_title = None,
        include_x_label = True,
        include_y_label = True,
        include_median = True,
        include_cs = True,
        include_prop_correct = True,
        plot_width = 3.5,
        plot_height = 3.0,
        xy_label_size = 16.0,
        title_size = 16.0,
        pad_left = 0.2,
        pad_right = 0.99,
        pad_bottom = 0.18,
        pad_top = 0.9,
        lower_annotation_y = 0.02,
        upper_annotation_y = 0.92,
        plot_file_prefix = None,
        plot_dir = project_util.PLOT_DIR,
        model_key = "model",
        num_events_key = "num_events",
        ):
    if show_all_models and (number_of_comparisons != 3):
        raise Exception("show all models only supported for 3 comparisons")
    _LOG.info("Generating model plots...")

    cmap = truncate_color_map(plt.cm.binary, 0.0, 0.65, 100)

    model_to_index = {
            "000": 0,
            "001": 1,
            "010": 2,
            "011": 3,
            "012": 4,
            }
    index_to_model = {}
    for k, v in model_to_index.items():
        index_to_model[v] = k

    plt.close('all')
    fig = plt.figure(figsize = (plot_width, plot_height))
    gs = gridspec.GridSpec(1, 1,
            wspace = 0.0,
            hspace = 0.0)

    true_map_nevents = []
    true_map_model = []
    true_map_nevents_probs = []
    for i in range(number_of_comparisons):
        true_map_nevents.append([0 for i in range(number_of_comparisons)])
        true_map_nevents_probs.append([[] for i in range(number_of_comparisons)])
    for i in range(5):
        true_map_model.append([0 for i in range(5)])
        true_map_nevents_probs.append([[] for i in range(5)])
    true_nevents = tuple(int(x) for x in results["true_{num_events}".format(num_events = num_events_key)])
    map_nevents = tuple(int(x) for x in results["map_{num_events}".format(num_events = num_events_key)])
    true_model = tuple(x for x in results["true_{model}".format(model = model_key)])
    map_model = tuple(x for x in results["map_{model}".format(model = model_key)])
    true_nevents_cred_levels = tuple(float(x) for x in results["true_{num_events}_cred_level".format(num_events = num_events_key)])
    true_model_cred_levels = tuple(float(x) for x in results["true_{model}_cred_level".format(model = model_key)])
    assert(len(true_nevents) == len(map_nevents))
    assert(len(true_nevents) == len(true_nevents_cred_levels))
    assert(len(true_nevents) == len(true_model_cred_levels))
    assert(len(true_nevents) == len(true_model))
    assert(len(true_nevents) == len(map_model))

    true_nevents_probs = []
    map_nevents_probs = []
    for i in range(len(true_nevents)):
        true_nevents_probs.append(float(
            results["{num_events}_{n}_p".format(num_events = num_events_key, n = true_nevents[i])][i]))
        map_nevents_probs.append(float(
            results["{num_events}_{n}_p".format(num_events = num_events_key, n = map_nevents[i])][i]))
    assert(len(true_nevents) == len(true_nevents_probs))
    assert(len(true_nevents) == len(map_nevents_probs))

    mean_true_nevents_prob = sum(true_nevents_probs) / len(true_nevents_probs)
    median_true_nevents_prob = pycoevolity.stats.median(true_nevents_probs)

    true_model_probs = tuple(float(x) for x in results["true_{model}_p".format(model = model_key)])
    assert(len(true_nevents) == len(true_model_probs))

    mean_true_model_prob = sum(true_model_probs) / len(true_model_probs)
    median_true_model_prob = pycoevolity.stats.median(true_model_probs)

    nevents_within_95_cred = 0
    model_within_95_cred = 0
    ncorrect = 0
    model_ncorrect = 0
    for i in range(len(true_nevents)):
        true_map_nevents[map_nevents[i] - 1][true_nevents[i] - 1] += 1
        true_map_nevents_probs[map_nevents[i] - 1][true_nevents[i] - 1].append(map_nevents_probs[i])
        if show_all_models:
            true_map_model[model_to_index[map_model[i]]][model_to_index[true_model[i]]] += 1
        if true_nevents_cred_levels[i] <= 0.95:
            nevents_within_95_cred += 1
        if true_model_cred_levels[i] <= 0.95:
            model_within_95_cred += 1
        if true_nevents[i] == map_nevents[i]:
            ncorrect += 1
        if true_model[i] == map_model[i]:
            model_ncorrect += 1
    p_nevents_within_95_cred = nevents_within_95_cred / float(len(true_nevents))
    p_model_within_95_cred = model_within_95_cred / float(len(true_nevents))
    p_correct = ncorrect / float(len(true_nevents))
    p_model_correct = model_ncorrect /  float(len(true_nevents))

    _LOG.info("p(nevents within CS) = {0:.4f}".format(p_nevents_within_95_cred))
    _LOG.info("p(model within CS) = {0:.4f}".format(p_model_within_95_cred))
    ax = plt.subplot(gs[0, 0])

    if show_all_models:
        ax.imshow(true_map_model,
                origin = 'lower',
                cmap = cmap,
                interpolation = 'none',
                aspect = 'auto'
                )
        for i, row_list in enumerate(true_map_model):
            for j, n in enumerate(row_list):
                ax.text(j, i,
                        str(n),
                        horizontalalignment = "center",
                        verticalalignment = "center",
                        # fontsize = 8,
                        )
    else:
        ax.imshow(true_map_nevents,
                origin = 'lower',
                cmap = cmap,
                interpolation = 'none',
                aspect = 'auto'
                )
        for i, row_list in enumerate(true_map_nevents):
            for j, num_events in enumerate(row_list):
                ax.text(j, i,
                        str(num_events),
                        horizontalalignment = "center",
                        verticalalignment = "center")

    if include_cs:
        if show_all_models:
            ax.text(0.98, lower_annotation_y,
                    "$p(\\mathcal{{T}} \\in \\textrm{{\\sffamily CS}}) = {0:.3f}$".format(
                            p_model_within_95_cred),
                    horizontalalignment = "right",
                    verticalalignment = "bottom",
                    transform = ax.transAxes,
                    zorder = 300,
                    bbox = {
                        'facecolor': 'white',
                        'edgecolor': 'white',
                        'pad': 2}
                    )
        else:
            ax.text(0.98, lower_annotation_y,
                    "$p(k \\in \\textrm{{\\sffamily CS}}) = {0:.3f}$".format(
                            p_nevents_within_95_cred),
                    horizontalalignment = "right",
                    verticalalignment = "bottom",
                    transform = ax.transAxes,
                    zorder = 300,
                    bbox = {
                        'facecolor': 'white',
                        'edgecolor': 'white',
                        'pad': 2}
                    )
    if include_prop_correct:
        if show_all_models:
            ax.text(0.02, upper_annotation_y,
                    "$p(\\hat{{\\mathcal{{T}}}} = \\mathcal{{T}}) = {0:.3f}$".format(
                            p_model_correct),
                    horizontalalignment = "left",
                    verticalalignment = "bottom",
                    transform = ax.transAxes,
                    zorder = 300,
                    bbox = {
                        'facecolor': 'white',
                        'edgecolor': 'white',
                        'pad': 2}
                    )
        else:
            ax.text(0.02, upper_annotation_y,
                    "$p(\\hat{{k}} = k) = {0:.3f}$".format(
                            p_correct),
                    horizontalalignment = "left",
                    verticalalignment = "bottom",
                    transform = ax.transAxes,
                    zorder = 300,
                    bbox = {
                        'facecolor': 'white',
                        'edgecolor': 'white',
                        'pad': 2}
                    )
    if include_median:
        if show_all_models:
            ax.text(0.98, upper_annotation_y,
                    "$\\widetilde{{p(\\mathcal{{T}}|\\mathbf{{D}})}} = {0:.3f}$".format(
                            median_true_model_prob),
                    horizontalalignment = "right",
                    verticalalignment = "bottom",
                    transform = ax.transAxes,
                    zorder = 300,
                    bbox = {
                        'facecolor': 'white',
                        'edgecolor': 'white',
                        'pad': 2}
                    )
        else:
            ax.text(0.98, upper_annotation_y,
                    "$\\widetilde{{p(k|\\mathbf{{D}})}} = {0:.3f}$".format(
                            median_true_nevents_prob),
                    horizontalalignment = "right",
                    verticalalignment = "bottom",
                    transform = ax.transAxes,
                    zorder = 300,
                    bbox = {
                        'facecolor': 'white',
                        'edgecolor': 'white',
                        'pad': 2}
                    )
    if include_x_label:
        if show_all_models:
            ax.set_xlabel("True model ($\\mathcal{{T}}$)",
                    # labelpad = 8.0,
                    fontsize = xy_label_size)
        else:
            ax.set_xlabel("True \\# of events ($k$)",
                    # labelpad = 8.0,
                    fontsize = xy_label_size)
    if include_y_label:
        if show_all_models:
            ax.set_ylabel("MAP model ($\\hat{{\\mathcal{{T}}}}$)",
                    labelpad = 8.0,
                    fontsize = xy_label_size)
        else:
            ax.set_ylabel("MAP \\# of events ($\\hat{{k}}$)",
                    labelpad = 8.0,
                    fontsize = xy_label_size)
    if plot_title:
        ax.set_title(plot_title,
                fontsize = title_size)

    # Make sure ticks correspond only with number of events or model
    if not show_all_models:
        ax.xaxis.set_ticks(range(number_of_comparisons))
        ax.yaxis.set_ticks(range(number_of_comparisons))
    else:
        ax.xaxis.set_ticks(range(5))
        ax.yaxis.set_ticks(range(5))
    xtick_labels = [item for item in ax.get_xticklabels()]
    for i in range(len(xtick_labels)):
        if show_all_models:
            xtick_labels[i].set_text(index_to_model[i])
        else:
            xtick_labels[i].set_text(str(i + 1))
    ytick_labels = [item for item in ax.get_yticklabels()]
    for i in range(len(ytick_labels)):
        if show_all_models:
            ytick_labels[i].set_text(index_to_model[i])
        else:
            ytick_labels[i].set_text(str(i + 1))
    ax.set_xticklabels(xtick_labels)
    ax.set_yticklabels(ytick_labels)

    gs.update(
            left = pad_left,
            right = pad_right,
            bottom = pad_bottom,
            top = pad_top)

    plot_path = os.path.join(plot_dir,
            "{0}-nevents.pdf".format(plot_file_prefix))
    plt.savefig(plot_path)
    _LOG.info("Plots written to {0!r}\n".format(plot_path))

def parse_results(paths):
    return pycoevolity.parsing.get_dict_from_spreadsheets(
            paths,
            sep = "\t",
            offset = 0)

def get_data_grid(data_dict, template):
    data_grid = []
    for row in template:
        r = []
        for cell in row: 
            r.append(data_dict[cell[0]][cell[1]])
        data_grid.append(r)
    return data_grid


def main_cli(argv = sys.argv):
    try:
        os.makedirs(project_util.PLOT_DIR)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    brooks_gelman_1998_recommended_psrf = 1.2

    pad_left = 0.16
    pad_right = 0.94
    pad_bottom = 0.12
    pad_top = 0.965
    plot_width = 2.8
    plot_height = 2.2

    all_sim_config_names = (
            "fixed-pairs-10-independent-time-1_0-0_05",
            "fixed-pairs-10-independent-time-1_0-0_05-chars-5000",
            "fixed-pairs-10-independent-time-1_0-0_05-chars-10000",
            "fixed-pairs-10-independent-time-1_0-0_05-chars-50000",
            "fixed-pairs-10-independent-time-1_0-0_05-chars-100000",
            "fixed-pairs-10-simultaneous-time-1_0-0_05",
            "pairs-10-dpp-conc-2_0-2_71-time-1_0-0_05",
            "pairs-10-pyp-conc-2_0-1_79-disc-1_0-4_0-time-1_0-0_05",
            "pairs-10-unif-sw-0_55-7_32-time-1_0-0_05",
            )
    sim_config_names = (
            "fixed-pairs-10-independent-time-1_0-0_05",
            "fixed-pairs-10-simultaneous-time-1_0-0_05",
            "pairs-10-dpp-conc-2_0-2_71-time-1_0-0_05",
            "pairs-10-pyp-conc-2_0-1_79-disc-1_0-4_0-time-1_0-0_05",
            "pairs-10-unif-sw-0_55-7_32-time-1_0-0_05",
            )
    nchars_sim_config_names = (
            "fixed-pairs-10-independent-time-1_0-0_05-chars-5000",
            "fixed-pairs-10-independent-time-1_0-0_05-chars-10000",
            "fixed-pairs-10-independent-time-1_0-0_05-chars-50000",
            "fixed-pairs-10-independent-time-1_0-0_05-chars-100000",
            "fixed-pairs-10-independent-time-1_0-0_05",
            )
    analysis_config_names = (
            "pairs-10-dpp-conc-2_0-2_71-time-1_0-0_05",
            "pairs-10-pyp-conc-2_0-1_79-disc-1_0-4_0-time-1_0-0_05",
            "pairs-10-unif-sw-0_55-7_32-time-1_0-0_05",
            )
    cfg_to_label = {
            "fixed-pairs-10-independent-time-1_0-0_05"                  : "Independent",
            "fixed-pairs-10-simultaneous-time-1_0-0_05"                 : "Simultaneous",
            "pairs-10-dpp-conc-2_0-2_71-time-1_0-0_05"                  : "DP",
            "pairs-10-pyp-conc-2_0-1_79-disc-1_0-4_0-time-1_0-0_05"     : "PYP",
            "pairs-10-unif-sw-0_55-7_32-time-1_0-0_05"                  : "Uniform",
            }
    cfg_to_correct_nevents = {
            "fixed-pairs-10-independent-time-1_0-0_05-chars-5000"       : 10,
            "fixed-pairs-10-independent-time-1_0-0_05-chars-10000"      : 10,
            "fixed-pairs-10-independent-time-1_0-0_05-chars-50000"      : 10,
            "fixed-pairs-10-independent-time-1_0-0_05-chars-100000"     : 10,
            "fixed-pairs-10-independent-time-1_0-0_05"                  : 10,
            "fixed-pairs-10-simultaneous-time-1_0-0_05"                 : 1,
            }
    nchars_cfg_to_label = {
            "fixed-pairs-10-independent-time-1_0-0_05-chars-5000"       : "5k",
            "fixed-pairs-10-independent-time-1_0-0_05-chars-10000"      : "10k",
            "fixed-pairs-10-independent-time-1_0-0_05-chars-50000"      : "50k",
            "fixed-pairs-10-independent-time-1_0-0_05-chars-100000"     : "100k",
            "fixed-pairs-10-independent-time-1_0-0_05"                  : "500k",
            }
    
    cfg_grid = tuple(
            tuple((row, col) for col in analysis_config_names 
                ) for row in sim_config_names)
    analysis_cfg_grid = tuple(
            tuple((row, col) for col in analysis_config_names 
                ) for row in analysis_config_names)
    fixed_cfg_grid = tuple(
            tuple((row, col) for col in analysis_config_names 
                ) for row in sim_config_names[0:2])
    nchars_cfg_grid = tuple(
            tuple((row, col) for col in analysis_config_names 
                ) for row in nchars_sim_config_names)
    
    column_labels = tuple(cfg_to_label[c] for c in analysis_config_names)
    row_labels = tuple(cfg_to_label[c] for c in sim_config_names)
    fixed_row_labels = tuple(cfg_to_label[c] for c in sim_config_names[0:2])
    nchars_row_labels = tuple(nchars_cfg_to_label[c] for c in nchars_sim_config_names)

    sys.stdout.write("Parsing results...\n")
    header = None
    results = {}
    var_only_results = {}
    for sim_cfg in all_sim_config_names:
        results[sim_cfg] = {}
        var_only_results[sim_cfg] = {}
        sim_dir = os.path.join(project_util.SIM_DIR, sim_cfg)
        batch_dirs = tuple(project_util.batch_dir_iter(sim_dir))
        for analysis_cfg in analysis_config_names:
            result_paths = [
                    os.path.join(
                        bd,
                        analysis_cfg + "-results.tsv.gz"
                        ) for bd in batch_dirs
                    ]
            results[sim_cfg][analysis_cfg] = parse_results(result_paths)
            header = sorted(results[sim_cfg][analysis_cfg])
            var_only_analysis_cfg = "var-only-" + analysis_cfg
            var_only_result_paths = [
                    os.path.join(
                        bd,
                        var_only_analysis_cfg + "-results.tsv.gz"
                        ) for bd in batch_dirs
                    ]
            var_only_results[sim_cfg][analysis_cfg] = parse_results(
                    var_only_result_paths)
            sys.stdout.write("\n")
            sys.stdout.write("Parsed results for data simulated under:\n")
            sys.stdout.write("    {0}\n".format(sim_cfg))
            sys.stdout.write("and analyzed under:\n")
            sys.stdout.write("    {0}\n".format(analysis_cfg))
            sys.stdout.write(
                    "    Number of reps using variable sites: {0}\n".format(
                        len(results[sim_cfg][analysis_cfg]["true_model"])))
            sys.stdout.write(
                    "    Number of reps ignoring variable sites: {0}\n".format(
                        len(var_only_results[sim_cfg][analysis_cfg]["true_model"])))

    results_grid = get_data_grid(results, cfg_grid)
    var_only_results_grid = get_data_grid(var_only_results, cfg_grid)

    analysis_results_grid = get_data_grid(results, analysis_cfg_grid)
    var_only_analysis_results_grid = get_data_grid(var_only_results, analysis_cfg_grid)

    fixed_results_grid = get_data_grid(results, fixed_cfg_grid)
    var_only_fixed_results_grid = get_data_grid(var_only_results, fixed_cfg_grid)
    fixed_correct_nevents_grid = []
    for row in fixed_cfg_grid:
        r = [cfg_to_correct_nevents[cell[0]] for cell in row]
        fixed_correct_nevents_grid.append(r)

    nchars_results_grid = get_data_grid(results, nchars_cfg_grid)
    var_only_nchars_results_grid = get_data_grid(var_only_results, nchars_cfg_grid)
    nchars_correct_nevents_grid = []
    for row in nchars_cfg_grid:
        r = [cfg_to_correct_nevents[cell[0]] for cell in row]
        nchars_correct_nevents_grid.append(r)

    height_parameters = tuple(
            h[5:] for h in header if h.startswith("mean_root_height_c")
    )
    root_size_parameters = tuple(
            h[5:] for h in header if h.startswith("mean_pop_size_root_c")
    )
    leaf_size_parameters = tuple(
            h[5:] for h in header if h.startswith("mean_pop_size_c")
    )

    parameters_to_plot = {
            "div-time": {
                    "headers": height_parameters,
                    "label": "divergence time",
                    "short_label": "time",
                    "symbol": "t",
                    "xy_limits": None,
                    "pad_left": pad_left,
            },
            "ancestor-size": {
                    "headers": root_size_parameters,
                    "label": "ancestor population size",
                    "short_label": "size",
                    "symbol": "N_e\\mu",
                    "xy_limits": None,
                    "pad_left": pad_left + 0.01,
            },
            "descendant-size": {
                    "headers": leaf_size_parameters,
                    "label": "descendant population size",
                    "short_label": "size",
                    "symbol": "N_e\\mu",
                    "xy_limits": None,
                    "pad_left": pad_left,
            },
    }


    for parameter, p_info in parameters_to_plot.items():
        data = {}
        var_only_data = {}
        for sim_cfg in results:
            data[sim_cfg] = {}
            var_only_data[sim_cfg] = {}
            for analysis_cfg in results[sim_cfg]:
                data[sim_cfg][analysis_cfg] = ScatterData.init(
                        results[sim_cfg][analysis_cfg],
                        p_info["headers"],
                        highlight_parameter_prefix = "psrf",
                        highlight_threshold = brooks_gelman_1998_recommended_psrf,
                        )
                var_only_data[sim_cfg][analysis_cfg] = ScatterData.init(
                        var_only_results[sim_cfg][analysis_cfg],
                        p_info["headers"],
                        highlight_parameter_prefix = "psrf",
                        highlight_threshold = brooks_gelman_1998_recommended_psrf,
                        )
        
        x_label = "True {0} (${1}$)".format(
                p_info["label"],
                p_info["symbol"])
        y_label = "Estimated {0} ($\\hat{{{1}}}$)".format(
                p_info["label"],
                p_info["symbol"])

        data_grid = get_data_grid(data, cfg_grid)
        var_only_data_grid = get_data_grid(var_only_data, cfg_grid)

        prefix = "infer-columns-by-data-rows-" + parameter
        generate_scatter_plot_grid(
                data_grid = data_grid,
                plot_file_prefix = prefix,
                parameter_symbol = p_info["symbol"],
                column_labels = column_labels,
                row_labels = row_labels,
                plot_width = plot_width,
                plot_height = plot_height,
                pad_left = p_info["pad_left"],
                pad_right = pad_right,
                pad_bottom = pad_bottom,
                pad_top = pad_top,
                x_label = x_label,
                x_label_size = 18.0,
                y_label = y_label,
                y_label_size = 18.0,
                force_shared_x_range = True,
                force_shared_y_range = True,
                force_shared_xy_ranges = True,
                force_shared_spines = True,
                include_coverage = True,
                include_rmse = True,
                include_identity_line = True,
                include_error_bars = True,
                plot_dir = project_util.PLOT_DIR)
        generate_scatter_plot_grid(
                data_grid = var_only_data_grid,
                plot_file_prefix = "var-only-" + prefix,
                parameter_symbol = p_info["symbol"],
                column_labels = column_labels,
                row_labels = row_labels,
                plot_width = plot_width,
                plot_height = plot_height,
                pad_left = p_info["pad_left"],
                pad_right = pad_right,
                pad_bottom = pad_bottom,
                pad_top = pad_top,
                x_label = x_label,
                x_label_size = 18.0,
                y_label = y_label,
                y_label_size = 18.0,
                force_shared_x_range = True,
                force_shared_y_range = True,
                force_shared_xy_ranges = True,
                force_shared_spines = True,
                include_coverage = True,
                include_rmse = True,
                include_identity_line = True,
                include_error_bars = True,
                plot_dir = project_util.PLOT_DIR)

        # Plot results for varying dataset size
        data_grid = get_data_grid(data, nchars_cfg_grid)
        var_only_data_grid = get_data_grid(var_only_data, nchars_cfg_grid)

        prefix = "nchars-" + parameter
        generate_scatter_plot_grid(
                data_grid = data_grid,
                plot_file_prefix = prefix,
                parameter_symbol = p_info["symbol"],
                column_labels = column_labels,
                row_labels = nchars_row_labels,
                plot_width = plot_width,
                plot_height = plot_height,
                pad_left = p_info["pad_left"],
                pad_right = pad_right,
                pad_bottom = pad_bottom,
                pad_top = pad_top,
                x_label = x_label,
                x_label_size = 18.0,
                y_label = y_label,
                y_label_size = 18.0,
                force_shared_x_range = True,
                force_shared_y_range = True,
                force_shared_xy_ranges = True,
                force_shared_spines = True,
                include_coverage = True,
                include_rmse = True,
                include_identity_line = True,
                include_error_bars = True,
                plot_dir = project_util.PLOT_DIR)
        generate_scatter_plot_grid(
                data_grid = var_only_data_grid,
                plot_file_prefix = "var-only-" + prefix,
                parameter_symbol = p_info["symbol"],
                column_labels = column_labels,
                row_labels = nchars_row_labels,
                plot_width = plot_width,
                plot_height = plot_height,
                pad_left = p_info["pad_left"],
                pad_right = pad_right,
                pad_bottom = pad_bottom,
                pad_top = pad_top,
                x_label = x_label,
                x_label_size = 18.0,
                y_label = y_label,
                y_label_size = 18.0,
                force_shared_x_range = True,
                force_shared_y_range = True,
                force_shared_xy_ranges = True,
                force_shared_spines = True,
                include_coverage = True,
                include_rmse = True,
                include_identity_line = True,
                include_error_bars = True,
                plot_dir = project_util.PLOT_DIR)

    # Generate concentration/split weight plots
    for config_name in analysis_config_names:
        data = ScatterData.init(
                results[config_name][config_name],
                ["concentration"],
                highlight_parameter_prefix = "psrf",
                highlight_threshold = brooks_gelman_1998_recommended_psrf,
                )
        var_only_data = ScatterData.init(
                var_only_results[config_name][config_name],
                ["concentration"],
                highlight_parameter_prefix = "psrf",
                highlight_threshold = brooks_gelman_1998_recommended_psrf,
                )
        
        x_label = "True concentration"
        y_label = "Estimated concentration"
        if config_name.startswith("pairs-10-unif"):
            x_label = "True split weight"
            y_label = "Estimated split weight"


        prefix = "concentration-" + cfg_to_label[config_name]

        generate_scatter_plot(
                data = data,
                plot_file_prefix = prefix,
                parameter_symbol = "\\alpha",
                x_label = x_label,
                y_label = y_label,
                include_coverage = True,
                include_rmse = True,
                include_identity_line = True,
                include_error_bars = True,
                plot_dir = project_util.PLOT_DIR)
        generate_scatter_plot(
                data = var_only_data,
                plot_file_prefix = "var-only-" + prefix,
                parameter_symbol = "\\alpha",
                x_label = x_label,
                y_label = y_label,
                include_coverage = True,
                include_rmse = True,
                include_identity_line = True,
                include_error_bars = True,
                plot_dir = project_util.PLOT_DIR)

    # Generate model plots
    prefix = "infer-columns-by-data-rows"
    generate_model_plot_grid(
            results_grid = analysis_results_grid,
            column_labels = column_labels,
            row_labels = column_labels,
            number_of_comparisons = len(height_parameters),
            plot_width = plot_width - 0.6,
            plot_height = plot_height,
            pad_left = pad_left - 0.07,
            pad_right = pad_right,
            pad_bottom = pad_bottom - 0.035,
            pad_top = pad_top - 0.032,
            y_label_size = 18.0,
            y_label = None,
            number_font_size = 10.0,
            force_shared_spines = False,
            plot_as_histogram = False,
            histogram_correct_values = [],
            plot_file_prefix = prefix,
            plot_dir = project_util.PLOT_DIR)
    generate_model_plot_grid(
            results_grid = var_only_analysis_results_grid,
            column_labels = column_labels,
            row_labels = column_labels,
            number_of_comparisons = len(height_parameters),
            plot_width = plot_width - 0.6,
            plot_height = plot_height,
            pad_left = pad_left - 0.07,
            pad_right = pad_right,
            pad_bottom = pad_bottom - 0.035,
            pad_top = pad_top - 0.032,
            y_label_size = 18.0,
            y_label = None,
            number_font_size = 10.0,
            force_shared_spines = False,
            plot_as_histogram = False,
            plot_file_prefix = "var-only-" + prefix,
            plot_dir = project_util.PLOT_DIR)

    prefix = "infer-columns-by-fixed-rows"
    generate_model_plot_grid(
            results_grid = fixed_results_grid,
            column_labels = column_labels,
            row_labels = fixed_row_labels,
            number_of_comparisons = len(height_parameters),
            plot_width = plot_width - 0.6,
            plot_height = plot_height,
            pad_left = pad_left - 0.07,
            pad_right = pad_right,
            pad_bottom = pad_bottom + 0.01,
            pad_top = pad_top - 0.065,
            y_label_size = 18.0,
            y_label = None,
            number_font_size = 10.0,
            force_shared_spines = False,
            plot_as_histogram = True,
            histogram_correct_values = fixed_correct_nevents_grid,
            plot_file_prefix = prefix,
            plot_dir = project_util.PLOT_DIR)
    generate_model_plot_grid(
            results_grid = var_only_fixed_results_grid,
            column_labels = column_labels,
            row_labels = fixed_row_labels,
            number_of_comparisons = len(height_parameters),
            plot_width = plot_width - 0.6,
            plot_height = plot_height,
            pad_left = pad_left - 0.07,
            pad_right = pad_right,
            pad_bottom = pad_bottom + 0.01,
            pad_top = pad_top - 0.065,
            y_label_size = 18.0,
            y_label = None,
            number_font_size = 10.0,
            force_shared_spines = False,
            plot_as_histogram = True,
            histogram_correct_values = fixed_correct_nevents_grid,
            plot_file_prefix = "var-only-" + prefix,
            plot_dir = project_util.PLOT_DIR)

    # Generate model plots for different sized datasets
    prefix = "nchars"
    generate_model_plot_grid(
            results_grid = nchars_results_grid,
            column_labels = column_labels,
            row_labels = nchars_row_labels,
            number_of_comparisons = len(height_parameters),
            plot_width = plot_width - 0.6,
            plot_height = plot_height,
            pad_left = pad_left - 0.07,
            pad_right = pad_right,
            pad_bottom = pad_bottom - 0.035,
            pad_top = pad_top - 0.032,
            y_label_size = 18.0,
            y_label = None,
            number_font_size = 10.0,
            force_shared_spines = False,
            plot_as_histogram = True,
            histogram_correct_values = nchars_correct_nevents_grid,
            plot_file_prefix = prefix,
            plot_dir = project_util.PLOT_DIR)
    generate_model_plot_grid(
            results_grid = var_only_nchars_results_grid,
            column_labels = column_labels,
            row_labels = nchars_row_labels,
            number_of_comparisons = len(height_parameters),
            plot_width = plot_width - 0.6,
            plot_height = plot_height,
            pad_left = pad_left - 0.07,
            pad_right = pad_right,
            pad_bottom = pad_bottom - 0.035,
            pad_top = pad_top - 0.032,
            y_label_size = 18.0,
            y_label = None,
            number_font_size = 10.0,
            force_shared_spines = False,
            plot_as_histogram = True,
            histogram_correct_values = nchars_correct_nevents_grid,
            plot_file_prefix = "var-only-" + prefix,
            plot_dir = project_util.PLOT_DIR)

    jitter = 0.12
    alpha = 0.5
    for sim_cfg in results:
        for analysis_cfg in results[sim_cfg]:
            prefix = "{0}-{1}".format(sim_cfg, analysis_cfg)
            r = results[sim_cfg][analysis_cfg]
            vo_r = var_only_results[sim_cfg][analysis_cfg]
            nshared_v_abs_error, nshared_v_ci_width = BoxData.init_time_v_sharing(
                    results = r,
                    estimator_prefix = "mean")
            generate_box_plot(
                    data = nshared_v_abs_error,
                    plot_file_prefix = "shared-v-abs-error-" + prefix,
                    title = None,
                    title_size = 16.0,
                    x_label = None,
                    x_label_size = 16.0,
                    y_label = None,
                    y_label_size = 16.0,
                    plot_width = plot_width,
                    plot_height = plot_height,
                    pad_left = pad_left + 0.02,
                    pad_right = pad_right + 0.02,
                    pad_bottom = pad_bottom,
                    pad_top = pad_top,
                    jitter = jitter,
                    alpha = alpha,
                    rasterized = False)
            generate_box_plot(
                    data = nshared_v_ci_width,
                    plot_file_prefix = "shared-v-ci-width-" + prefix,
                    title = None,
                    title_size = 16.0,
                    x_label = None,
                    x_label_size = 16.0,
                    y_label = None,
                    y_label_size = 16.0,
                    plot_width = plot_width,
                    plot_height = plot_height,
                    pad_left = pad_left + 0.02,
                    pad_right = pad_right + 0.02,
                    pad_bottom = pad_bottom,
                    pad_top = pad_top,
                    jitter = jitter,
                    alpha = alpha,
                    rasterized = False)
            vo_nshared_v_abs_error, vo_nshared_v_ci_width = BoxData.init_time_v_sharing(
                    results = vo_r,
                    estimator_prefix = "mean")
            generate_box_plot(
                    data = vo_nshared_v_abs_error,
                    plot_file_prefix = "var-only-shared-v-abs-error-" + prefix,
                    title = None,
                    title_size = 16.0,
                    x_label = None,
                    x_label_size = 16.0,
                    y_label = None,
                    y_label_size = 16.0,
                    plot_width = plot_width,
                    plot_height = plot_height,
                    pad_left = pad_left + 0.02,
                    pad_right = pad_right + 0.02,
                    pad_bottom = pad_bottom,
                    pad_top = pad_top,
                    jitter = jitter,
                    alpha = alpha,
                    rasterized = False)
            generate_box_plot(
                    data = vo_nshared_v_ci_width,
                    plot_file_prefix = "var-only-shared-v-ci-width-" + prefix,
                    title = None,
                    title_size = 16.0,
                    x_label = None,
                    x_label_size = 16.0,
                    y_label = None,
                    y_label_size = 16.0,
                    plot_width = plot_width,
                    plot_height = plot_height,
                    pad_left = pad_left + 0.02,
                    pad_right = pad_right + 0.02,
                    pad_bottom = pad_bottom,
                    pad_top = pad_top,
                    jitter = jitter,
                    alpha = alpha,
                    rasterized = False)


    histograms_to_plot = {
            "n-var-sites": {
                    "headers": tuple(
                            h for h in header if h.startswith("n_var_sites_c")
                            ),
                    "label": "Number of variable sites",
                    "short_label": "No. variable sites",
                    "ndigits": 0,
                    "center_key": "mean",
            },
            "ess-ln-likelihood": {
                    "headers": [
                            "ess_sum_ln_likelihood",
                    ],
                    "label": "Effective sample size of log likelihood",
                    "short_label": "ESS of lnL",
                    "ndigits": 1,
                    "center_key": "mean",
            },
            "ess-div-time": {
                    "headers": tuple(
                            h for h in header if h.startswith("ess_sum_root_height_c")
                            ),
                    "label": "Effective sample size of event time",
                    "short_label": "ESS of time",
                    "ndigits": 1,
                    "center_key": "mean",
            },
            "ess-root-pop-size": {
                    "headers": tuple(
                            h for h in header if h.startswith("ess_sum_pop_size_root_c")
                            ),
                    "label": "Effective sample size of ancestral population size",
                    "short_label": "ESS of ancestral size",
                    "ndigits": 1,
                    "center_key": "mean",
            },
            "psrf-ln-likelihood": {
                    "headers": [
                            "psrf_ln_likelihood",
                    ],
                    "label": "PSRF of log likelihood",
                    "short_label": "PSRF of lnL",
                    "ndigits": 2,
                    "center_key": "mean",
            },
            "psrf-div-time": {
                    "headers": tuple(
                            h for h in header if h.startswith("psrf_root_height_c")
                            ),
                    "label": "PSRF of event time",
                    "short_label": "PSRF of time",
                    "ndigits": 2,
                    "center_key": "mean",
            },
            "run-time": {
                    "headers": [
                            "mean_run_time",
                    ],
                    "label": "Run time (seconds)",
                    "short_label": "Run time (seconds)",
                    "ndigits": 1,
                    "center_key": "median",
            },
    }

    for parameter, p_info in histograms_to_plot.items():
        data = {}
        var_only_data = {}
        for sim_cfg in results:
            data[sim_cfg] = {}
            var_only_data[sim_cfg] = {}
            for analysis_cfg in results[sim_cfg]:
                data[sim_cfg][analysis_cfg] = HistogramData.init(
                        results[sim_cfg][analysis_cfg],
                        p_info["headers"],
                        False)
                var_only_data[sim_cfg][analysis_cfg] = HistogramData.init(
                        var_only_results[sim_cfg][analysis_cfg],
                        p_info["headers"],
                        False)

        data_grid = get_data_grid(data, cfg_grid)
        var_only_data_grid = get_data_grid(var_only_data, cfg_grid)

        prefix = "infer-columns-by-data-rows-" + parameter
        generate_histogram_grid(
                data_grid = data_grid,
                plot_file_prefix = prefix,
                column_labels = column_labels,
                row_labels = row_labels,
                parameter_label = p_info["label"],
                range_key = "range",
                center_key = p_info["center_key"],
                number_of_digits = p_info["ndigits"],
                plot_width = plot_width,
                plot_height = plot_height,
                pad_left = pad_left,
                pad_right = pad_right,
                pad_bottom = pad_bottom,
                pad_top = pad_top,
                force_shared_x_range = True,
                force_shared_bins = True,
                force_shared_y_range = True,
                force_shared_spines = True,
                plot_dir = project_util.PLOT_DIR)
        generate_histogram_grid(
                data_grid = var_only_data_grid,
                plot_file_prefix = "var-only-" + prefix,
                column_labels = column_labels,
                row_labels = row_labels,
                parameter_label = p_info["label"],
                range_key = "range",
                center_key = p_info["center_key"],
                number_of_digits = p_info["ndigits"],
                plot_width = plot_width,
                plot_height = plot_height,
                pad_left = pad_left,
                pad_right = pad_right,
                pad_bottom = pad_bottom,
                pad_top = pad_top,
                force_shared_x_range = True,
                force_shared_bins = True,
                force_shared_y_range = True,
                force_shared_spines = True,
                plot_dir = project_util.PLOT_DIR)

        data_grid = get_data_grid(data, nchars_cfg_grid)
        var_only_data_grid = get_data_grid(var_only_data, nchars_cfg_grid)

        prefix = "nchars-" + parameter
        generate_histogram_grid(
                data_grid = data_grid,
                plot_file_prefix = prefix,
                column_labels = column_labels,
                row_labels = nchars_row_labels,
                parameter_label = p_info["label"],
                range_key = "range",
                center_key = p_info["center_key"],
                number_of_digits = p_info["ndigits"],
                plot_width = plot_width,
                plot_height = plot_height,
                pad_left = pad_left,
                pad_right = pad_right,
                pad_bottom = pad_bottom,
                pad_top = pad_top,
                force_shared_x_range = True,
                force_shared_bins = True,
                force_shared_y_range = True,
                force_shared_spines = True,
                plot_dir = project_util.PLOT_DIR)
        generate_histogram_grid(
                data_grid = var_only_data_grid,
                plot_file_prefix = "var-only-" + prefix,
                column_labels = column_labels,
                row_labels = nchars_row_labels,
                parameter_label = p_info["label"],
                range_key = "range",
                center_key = p_info["center_key"],
                number_of_digits = p_info["ndigits"],
                plot_width = plot_width,
                plot_height = plot_height,
                pad_left = pad_left,
                pad_right = pad_right,
                pad_bottom = pad_bottom,
                pad_top = pad_top,
                force_shared_x_range = True,
                force_shared_bins = True,
                force_shared_y_range = True,
                force_shared_spines = True,
                plot_dir = project_util.PLOT_DIR)


if __name__ == "__main__":
    main_cli()
