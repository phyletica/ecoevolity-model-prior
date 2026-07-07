#! /usr/bin/env python

import sys
import os
import argparse

import matplotlib as mpl

# Use TrueType (42) fonts rather than Type 3 fonts
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
# tex_font_settings = {
#         "text.usetex": True,
#         "font.family": "sans-serif",
#         "text.latex.preamble" : [
#                 "\\usepackage[T1]{fontenc}",
#                 "\\usepackage{amssymb}",
#                 "\\usepackage[cm]{sfmath}",
#                 ]
# }
# mpl.rcParams.update(tex_font_settings)

import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy.stats
import numpy

import pycoevolity


class ModelKSHelper(object):
    def __init__(self, mcmc_data): # sampled_models, models_to_expected_probs):
        model_keys = sort_models(mcmc_data.keys())
        model_to_index = {model_keys[i] : i for i in range(len(model_keys))}
        # index_to_model = {model_to_index[m] : m for m in model_keys}
        self.index_to_prob = {model_to_index[m] : mcmc_data[m]["expected_prob"] for m in model_keys}
        samples = []
        for m in model_keys:
            idx = model_to_index[m]
            count = mcmc_data[m]["num_samples"]
            samples.extend([idx] * count)
        self.samples = tuple(samples)

    def get_samples(self):
        return self.samples

    def get_expected_samples(self, n = None):
        if not n:
            n = len(self.samples)
        expected_samples = []
        for idx, prob in self.index_to_prob.items():
            expected_count = int(round(prob * n))
            expected_samples.extend([idx] * expected_count)
        return tuple(expected_samples)


def parse_results_tsv(path):
    data = pycoevolity.parsing.get_dict_from_spreadsheets(
            [path],
            sep = "\t",
            offset = 0)
    data["num_samples"] = tuple([int(x) for x in data["num_samples"]])
    data["expected_prob"] = tuple([float(x) for x in data["expected_prob"]])
    total_samples = sum(data["num_samples"])
    d = {}
    for i in range(len(data["model"])):
        obs_freq = data["num_samples"][i] / total_samples
        assert data["model"][i] not in d
        d[data["model"][i]] = {
            "num_samples" : data["num_samples"][i],
            "expected_prob" : data["expected_prob"][i],
            "sample_freq" : obs_freq,
        }
    return d

def sort_models(models):
    k_to_models = {}
    for model in models:
        indices = [int(x) for x in model.split(",")]
        max_index = max(indices)
        if max_index in k_to_models:
            k_to_models[max_index].append(model)
        else:
            k_to_models[max_index] = [model]
    sorted_models = []
    for k in sorted(k_to_models.keys()):
        sorted_models.extend(sorted(k_to_models[k]))
    return tuple(sorted_models)

def plot_rj_test_results(tsv_path):
    path_prefix = os.path.splitext(tsv_path)[0]
    plot_path = f"{path_prefix}.pdf"
    data = parse_results_tsv(tsv_path)
    models = sort_models(data.keys())
    expected_probs = [data[m]["expected_prob"] for m in models]
    sample_freqs = [data[m]["sample_freq"] for m in models]
    expected_cdf = [0.0]
    sample_cdf = [0.0]
    expected_sum = 0.0
    sample_sum = 0.0
    max_abs_diff = -1.0
    for i in range(len(expected_probs)):
        expected_sum += expected_probs[i]
        sample_sum += sample_freqs[i]
        expected_cdf.append(expected_sum)
        sample_cdf.append(sample_sum)
        if abs(expected_sum - sample_sum) > max_abs_diff:
            max_abs_diff = abs(expected_sum - sample_sum)

    assert abs(expected_cdf[-1] - 1.0) < 1e-8
    assert abs(sample_cdf[-1] - 1.0) < 1e-8
    models_to_probs = {m : data[m]["expected_prob"] for m in models}

    plt.close('all')
    # fig = plt.figure(figsize = (plot_width, plot_height))
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 1,
            wspace = 0.0,
            hspace = 0.0)
    ax = plt.subplot(gs[0, 0])
    ax.plot(
        [''] + list(models),
        expected_cdf,
        marker = 'o',
        drawstyle = 'steps-mid',
        color = 'C0',
        label = "Expectation",
    )
    ax.plot(
        [''] + list(models),
        sample_cdf,
        marker = 'o',
        drawstyle = 'steps-mid',
        color = 'C1',
        label = "MCMC sample",
    )
    plt.setp(
        ax.get_xticklabels(),
        rotation = 90,
        # ha = 'right',
    )
    ax.set_xlabel("Divergence model")
    ax.set_ylabel("Cumulative probability")

    ks_helper = ModelKSHelper(data)
    res = scipy.stats.kstest(
        ks_helper.get_samples(),
        ks_helper.get_expected_samples(10000000),
        alternative = "two-sided",
    )
    n = len(ks_helper.samples)
    pval = scipy.stats.kstwo.sf(max_abs_diff, n)
    # Make sure my exact 1-sample KS test statistic and pvalue match the
    # 2-sample KS test with a huge number of expected samples. This is a sanity
    # check to make sure I am using scipy.stats.kstwo.sf correctly to get the
    # p-value.
    assert abs(max_abs_diff - res.statistic) < 1e-5
    assert abs(pval - res.pvalue) < 0.005
    ks_str = f"KS $D$ = {max_abs_diff:.2g}\n$p$ = {pval:.2g}"
    ax.text(
        0.02, 0.98,
        ks_str,
        horizontalalignment = "left",
        verticalalignment = "top",
        transform = ax.transAxes,
        zorder = 500,
        fontsize = 'small',
        # bbox = {
        #     'facecolor': 'white',
        #     'edgecolor': 'white',
        #     'pad': 2},
    )
    # ax.legend(bbox_to_anchor = (1.01, 0.5), loc = "center left")
    ax.legend(loc = "lower right")
    # gs.update(
    #     left = 0.02,
    #     right = 1.0,
    #     bottom = 0.05,
    #     top = 1.0,
    # )
    plt.savefig(plot_path, bbox_inches = 'tight', pad_inches = 0.02)


def parse_cli_args():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'tsv_paths',
        metavar = 'RJ-TEST-TSV-FILE',
        type = pycoevolity.argparse_utils.arg_is_file,
        nargs = "*",
        help = (
            'Paths to TSV files of test results output by ecoevolity of the '
            'reversible-jump operator.'
        ),
    )

    args = parser.parse_args()
    return args

def main_cli():
    args = parse_cli_args()

    for tsv_path in args.tsv_paths:
        plot_rj_test_results(tsv_path)

if __name__ == "__main__":
    plt.style.use('tableau-colorblind10')
    main_cli()
