#!/usr/bin/env python

import os
import sys
import argparse
import numpy
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import pycoevolity

import project_util


def almost_equal(x, y,
        proportional_tolerance = 1e-6):
    abs_tol = max(math.fabs(x), math.fabs(y)) * proportional_tolerance
    diff = math.fabs(x - y)
    if (diff > abs_tol):
        return False
    return True

def get_yaml_config(path):
    with open(path, 'r') as stream:
        config = load(stream, Loader = Loader)
    return config

def plot_samples_with_distribution(samples,
        scipy_stats_dist,
        plot_path):
    fig = plt.figure(figsize = (5.0, 4.0))
    gs = gridspec.GridSpec(1, 1,
            wspace = 0.0,
            hspace = 0.0)
    ax = plt.subplot(gs[0, 0])
    n, bins, patches = ax.hist(samples, density = True)

    x = numpy.linspace(scipy_stats_dist.ppf(0.001), scipy_stats_dist.ppf(0.999), 100)
    ax.plot(x, scipy_stats_dist.pdf(x))
    fig.tight_layout()
    plt.savefig(plot_path)

def get_distribution(prior_settings):
    assert len(prior_settings) == 1
    prior_name = list(prior_settings.keys())[0]
    prior_parameters = prior_settings[prior_name]
    if prior_name == "gamma_distribution":
        shape = float(prior_parameters["shape"])
        scale = float(prior_parameters["scale"])
        dist = scipy.stats.gamma(shape, scale = scale)
        return dist
    if prior_name == "exponential_distribution":
        scale = 1.0 / float(prior_parameters["rate"])
        dist = scipy.stats.gamma(1.0, scale = scale)
        return dist
    if prior_name == "beta_distribution":
        a = float(prior_parameters["alpha"])
        b = float(prior_parameters["beta"])
        dist = scipy.stats.beta(a = a, b = b)
        return dist
    raise Exception("Unexpected prior distribution: {0}".format(prior_name))

def process_parameter(
        parameter_name,
        parameter_settings,
        posterior_samples,
        output_dir):
    values = []
    is_estimated = parameter_settings.get("estimate", False)
    if is_estimated:
        prior_settings = parameter_settings["prior"]
        prior_distribution = get_distribution(prior_settings)
        plot_path = os.path.join(output_dir,
                "prior-" + parameter_name + ".pdf")
        plot_samples_with_distribution(posterior_samples,
                prior_distribution, plot_path)
    else:
        fixed_value = parameter_settings["value"]
        for v in values:
            assert almost_equal(fixed_value, v)


def main_cli():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('config_path',
            metavar = 'ECOEVOLITY-CONFIG-PATH',
            type = project_util.arg_is_file,
            help = ('Path to ecoevolity configuration file that was used to '
                    'to sample from the prior.'))
    parser.add_argument('log_paths',
            metavar = 'ECOEVOLITY-STATE-LOG-PATH',
            nargs = '+',
            type = project_util.arg_is_file,
            help = ('Paths to ecoevolity state log files.'))
    parser.add_argument('-b', '--burnin',
            action = 'store',
            type = project_util.arg_is_nonnegative_int,
            default = 0,
            help = ('The number of samples to remove from the beginning of '
                    'each log file as burn in.'))
    args = parser.parse_args()

    output_dir = os.path.dirname(args.log_paths[0])
    posterior_sample = pycoevolity.posterior.PosteriorSample(args.log_paths)
    config = get_yaml_config(args.config_path)

    model_prior_settings = config.get("event_model_prior")
    assert len(model_prior_settings) == 1
    model_prior_name = list(model_prior_settings.keys())[0]
    if not model_prior_name == "fixed":
        model_prior_parameters = model_prior_settings[model_prior_name][
                "parameters"]
        for parameter_name, parameter_settings in model_prior_parameters.items():
            values = posterior_sample.parameter_samples[parameter_name]
            process_parameter(
                    parameter_name = parameter_name,
                    parameter_settings = parameter_settings,
                    posterior_samples = values,
                    output_dir = output_dir)

    event_time_prior = config["event_time_prior"]
    event_time_settings = {
            "estimate" : True,
            "prior" : event_time_prior,
            }
    values = []
    for k in posterior_sample.get_height_keys():
        values.extend(posterior_sample.parameter_samples[k])
    process_parameter(
            parameter_name = "event_time",
            parameter_settings = event_time_settings,
            posterior_samples = values,
            output_dir = output_dir)
    

    labels, root_relative_sizes = posterior_sample.get_relative_population_sizes_2d()
    root_relative_population_size_settings = config[
            "global_comparison_settings"][
                "parameters"][
                    "root_relative_population_size"]
    process_parameter(
            parameter_name = "root_relative_population_size",
            parameter_settings = root_relative_population_size_settings,
            posterior_samples = root_relative_sizes,
            output_dir = output_dir)

    leaf_population_size_settings = config[
            "global_comparison_settings"][
                "parameters"][
                    "population_size"]
    values = []
    for k in posterior_sample.get_descendant_pop_size_keys():
        values.extend(posterior_sample.parameter_samples[k])
    process_parameter(
            parameter_name = "leaf_population_size",
            parameter_settings = leaf_population_size_settings,
            posterior_samples = values,
            output_dir = output_dir)


if __name__ == "__main__":
    main_cli()
