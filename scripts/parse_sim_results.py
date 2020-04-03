#! /usr/bin/env python

import sys
import os
import re
import math
import logging
import argparse

import pycoevolity

import project_util

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
_LOG = logging.getLogger(os.path.basename(__file__))

def get_population_labels(state_log_path):
    posterior_sample = pycoevolity.posterior.PosteriorSample([state_log_path])
    return posterior_sample.tip_labels

def line_count(path):
    count = 0
    with open(path) as stream:
        for line in stream:
            count += 1
    return count

def get_parameter_names(population_labels):
    p = ["ln_likelihood"]
    p.append("concentration")
    p.append("number_of_events")
    for pop_labels in population_labels:
        assert (len(pop_labels) > 0) and (len(pop_labels) < 3)
        p.append("ln_likelihood_{0}".format(pop_labels[0]))
        p.append("root_height_{0}".format(pop_labels[0]))
        p.append("mutation_rate_{0}".format(pop_labels[0]))
        p.append("freq_1_{0}".format(pop_labels[0]))
        p.append("pop_size_root_{0}".format(pop_labels[0]))
        p.append("pop_size_{0}".format(pop_labels[0]))
        if len(pop_labels) > 1:
            p.append("pop_size_{0}".format(pop_labels[1]))
    return p

def get_results_header(population_labels):
    h = [
            "batch",
            "sim",
            "sample_size",
            "mean_run_time",
            "mean_n_var_sites",
            "true_model",
            "map_model",
            "true_model_cred_level",
            "map_model_p",
            "true_model_p",
            "true_num_events",
            "map_num_events",
            "true_num_events_cred_level",
        ]

    number_of_comparisons = len(population_labels)

    for i in range(number_of_comparisons):
        h.append("num_events_{0}_p".format(i+1))

    for i in range(number_of_comparisons):
        h.append("n_var_sites_c{0}".format(i+1))

    for p in get_parameter_names(population_labels):
        h.append("true_{0}".format(p))
        h.append("true_{0}_rank".format(p))
        h.append("mean_{0}".format(p))
        h.append("median_{0}".format(p))
        h.append("stddev_{0}".format(p))
        h.append("hpdi_95_lower_{0}".format(p))
        h.append("hpdi_95_upper_{0}".format(p))
        h.append("eti_95_lower_{0}".format(p))
        h.append("eti_95_upper_{0}".format(p))
        h.append("ess_{0}".format(p))
        h.append("ess_sum_{0}".format(p))
        h.append("psrf_{0}".format(p))
    return h

def get_empty_results_dict(population_labels):
    h = get_results_header(population_labels)
    return dict(zip(h, ([] for i in range(len(h)))))

def get_results_from_sim_rep(
        posterior_paths,
        stdout_paths,
        true_path,
        parameter_names,
        batch_number,
        sim_number,
        number_of_comparisons,
        expected_number_of_samples = 1501,
        burnin = 501):
    posterior_paths = sorted(posterior_paths)
    stdout_paths = sorted(stdout_paths)
    nchains = len(posterior_paths)
    assert(nchains == len(stdout_paths))
    if nchains > 1:
        lc = line_count(posterior_paths[0])
        for i in range(1, nchains):
            assert(lc == line_count(posterior_paths[i]))

    results = {}
    post_sample = pycoevolity.posterior.PosteriorSample(
            posterior_paths,
            burnin = burnin)
    nsamples_per_chain = expected_number_of_samples - burnin
    assert(post_sample.number_of_samples == nchains * nsamples_per_chain)

    true_values = pycoevolity.parsing.get_dict_from_spreadsheets(
            [true_path],
            sep = "\t",
            header = None)
    for v in true_values.values():
        assert(len(v) == 1)

    results["batch"] = batch_number
    results["sim"] = sim_number
    results["sample_size"] = post_sample.number_of_samples

    stdout = pycoevolity.parsing.EcoevolityStdOut(stdout_paths[0])
    assert(number_of_comparisons == stdout.number_of_comparisons)
    results["mean_n_var_sites"] = stdout.get_mean_number_of_variable_sites()
    for i in range(number_of_comparisons):
        results["n_var_sites_c{0}".format(i + 1)] = stdout.get_number_of_variable_sites(i)
    run_times = [stdout.run_time]
    for i in range(1, len(stdout_paths)):
        so = pycoevolity.parsing.EcoevolityStdOut(stdout_paths[i])
        run_times.append(so.run_time)
        for j in range(number_of_comparisons):
            assert(results["n_var_sites_c{0}".format(j + 1)] == so.get_number_of_variable_sites(j))
    results["mean_run_time"] = sum(run_times) / float(len(run_times))
    
    true_model = tuple(int(true_values[h][0]) for h in post_sample.height_index_keys)
    true_model_p = post_sample.get_model_probability(true_model)
    true_model_cred = post_sample.get_model_credibility_level(true_model)
    map_models = post_sample.get_map_models()
    map_model = map_models[0]
    if len(map_models) > 1:
        if true_model in map_models:
            map_model = true_model
    map_model_p = post_sample.get_model_probability(map_model)
    results["true_model"] = "".join((str(i) for i in true_model))
    results["map_model"] = "".join((str(i) for i in map_model))
    results["true_model_cred_level"] = true_model_cred
    results["map_model_p"] = map_model_p
    results["true_model_p"] = true_model_p
    
    true_nevents = int(true_values["number_of_events"][0])
    true_nevents_p = post_sample.get_number_of_events_probability(true_nevents)
    true_nevents_cred = post_sample.get_number_of_events_credibility_level(true_nevents)
    map_numbers_of_events = post_sample.get_map_numbers_of_events()
    map_nevents = map_numbers_of_events[0]
    if len(map_numbers_of_events) > 1:
        if true_nevents in map_numbers_of_events:
            map_nevents = true_nevents
    results["true_num_events"] = true_nevents
    results["map_num_events"] = map_nevents
    results["true_num_events_cred_level"] = true_nevents_cred
    for i in range(number_of_comparisons):
        results["num_events_{0}_p".format(i + 1)] = post_sample.get_number_of_events_probability(i + 1)
    
    for parameter in parameter_names:
        true_param = parameter
        post_param = parameter
        if parameter == "concentration":
            if parameter not in true_values:
                true_param = "split_weight"
            if parameter not in post_sample.parameter_samples:
                post_param = "split_weight"
        # The constrained models do not output a column for
        # concentration/split_weight, so we will just set the true value to
        # zero for these cases.
        try:
            true_val = float(true_values[true_param][0])
        except KeyError:
            if true_param != "split_weight":
                raise
            true_val = 0.0
        true_val_rank = post_sample.get_rank(post_param, true_val)
        ss = pycoevolity.stats.get_summary(
                post_sample.parameter_samples[post_param])
        ess = pycoevolity.stats.effective_sample_size(
                post_sample.parameter_samples[post_param])
        ess_sum = 0.0
        samples_by_chain = []
        for i in range(nchains):
            chain_samples = post_sample.parameter_samples[post_param][
                    i * nsamples_per_chain : (i + 1) * nsamples_per_chain]
            assert(len(chain_samples) == nsamples_per_chain)
            ess_sum += pycoevolity.stats.effective_sample_size(chain_samples)
            if nchains > 1:
                samples_by_chain.append(chain_samples)
        psrf = -1.0
        if nchains > 1:
            psrf = pycoevolity.stats.potential_scale_reduction_factor(samples_by_chain)
        results["true_{0}".format(parameter)] = true_val
        results["true_{0}_rank".format(parameter)] = true_val_rank
        results["mean_{0}".format(parameter)] = ss["mean"]
        results["median_{0}".format(parameter)] = ss["median"]
        results["stddev_{0}".format(parameter)] = math.sqrt(ss["variance"])
        results["hpdi_95_lower_{0}".format(parameter)] = ss["hpdi_95"][0]
        results["hpdi_95_upper_{0}".format(parameter)] = ss["hpdi_95"][1]
        results["eti_95_lower_{0}".format(parameter)] = ss["qi_95"][0]
        results["eti_95_upper_{0}".format(parameter)] = ss["qi_95"][1]
        results["ess_{0}".format(parameter)] = ess
        results["ess_sum_{0}".format(parameter)] = ess_sum
        results["psrf_{0}".format(parameter)] = psrf
    return results

def get_and_vet_state_log_paths(path_to_sim_directory):
    state_log_paths = project_util.get_sim_state_log_paths(
            path_to_sim_directory)
    sim_rep_numbers = None
    run_numbers = None
    for config_name, sims in state_log_paths.items():
        if sim_rep_numbers is None:
            sim_rep_numbers = sorted(sims.keys())
        else:
            if sim_rep_numbers != sorted(sims.keys()):
                raise Exception(
                        "Unexpected sim rep numbers found in state log paths "
                        "of '{0}' in '{1}'".format(config_name,
                                path_to_sim_directory))
        for sim_rep_num, runs in sims.items():
            if run_numbers is None:
                run_numbers = sorted(runs.keys())
            else:
                if run_numbers != sorted(runs.keys()):
                    raise Exception(
                            "Unexpected run numbers found in state log paths "
                            "for sim rep {0} of '{1}' in '{2}'".format(
                                sim_rep_num,
                                config_name,
                                path_to_sim_directory))
    return state_log_paths, sim_rep_numbers, run_numbers

def parse_simulation_results(
        path_to_sim_directory,
        expected_number_of_samples = 1501,
        burnin = 501):
    batch_number_match = project_util.BATCH_DIR_ENDING_PATTERN.match(
            path_to_sim_directory)
    if not batch_number_match:
        raise Exception(
                "Could not parse batch number from path to sim directory: "
                "'{0}'".format(path_to_sim_directory))
    batch_number = int(batch_number_match.group("batch_num"))

    state_log_paths, sim_rep_numbers, run_numbers = get_and_vet_state_log_paths(
            path_to_sim_directory)
    population_labels = None
    header = None
    for config_name in state_log_paths:
        _LOG.info("Parsing results for '{0}'".format(config_name))
        if population_labels is None:
            population_labels = get_population_labels(
                    state_log_paths[config_name][sim_rep_numbers[0]][
                        run_numbers[0]])
            header = get_results_header(population_labels)
        number_of_comparisons = len(population_labels)
        parameter_names = get_parameter_names(population_labels)
        results = get_empty_results_dict(population_labels)
        results_path = os.path.join(path_to_sim_directory,
                config_name + "-results.tsv")
        if (os.path.exists(results_path) or
                (os.path.exists(results_path + ".gz"))):
            _LOG.warning("Results path '{0}' already exists; skipping!".format(
                    results_path))
            continue

        for sim_rep_num in sim_rep_numbers:
            log_paths = [state_log_paths[config_name][sim_rep_num][x] for x in run_numbers]
            true_path = os.path.join(path_to_sim_directory,
                        "simcoevolity-sim-{0}-true-values.txt".format(
                                sim_rep_num))
            if not os.path.isfile(true_path):
                raise Exception(
                        "Expected file missing: {0}".format(true_path))
            stdout_paths = []
            for run_num in run_numbers:
                stdout_p = os.path.join(path_to_sim_directory,
                        "run-{run_num}-{config_name}-sim-{sim_num}-config.yml.out".format(
                                run_num = run_num,
                                config_name = config_name,
                                sim_num = sim_rep_num))
                if not os.path.isfile(stdout_p):
                    raise Exception(
                            "Expected file missing: {0}".format(stdout_p))
                stdout_paths.append(stdout_p)

            rep_results = get_results_from_sim_rep(
                    posterior_paths = log_paths,
                    stdout_paths = stdout_paths,
                    true_path = true_path,
                    parameter_names = parameter_names,
                    batch_number = batch_number,
                    sim_number = sim_rep_num,
                    number_of_comparisons = number_of_comparisons,
                    expected_number_of_samples = expected_number_of_samples,
                    burnin = burnin)
            for k, v in rep_results.items():
                results[k].append(v)

        assert not os.path.exists(results_path)
        with open(results_path, 'w') as out:
            for line in pycoevolity.parsing.dict_line_iter(
                    results,
                    sep = '\t',
                    header = header):
                out.write(line)


def main_cli():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('sim_dirs',
            metavar = 'SIMCOEVOLITY-OUTPUT-DIR',
            nargs = '+',
            type = project_util.arg_is_dir,
            help = ('Path to directory with simcoevolity output files.'))
    parser.add_argument('-s', '--expected-number-of-samples',
            action = 'store',
            type = int,
            default = 1501,
            help = ('Number of MCMC samples that should be found in each log '
                    'file of each analysis.'))
    parser.add_argument('--burnin',
            action = 'store',
            type = int,
            default = 501,
            help = ('Number of MCMC samples to be ignored as burnin from the '
                    'beginning of every chain.'))

    args = parser.parse_args()

    for sim_dir in args.sim_dirs:
        batch_dir_match = project_util.BATCH_DIR_ENDING_PATTERN.match(sim_dir)
        if not batch_dir_match:
            raise Exception("The following path is not a batch directory\n"
                    "  {0}".format(sim_dir))

    for sim_dir in args.sim_dirs:
        _LOG.info("Parsing sim results in '{0}'".format(sim_dir))
        parse_simulation_results(
                sim_dir,
                expected_number_of_samples = args.expected_number_of_samples,
                burnin = args.burnin)


if __name__ == "__main__":
    main_cli()
