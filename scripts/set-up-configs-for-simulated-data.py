#! /usr/bin/env python

import sys
import os
import argparse

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import project_util


def get_yaml_config(path):
    with open(path, 'r') as stream:
        config = load(stream, Loader = Loader)
    return config

def convert_for_variable_sites_only(config):
    if "global_comparison_settings" in config:
        config["global_comparison_settings"]["constant_sites_removed"] = True
    for comparison in config["comparisons"]:
        comparison_settings = comparison["comparison"]
        if "constant_sites_removed" in comparison_settings:
            comparison_settings["constant_sites_removed"] = True
    return config

def copy_data_paths(cfg_to_modify, cfg_to_copy):
    assert len(cfg_to_modify["comparisons"]) == len(cfg_to_copy["comparisons"])
    for i in range(len(cfg_to_copy["comparisons"])):
        cfg_to_modify["comparisons"][i]["comparison"]["path"] = cfg_to_copy[
                      "comparisons"][i]["comparison"]["path"]


def main_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument('sim_dir',
            metavar = 'SIMCOEVOLITY-OUTPUT-DIR',
            nargs = 1,
            type = project_util.arg_is_dir,
            help = ('Path to directory with simcoevolity output files.'))

    args = parser.parse_args()

    configs_to_use_names = [
            "pairs-8-dpp-conc-4_0-0_97-time-1_0-0_05",
            "pairs-8-pyp-conc-4_0-0_643-disc-1_0-4_0-time-1_0-0_05",
            "pairs-8-unif-sw-1_0-2_37-time-1_0-0_05",
            ]
    configs_to_use = {}
    for config_name in configs_to_use_names:
        cfg_path = os.path.join(project_util.CONFIG_DIR, config_name + ".yml")
        cfg = get_yaml_config(cfg_path)
        configs_to_use[config_name] = cfg
        var_only_cfg = convert_for_variable_sites_only(get_yaml_config(cfg_path))
        configs_to_use["var-only-" + config_name] = var_only_cfg
        
    simco_config_path_iter = project_util.simcoevolity_config_iter(
            args.sim_dir)
    for simco_config_path in simco_config_path_iter:
        dir_path = os.path.dirname(simco_config_path)
        simco_config_file = os.path.basename(simco_config_path)
        simco_config = get_yaml_config(simco_config_path)
        for config_name, config in configs_to_use.items():
            new_config_file = simco_config_file.replace(
                    "simcoevolity-",
                    config_name + "-")
            new_config_path = os.path.join(dir_path, new_config_file)
            if os.path.exists(new_config_path):
                sys.stderr.write(
                        "Skipping config that already exists: {0}\n".format(
                                new_config_path))
                continue
            copy_data_paths(config, simco_config)
            assert not os.path.exists(new_config_path)
            with open(new_config_path, 'w') as stream:
                dump(config, stream, Dumper = Dumper, indent = 4)


if __name__ == "__main__":
    main_cli()
