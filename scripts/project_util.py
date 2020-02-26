#! /usr/bin/env python

import os
import sys
import re
import argparse

# Project paths
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SIM_DIR = os.path.join(PROJECT_DIR, 'ecoevolity-simulations')
CONFIG_DIR = os.path.join(PROJECT_DIR, 'ecoevolity-configs')
BIN_DIR = os.path.join(PROJECT_DIR, 'bin')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
PLOT_DIR = os.path.join(RESULTS_DIR, 'plots')

# Project regular expressions
SIMCOEVOLITY_CONFIG_NAME_PATTERN_STR = (
        r"simcoevolity-sim-(?P<sim_num>\d+)-config.yml")
SIMCOEVOLITY_CONFIG_NAME_PATTERN = re.compile(
        r"^" + SIMCOEVOLITY_CONFIG_NAME_PATTERN_STR + r"$")

SIM_CONFIG_TO_USE_PATTERN_STR = (
        r"(?P<var_only>var-only-)?(?P<config_name>pairs-\S+)-sim-(?P<sim_num>\d+)-config.yml")
SIM_CONFIG_TO_USE_PATTERN = re.compile(
        r"^" + SIM_CONFIG_TO_USE_PATTERN_STR + r"$")


PBS_HEADER =  """#! /bin/bash

if [ -z "$ECOEVOLITY_MODEL_PRIOR_PROJECT_DIR" ]
then
    echo "ERROR: ECOEVOLITY_MODEL_PRIOR_PROJECT_DIR is not set"
    exit 1
fi

if [ -z "$ECOEVOLITY_MODEL_PRIOR_BIN_DIR" ]
then
    echo "ERROR: ECOEVOLITY_MODEL_PRIOR_BIN_DIR is not set"
    exit 1
fi

if [ -n "$PBS_JOBNAME" ]
then
    cd $PBS_O_WORKDIR
    source "${ECOEVOLITY_MODEL_PRIOR_PROJECT_DIR}/modules-to-load.sh" 
fi

"""


def file_path_iter(directory, regex_pattern):
    for dir_path, dir_names, file_names in os.walk(directory):
        for f_name in file_names:
            if regex_pattern.match(f_name):
                path = os.path.join(dir_path, f_name)
                yield path

def simcoevolity_config_iter(sim_directory = None):
    if sim_directory is None:
        sim_directory = SIM_DIR
    for path in file_path_iter(sim_directory, SIMCOEVOLITY_CONFIG_NAME_PATTERN):
        yield path

def sim_configs_to_use_iter(sim_directory = None):
    if sim_directory is None:
        sim_directory = SIM_DIR
    for path in file_path_iter(sim_directory, SIM_CONFIG_TO_USE_PATTERN):
        yield path

# Utility functions for argparse
def arg_is_path(path):
    try:
        if not os.path.exists(path):
            raise
    except:
        msg = 'path {0!r} does not exist'.format(path)
        raise argparse.ArgumentTypeError(msg)
    return path

def arg_is_file(path):
    try:
        if not os.path.isfile(path):
            raise
    except:
        msg = '{0!r} is not a file'.format(path)
        raise argparse.ArgumentTypeError(msg)
    return path

def arg_is_dir(path):
    try:
        if not os.path.isdir(path):
            raise
    except:
        msg = '{0!r} is not a directory'.format(path)
        raise argparse.ArgumentTypeError(msg)
    return path

def arg_is_nonnegative_int(i):
    try:
        if int(i) < 0:
            raise
    except:
        msg = '{0!r} is not a non-negative integer'.format(i)
        raise argparse.ArgumentTypeError(msg)
    return int(i)

def arg_is_positive_int(i):
    try:
        if int(i) < 1:
            raise
    except:
        msg = '{0!r} is not a positive integer'.format(i)
        raise argparse.ArgumentTypeError(msg)
    return int(i)

def arg_is_positive_float(i):
    try:
        if float(i) <= 0.0:
            raise
    except:
        msg = '{0!r} is not a positive real number'.format(i)
        raise argparse.ArgumentTypeError(msg)
    return float(i)

def arg_is_nonnegative_float(i):
    try:
        if float(i) < 0.0:
            raise
    except:
        msg = '{0!r} is not a non-negative real number'.format(i)
        raise argparse.ArgumentTypeError(msg)
    return float(i)


def main():
    sys.stdout.write("{0}".format(PROJECT_DIR))


if __name__ == '__main__':
    main()
