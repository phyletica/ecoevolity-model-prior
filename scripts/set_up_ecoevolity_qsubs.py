#! /usr/bin/env python

import sys
import os
import argparse
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import project_util

def write_qsub(config_path,
        run_number = 1,
        relax_missing_sites = False):
    qsub_prefix = os.path.splitext(config_path)[0]
    qsub_path = "{0}-run-{1}-qsub.sh".format(qsub_prefix, run_number)
    if os.path.exists(qsub_path):
        return
    config_file = os.path.basename(config_path)
    stdout_path = "run-{0}-{1}.out".format(run_number, config_file)
    seed = run_number
    assert(not os.path.exists(qsub_path))
    with open(qsub_path, 'w') as out:
        out.write(project_util.PBS_HEADER)
        out.write("exe_path=\"${ECOEVOLITY_MODEL_PRIOR_BIN_DIR}/ecoevolity\"\n\n")
        if relax_missing_sites:
            out.write("$exe_path --seed {0} --prefix run-{1}- --relax-constant-sites --relax-missing-sites {2} 1>{3} 2>&1\n".format(
                    seed,
                    run_number,
                    config_file,
                    stdout_path))
        else:
            out.write("$exe_path --seed {0} --prefix run-{1}- --relax-constant-sites {2} 1>{3} 2>&1\n".format(
                    seed,
                    run_number,
                    config_file,
                    stdout_path))


def main_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument('sim_dir',
            metavar = 'SIMCOEVOLITY-OUTPUT-DIR',
            type = project_util.arg_is_dir,
            help = ('Path to directory with simcoevolity output files.'))
    parser.add_argument('--number-of-runs',
            action = 'store',
            type = int,
            default = 4,
            help = 'Number of qsubs to generate per config (Default: 4).')

    args = parser.parse_args()

    for config_path in project_util.sim_configs_to_use_iter(args.sim_dir):
        for i in range(args.number_of_runs):
            write_qsub(config_path = config_path,
                    run_number = i + 1,
                    relax_missing_sites = False)
    

if __name__ == "__main__":
    main_cli()
