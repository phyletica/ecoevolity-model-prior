#! /usr/bin/env python

import os
import sys
import random
import re
import argparse

import project_util

def get_new_script_path(template_path, batch_id_string):
    out_dir = os.path.dirname(template_path)
    template_file_name = os.path.basename(template_path)
    if not template_file_name.startswith("template-"):
        raise Exception(
                "Sim script template path should start with 'template-'")
    template_file_prefix, extension = os.path.splitext(template_file_name)
    out_file_prefix = template_file_prefix[len("template-"):]
    out_file_name = "{prefix}-batch-{batch_id}.sh".format(
            prefix = out_file_prefix,
            batch_id = batch_id_string)
    out_path = os.path.join(out_dir, out_file_name)
    return out_path

def convert_sim_script(in_stream, out_stream, batch_id_string, number_of_reps):
    found_seed_line = False
    found_reps_line = False
    for line in in_stream:
        if line.startswith("rng_seed="):
            out_stream.write("rng_seed={}\n".format(batch_id_string))
            found_seed_line = True
        elif line.startswith("number_of_reps="):
            out_stream.write("number_of_reps={}\n".format(number_of_reps))
            found_reps_line = True
        else:
            out_stream.write(line)
    if not found_seed_line:
        raise Exception("Line setting 'rng_seed' not found")
    if not found_reps_line:
        raise Exception("Line setting 'number_of_reps' not found")

def main_cli():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('sim_script_template_paths',
            metavar = 'SIM-SCRIPT-TEMPLATE-PATH',
            nargs = '+',
            type = project_util.arg_is_file,
            help = ('Path to template shell script for running simcoevolity.'))
    parser.add_argument('-n', '--number-of-reps',
            action = 'store',
            default = 10,
            type = project_util.arg_is_positive_int,
            help = ('Number of simulation replicates for the new batch.'))
    parser.add_argument('--seed',
            action = 'store',
            type = project_util.arg_is_positive_int,
            help = ('Seed for random number generator.'))

    args = parser.parse_args()

    max_random_int = 999999999
    max_num_digits = len(str(max_random_int))

    rng = random.Random()
    if not args.seed:
        args.seed = random.randint(1, max_random_int)
    rng.seed(args.seed)

    batch_num = rng.randint(1, max_random_int)
    batch_num_str = str(batch_num).zfill(max_num_digits)

    for sim_path in args.sim_script_template_paths:
        out_path = get_new_script_path(sim_path, batch_num_str)
        if os.path.exists(out_path):
            raise Exception("Script path '{0}' already exists!".format(out_path))
        with open(sim_path, "r") as in_stream:
            with open(out_path, "w") as out_stream:
                try:
                    convert_sim_script(
                            in_stream = in_stream,
                            out_stream = out_stream,
                            batch_id_string = batch_num_str,
                            number_of_reps = args.number_of_reps)
                except:
                    sys.stderr.write(
                            "ERROR: problem converting '{0}'".format(sim_path))
                    raise
        sys.stdout.write("Script written to '{0}'\n".format(out_path))
    sys.stdout.write("\nSimcoevolity scripts successfully written.\n")
    sys.stdout.write("\nBatch ID:\n")
    sys.stdout.write("\t{0}\n".format(batch_num_str))

if __name__ == "__main__":
    main_cli()
