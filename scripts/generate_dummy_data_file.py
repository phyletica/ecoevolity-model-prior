#! /usr/bin/env python

"""
CLI program for generating dummy ecoevolity data files. 
"""

import sys
import os
import argparse


def arg_is_positive_int(i):
    """
    Returns int if argument is a positive integer; returns argparse error
    otherwise.

    >>> arg_is_positive_int(1) == 1
    True
    """

    try:
        if int(i) < 1:
            raise
    except:
        msg = '{0!r} is not a positive integer'.format(i)
        raise argparse.ArgumentTypeError(msg)
    return int(i)

def write_dummy_biallelic_data_file(
        nspecies = 2,
        ngenomes = 10,
        ncharacters = 1000,
        prefix = "sp",
        out = sys.stdout):
    nspecies_padding = len(str(nspecies))
    out.write("---\n");
    out.write("markers_are_dominant: false\n")
    out.write("population_labels: [")
    for sp_idx in range(nspecies):
        if sp_idx > 0:
            out.write(", ")
        sp_label = prefix + "{n:0{padding}d}".format(
                n = sp_idx + 1,
                padding = nspecies_padding)
        out.write("{0}".format(sp_label))
    out.write("]\n")
    out.write("allele_count_patterns:\n")
    out.write("    - [")
    for sp_idx in range(nspecies):
        if sp_idx > 0:
            out.write(", ")
        out.write("[1, {0}]".format(ngenomes))
    out.write("]\n")
    out.write("pattern_weights: [{0}]\n".format(ncharacters))

def main_cli(argv = sys.argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--nspecies',
            action = 'store',
            type = arg_is_positive_int,
            default = 2,
            help = ('The number of populations.'))
    parser.add_argument('-g', '--ngenomes',
            action = 'store',
            type = arg_is_positive_int,
            default = 10,
            help = ('The number of genomes sampled per population.'))
    parser.add_argument('-c', '--ncharacters',
            action = 'store',
            type = arg_is_positive_int,
            default = 1000,
            help = ('The number of biallelic characters.'))
    parser.add_argument('-p', '--prefix',
            action = 'store',
            type = str,
            default = "sp", 
            help = ('Prefix for species labels.'))

    if argv == sys.argv:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)

    write_dummy_biallelic_data_file(
            nspecies = args.nspecies,
            ngenomes = args.ngenomes,
            ncharacters = args.ncharacters,
            prefix = args.prefix,
            out = sys.stdout)
    

if __name__ == "__main__":
    main_cli()
