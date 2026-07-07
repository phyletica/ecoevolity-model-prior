#! /usr/bin/env python

import sys
import os
import re

import pycoevolity

import project_util


def main():
    tsv_file_pattern_str = r'^.*\.tsv\.gz$'
    tsv_file_pattern = re.compile(tsv_file_pattern_str)
    for path in project_util.file_path_iter(
            directory = project_util.SIM_DIR,
            regex_pattern = tsv_file_pattern):
        header = pycoevolity.parsing.parse_header_from_path(path)
        if not "mean_model_distance" in header:
            sys.stdout.write("{0}\n".format(path))

if __name__ == "__main__":
    main()
