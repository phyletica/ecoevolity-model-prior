#!/bin/bash

set -e

usage () {
    echo ""
    echo "Usage:"
    echo "  $0 <PATH-TO-SIM-DIR>"
    echo ""
    echo "Required positional argument:"
    echo "  PATH-TO-SIM-DIR  Path to diretory containing ecoeovlity output of "
    echo "                   analyses of data sets simulated with simcoevolity."
    echo ""
}

if [ -n "$PBS_JOBNAME" ]
then
    cd $PBS_O_WORKDIR
fi

# Make sure there is exactly one positional argument
if [ ! "$#" = 1 ]
then
    echo "ERROR: One argument must be provided; a path to a sim directory"
    usage
    exit 1
fi

# Path to the batch directory should be the only argument
batch_dir="$1"
# Remove any trailing slashes from the path
batch_dir="$(echo "$batch_dir" | sed 's:/*$::')"

# Make sure the argument is a valid directory
if [ ! -d "$batch_dir" ]
then
    echo "ERROR: Path is not a valid directory: $batch_dir"
    usage
    exit 1
fi

# Make sure the directory is a batch directory
if [ ! "$(echo "$batch_dir" | grep -c -E "batch-[0-9]{1,12}$")" -gt 0 ]
then
    echo "ERROR: The path provided doesn't seem to be a sim batch directory:"
    echo "    $batch_dir"
    usage
    exit 1
fi

(
    cd "$batch_dir"
    pwd
    for p in *.sh.o*; do if [ -e "$p" ]; then
        echo "Removing PBS output files..."
        rm *.sh.o*
        break
    fi; done
    for p in *qsub.sh; do if [ -e "$p" ]; then
        echo "Archiving and removing qsub scripts..."
        tar czf sim-files-qsub-scripts.tar.gz *qsub.sh && rm *qsub.sh
        break
    fi; done
    for p in *.yml; do if [ -e "$p" ]; then
        echo "Archiving and removing config files..."
        tar czf sim-files-configs.tar.gz *.yml && rm *.yml
        break
    fi; done
    for p in *chars.txt; do if [ -e "$p" ]; then
        echo "Archiving and removing data files..."
        tar czf sim-files-data.tar.gz *chars.txt && rm *chars.txt
        break
    fi; done
    for p in run-1-*; do if [ -e "$p" ]; then
        echo "Archiving and removing run 1 output files..."
        tar czf sim-files-run-1.tar.gz run-1-* && rm run-1-*
        break
    fi; done
    for p in run-2-*; do if [ -e "$p" ]; then
        echo "Archiving and removing run 2 output files..."
        tar czf sim-files-run-2.tar.gz run-2-* && rm run-2-*
        break
    fi; done
    for p in run-3-*; do if [ -e "$p" ]; then
        echo "Archiving and removing run 3 output files..."
        tar czf sim-files-run-3.tar.gz run-3-* && rm run-3-*
        break
    fi; done
    for p in run-4-*; do if [ -e "$p" ]; then
        echo "Archiving and removing run 4 output files..."
        tar czf sim-files-run-4.tar.gz run-4-* && rm run-4-*
        break
    fi; done
    for p in simcoevolity-sim-*-true-values.txt; do if [ -e "$p" ]; then
        echo "Archiving and removing files with true values..."
        tar czf sim-files-true-values.tar.gz simcoevolity-sim-*-true-values.txt && rm simcoevolity-sim-*-true-values.txt
        break
    fi; done
)
