#!/bin/bash

set -e

script_dir="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

usage () {
    echo ""
    echo "Usage:"
    echo "  $0 <PATH-TO-SIM-DIR> [ <PATH-TO-SIM-DIR> ... ]"
    echo ""
    echo "Required positional argument:"
    echo "  PATH-TO-SIM-DIR  Path to diretory containing ecoevolity output of "
    echo "                   analyses of data sets simulated with simcoevolity."
    echo ""
}

if [ -n "$PBS_JOBNAME" ]
then
    if [ -f "${PBS_O_HOME}/.bashrc" ]
    then
        source "${PBS_O_HOME}/.bashrc"
    fi
    cd $PBS_O_WORKDIR
fi

if [ -n "$(command -v conda)" ]
then
    eval "$(conda shell.bash hook)"
    conda activate ecoevolity-model-prior-project
fi

# Make sure there is at least one positional argument
if [ ! "$#" -gt 0 ]
then
    echo "ERROR: At least one argument must be provided; a path to a sim directory"
    usage
    exit 1
fi

# Vet arguments before doing anything
for batch_dir in $@
do
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
done

for batch_dir in $@
do
    # Remove any trailing slashes from the path
    batch_dir="$(echo "$batch_dir" | sed 's:/*$::')"

    echo ""
    echo "Extracting sim files from archives in:"
    echo "  ${batch_dir}"

    # Make sure archives are present
    if [ ! -f "${batch_dir}/sim-files-state-logs.tar.gz" ]
    then
        echo "  ERROR: Could not find archive of state logs"
        exit 1
    fi
    if [ ! -f "${batch_dir}/sim-files-stdout.tar.gz" ]
    then
        echo "  ERROR: Could not find archive of stdout files"
        exit 1
    fi
    if [ ! -f "${batch_dir}/sim-files-true-values.tar.gz" ]
    then
        echo "  ERROR: Could not find archive of true-values files"
        exit 1
    fi

    # git lfs needs paths relative to root of git repo (even if we are in a
    # subdirectory)
    rel_batch_dir="$(realpath "$batch_dir" --relative-to="..")"

    # If archives are ASCII text files, then they are pointers, and we need to
    # download the actual archives with git lfs
    asci_pattern=':[ ]+ASCII text[ ]*$'
    if [[ "$(file "${batch_dir}/sim-files-state-logs.tar.gz")" =~ $asci_pattern ]]
    then
        echo "  Downloading state log archive with git lfs..."
        git lfs pull --include "${rel_batch_dir}/sim-files-state-logs.tar.gz"
    fi
    if [[ "$(file "${batch_dir}/sim-files-stdout.tar.gz")" =~ $asci_pattern ]]
    then
        echo "  Downloading stdout file archive with git lfs..."
        git lfs pull --include "${rel_batch_dir}/sim-files-stdout.tar.gz"
    fi
    if [[ "$(file "${batch_dir}/sim-files-true-values.tar.gz")" =~ $asci_pattern ]]
    then
        echo "  Downloading true-values archive with git lfs..."
        git lfs pull --include "${rel_batch_dir}/sim-files-true-values.tar.gz"
    fi

    # Use subshell to change directory and extract archives
    (
        cd "$batch_dir"
        echo "  Extracting state logs..."
        tar xzf sim-files-state-logs.tar.gz || \
            { echo "  ERROR: Extraction failed!"; exit 1; }
        echo "  Extracting stdout files..."
        tar xzf sim-files-stdout.tar.gz || \
            { echo "  ERROR: Extraction failed!"; exit 1; }
        echo "  Extracting true-values files..."
        tar xzf sim-files-true-values.tar.gz || \
            { echo "  ERROR: Extraction failed!"; exit 1; }
    ) || exit 1

    # Back in script dir now that we've left the subshell
    echo "  Running parse_sim_results.py"
    ./parse_sim_results.py "$batch_dir"

    echo "  Removing extracted state log files..."
    rm "${batch_dir}/"run-?-*state*.log
    echo "  Removing extracted stdout files..."
    rm "${batch_dir}/"run-?-*.out
    echo "  Removing extracted true-values files..."
    rm "${batch_dir}/"simcoevolity-sim-*-true-values.txt
done
