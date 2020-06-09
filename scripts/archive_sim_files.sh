#!/bin/bash

set -e

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

# Make sure there is at least one positional argument
if [ ! "$#" -gt 0 ]
then
    echo "ERROR: At least one argument must be provided; a path to a sim directory"
    usage
    exit 1
fi

# Vet arguments before archiving/removing anything
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
    
    echo "Cleaning up contents of:"
    echo "  ${batch_dir}"

    (
        cd "$batch_dir"
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
        for p in run-?-*operator*.log; do if [ -e "$p" ]; then
            echo "Archiving and removing ecoevolity operator logs..."
            tar czf sim-files-op-logs.tar.gz run-?-*operator*.log && rm run-?-*operator*.log
            break
        fi; done
        for p in run-?-*state*.log; do if [ -e "$p" ]; then
            echo "Archiving and removing ecoevolity state logs..."
            tar czf sim-files-state-logs.tar.gz run-?-*state*.log && rm run-?-*state*.log
            break
        fi; done
        for p in run-?-*.out; do if [ -e "$p" ]; then
            echo "Archiving and removing ecoevolity stdout files..."
            tar czf sim-files-stdout.tar.gz run-?-*.out && rm run-?-*.out
            break
        fi; done
        for p in simcoevolity-sim-*-true-values.txt; do if [ -e "$p" ]; then
            echo "Archiving and removing files with true values..."
            tar czf sim-files-true-values.tar.gz simcoevolity-sim-*-true-values.txt && rm simcoevolity-sim-*-true-values.txt
            break
        fi; done
    )
done
