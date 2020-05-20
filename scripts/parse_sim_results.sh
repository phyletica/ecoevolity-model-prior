#! /bin/bash

set -e

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

./parse_sim_results.py $@
