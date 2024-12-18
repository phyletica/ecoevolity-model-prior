#!/bin/bash

function run_pyco_sumchains () {
    gzip -d -k run-?-${ecoevolity_config_prefix}-state-run-1.log.gz
    echo "pyco-sumchains run-?-${ecoevolity_config_prefix}-state-run-1.log"
    pyco-sumchains run-?-${ecoevolity_config_prefix}-state-run-1.log 1>pyco-sumchains-${ecoevolity_config_prefix}-table.txt 2>pyco-sumchains-${ecoevolity_config_prefix}-stderr.txt
    rm run-?-${ecoevolity_config_prefix}-state-run-1.log
}

set -e

current_dir="$(pwd)"
function return_on_exit () {
    cd "$current_dir"
}
trap return_on_exit EXIT

# Get path to project directory
project_dir="$( cd ../.. && pwd )"

if [ -n "$(command -v conda)" ]
then
    eval "$(conda shell.bash hook)"
    conda activate ecoevolity-model-prior-project
fi

ecoevolity_output_dir="${project_dir}/ecoevolity-gekkonid-outputs"

config_prefixes=( "cyrtodactylus-conc5-rate200" "cyrtodactylus-pyp5-rate200" "cyrtodactylus-unif5-rate200" )

for ecoevolity_config_prefix in "${config_prefixes[@]}"
do
    target_dir="${ecoevolity_output_dir}/${ecoevolity_config_prefix}"
    cd "${target_dir}"
    run_pyco_sumchains
done

cd "$current_dir"
