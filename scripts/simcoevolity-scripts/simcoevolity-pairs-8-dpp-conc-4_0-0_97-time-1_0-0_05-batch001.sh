#! /bin/bash

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

config_name="pairs-8-dpp-conc-4_0-0_97-time-1_0-0_05"
config_path="../../ecoevolity-configs/${config_name}.yml"
output_dir="../../ecoevolity-simulations/${config_name}/batch001"
rng_seed=1
number_of_reps=10

exe_path="${ECOEVOLITY_MODEL_PRIOR_BIN_DIR}/simcoevolity"
config_set_up_script_path="${ECOEVOLITY_MODEL_PRIOR_PROJECT_DIR}/scripts/set-up-configs-for-simulated-data.py"
qsub_set_up_script_path="${ECOEVOLITY_MODEL_PRIOR_PROJECT_DIR}/scripts/set-up-ecoevolity-qsubs.py"

mkdir -p "$output_dir"

$exe_path --seed="$rng_seed" -n "$number_of_reps" -o "$output_dir" "$config_path" && $config_set_up_script_path "$output_dir" && $qsub_set_up_script_path "$output_dir"
