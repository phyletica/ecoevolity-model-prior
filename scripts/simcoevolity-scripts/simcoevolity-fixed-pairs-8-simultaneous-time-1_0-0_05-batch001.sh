#! /bin/bash

if [ -f "${HOME}/.bash_ecoevolity_model_prior_project" ]
then
    source "${HOME}/.bash_ecoevolity_model_prior_project"
else
    echo "ERROR: File '~/.bash_ecoevolity_model_prior_project' does not exist."
    echo "       You probably need to run the project setup script."
    exit 1
fi

if [ -z "$ECOEVOLITY_MODEL_PRIOR_PROJECT_DIR" ]
then
    echo "ERROR: ECOEVOLITY_MODEL_PRIOR_PROJECT_DIR is not set"
    echo "       You probably need to run the project setup script."
    exit 1
elif [ ! -d "$ECOEVOLITY_MODEL_PRIOR_PROJECT_DIR" ]
then
    echo "ERROR: ECOEVOLITY_MODEL_PRIOR_PROJECT_DIR is not a valid directory:"
    echo "       '$ECOEVOLITY_MODEL_PRIOR_PROJECT_DIR'"
    echo "       You probably need to run the project setup script."
    exit 1
fi

if [ -z "$ECOEVOLITY_MODEL_PRIOR_BIN_DIR" ]
then
    echo "ERROR: ECOEVOLITY_MODEL_PRIOR_BIN_DIR is not set"
    exit 1
elif [ ! -d "$ECOEVOLITY_MODEL_PRIOR_BIN_DIR" ]
then
    echo "ERROR: ECOEVOLITY_MODEL_PRIOR_BIN_DIR is not a valid directory:"
    echo "       '$ECOEVOLITY_MODEL_PRIOR_BIN_DIR'"
    echo "       You probably need to run the project setup script."
    exit 1
fi

if [ -n "$PBS_JOBNAME" ]
then
    cd $PBS_O_WORKDIR
    source "${ECOEVOLITY_MODEL_PRIOR_PROJECT_DIR}/modules-to-load.sh" 
fi

config_name="fixed-pairs-8-simultaneous-time-1_0-0_05"
config_path="../../ecoevolity-configs/${config_name}.yml"
output_dir="../../ecoevolity-simulations/${config_name}/batch001"
rng_seed=1
number_of_reps=10

exe_path="${ECOEVOLITY_MODEL_PRIOR_BIN_DIR}/simcoevolity"
config_set_up_script_path="${ECOEVOLITY_MODEL_PRIOR_PROJECT_DIR}/scripts/set_up_configs_for_simulated_data.py"
qsub_set_up_script_path="${ECOEVOLITY_MODEL_PRIOR_PROJECT_DIR}/scripts/set_up_ecoevolity_qsubs.py"

mkdir -p "$output_dir"

$exe_path --seed="$rng_seed" -n "$number_of_reps" -o "$output_dir" "$config_path" && $config_set_up_script_path "$output_dir" && $qsub_set_up_script_path "$output_dir"
