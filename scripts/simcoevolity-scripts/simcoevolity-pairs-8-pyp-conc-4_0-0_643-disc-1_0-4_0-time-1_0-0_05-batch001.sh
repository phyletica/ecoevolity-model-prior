#! /bin/sh

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
    source ${PBS_O_HOME}/.bash_profile
    cd $PBS_O_WORKDIR
    source "${ECOEVOLITY_MODEL_PRIOR_PROJECT_DIR}/modules-to-load.sh" 
fi

path_to_this_script="$0"
name_of_this_script="$(basename "$path_to_this_script")"
name_after_prefix="${name_of_this_script#*simcoevolity-}"
config_name="${name_after_prefix%-batch*}"
config_path="../../ecoevolity-configs/${config_name}.yml"
output_dir="../../ecoevolity-simulations/${config_name}/batch001"
rng_seed=1
number_of_reps=10

exe_path="${ECOEVOLITY_MODEL_PRIOR_BIN_DIR}/simcoevolity"

mkdir -p "$output_dir"

$exe_path --seed="$rng_seed" -n "$number_of_reps" -o "$output_dir" "$config_path"
