#! /bin/bash

set -e

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

if [ "$#" -gt 1 ]
then
    echo "ERROR: At most one argument allowed; number of extra zeros for chain length"
    exit 1
fi

num_extra_zeros="0"
if [ "$#" = 1 ]
then
    case "$1" in
        ''|*[!0-9]*)
            echo "ERROR: Argument is not an integer"
            ;;
        *)
            num_extra_zeros="$1"
            ;;
    esac
fi

echo "Adding $num_extra_zeros extra zeros to the chain length of all configs"
extra_zeros=""
for i in $(seq 1 "$num_extra_zeros")
do
    extra_zeros="${extra_zeros}0"
done

echo $extra_zeros

sim_exe_path="${ECOEVOLITY_MODEL_PRIOR_BIN_DIR}/simcoevolity"
eco_exe_path="${ECOEVOLITY_MODEL_PRIOR_BIN_DIR}/ecoevolity"
sum_exe_path="${ECOEVOLITY_MODEL_PRIOR_BIN_DIR}/sumcoevolity"

# for cfg_path in ${ECOEVOLITY_MODEL_PRIOR_PROJECT_DIR}/ecoevolity-configs/*.yml
for cfg_path in ${ECOEVOLITY_MODEL_PRIOR_PROJECT_DIR}/ecoevolity-configs/*dpp-conc*.yml
do
    cfg_file="$(basename "$cfg_path")"
    cfg_name="${cfg_file%.*}"
    output_dir="../prior-sampling-tests/${cfg_name}"
    rng_seed=1
    number_of_reps=1
    
    mkdir -p "$output_dir"

    $sim_exe_path --seed="$rng_seed" -n "$number_of_reps" -o "$output_dir" "$cfg_path"

    sim_cfg_path="${output_dir}/simcoevolity-sim-0-config.yml"
    sed -i "s/chain_length: *[0-9]\+/&$extra_zeros/g" "$sim_cfg_path"

    $eco_exe_path --seed="$rng_seed" --ignore-data "$sim_cfg_path"

    state_log_path="${output_dir}/simcoevolity-sim-0-config-state-run-1.log"

    $sum_exe_path --seed="$rng_seed" -b 101 -c "$cfg_path" -n 100000 "$state_log_path"
    if [ -n "$(command -v pyco-sumevents)" ]
    then
        pyco-sumevents -f "${output_dir}/sumcoevolity-results-nevents.txt"
    fi

done
