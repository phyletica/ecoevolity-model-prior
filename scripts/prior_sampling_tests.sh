#! /bin/bash

set -e

if [ -n "$PBS_JOBNAME" ]
then
    cd $PBS_O_WORKDIR
fi

project_dir=".."
sim_exe_path="${project_dir}/bin/simcoevolity"
eco_exe_path="${project_dir}/bin/ecoevolity"
sum_exe_path="${project_dir}/bin/sumcoevolity"
plot_script_path="${project_dir}/scripts/plot_prior_samples.py"

if [ ! -f "$eco_exe_path" ]
then
    echo "ERROR: File '$eco_exe_path' does not exist."
    echo "       You probably need to run the project setup script."
    exit 1
fi

source "${project_dir}/modules-to-load.sh" >/dev/null 2>&1 || echo "    No modules loaded"

if [ -n "$(command -v conda)" ]
then
    eval "$(conda shell.bash hook)"
    conda activate ecoevolity-model-prior-project
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
            exit 1
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

for cfg_path in ${project_dir}/ecoevolity-configs/pairs-*.yml
do
    cfg_file="$(basename "$cfg_path")"
    cfg_name="${cfg_file%.*}"
    output_dir="../prior-sampling-tests/${cfg_name}"
    rng_seed=1
    
    mkdir -p "$output_dir"

    # Generate a single simulated dataset under the model specified in the
    # config
    $sim_exe_path --seed="$rng_seed" -n 1 -o "$output_dir" "$cfg_path"

    # Increase the number of MCMC samples from the prior by adding zeros to the
    # 'chain_length' setting in the config file output by simcoevolity
    sim_cfg_path="${output_dir}/simcoevolity-sim-0-config.yml"
    sed -i "s/chain_length: *[0-9]\+/&$extra_zeros/g" "$sim_cfg_path"

    # Sample from the prior using ecoevolity
    $eco_exe_path --seed="$rng_seed" --ignore-data "$sim_cfg_path"

    state_log_path="${output_dir}/simcoevolity-sim-0-config-state-run-1.log"

    # Use sumcoevolity to compare model prior expectations to samples 
    sum_output_prefix="${output_dir}/"
    $sum_exe_path --seed="$rng_seed" -p "$sum_output_prefix" -b 101 -c "$cfg_path" -n 1000000 "$state_log_path"
    if [ -n "$(command -v pyco-sumevents)" ]
    then
        (
            cd "$output_dir"
            pyco-sumevents -f "sumcoevolity-results-nevents.txt"
        )
    fi

    # Use custom plotting script to compare prior expectations to samples for
    # all other parameters
    $plot_script_path -b 101 "$cfg_path" "$state_log_path"
done
