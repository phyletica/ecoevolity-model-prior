#! /bin/bash

set -e

if [ -n "$PBS_JOBNAME" ]
then
    cd $PBS_O_WORKDIR
fi

rng_seed=799492476
number_of_reps=10

config_name="pairs-10-pyp-conc-2_0-1_79-disc-1_0-4_0-time-1_0-0_05"
config_path="../../ecoevolity-configs/${config_name}.yml"
output_dir="../../ecoevolity-simulations/${config_name}/batch-${rng_seed}"

project_dir="../.."
exe_path="${project_dir}/bin/simcoevolity"
config_set_up_script_path="${project_dir}/scripts/set_up_configs_for_simulated_data.py"
qsub_set_up_script_path="${project_dir}/scripts/set_up_ecoevolity_qsubs.py"

if [ ! -f "$exe_path" ]
then
    echo "ERROR: File '$exe_path' does not exist."
    echo "       You probably need to run the project setup script."
    exit 1
fi

source "${project_dir}/modules-to-load.sh" >/dev/null 2>&1 || echo "    No modules loaded"

if [ -n "$(command -v conda)" ]
then
    eval "$(conda shell.bash hook)"
    conda activate ecoevolity-model-prior-project
fi

mkdir -p "$output_dir"

$exe_path --seed="$rng_seed" -n "$number_of_reps" -o "$output_dir" "$config_path" && $config_set_up_script_path "$output_dir" && $qsub_set_up_script_path "$output_dir"
