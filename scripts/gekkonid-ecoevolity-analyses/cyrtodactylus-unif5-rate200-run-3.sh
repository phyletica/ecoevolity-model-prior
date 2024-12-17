#! /bin/bash

set -e

run=3
config_prefix="cyrtodactylus-unif5-rate200"

# Get path to project directory
project_dir="$( cd ../.. && pwd )"

# Load modules
echo "Loading modules..."
source "${project_dir}/modules-to-load.sh" >/dev/null 2>&1 || echo "    No modules loaded"

base_output_dir="${project_dir}/ecoevolity-gekkonid-outputs"
if [ ! -d "$base_output_dir" ]
then
    mkdir "$base_output_dir"
fi

output_dir="${base_output_dir}/${config_prefix}"
if [ ! -d "$output_dir" ]
then
    mkdir "$output_dir"
fi

config_path="${project_dir}/ecoevolity-configs/${config_prefix}.yml"
prefix="${output_dir}/run-${run}-"

"${project_dir}/bin/ecoevolity" --seed "$run" --prefix "$prefix" "$config_path" 1>"${prefix}${config_prefix}.out" 2>&1
