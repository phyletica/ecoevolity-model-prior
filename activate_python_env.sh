#! /bin/bash

set -e

# Get path to directory of this script
project_dir="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Load modules
echo "Loading modules..."
source ./modules-to-load.sh >/dev/null 2>&1 || echo "    No modules loaded"

if [ -f "${project_dir}/project-python-env/bin/activate" ]
then
    source "${project_dir}/project-python-env/bin/activate"
else
    echo "ERROR: Project python environment doesn't exist"
    echo "  You probably need to run \"setup_project_env.sh\""
fi
