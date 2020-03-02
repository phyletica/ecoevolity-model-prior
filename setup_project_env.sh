#! /bin/bash

set -e

ecoevolity_commit="932c358ce"

# Get path to directory of this script
project_dir="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

bashrc_path="${HOME}/.bashrc"
bash_project_config_path="${HOME}/.bash_ecoevolity_model_prior_project"

# Create a hidden config file in the home directory with useful path variables
# This will overwrite the config file if it exists to make sure it is current
echo "Creating shell config file for the project in the home directory... "
echo "export ECOEVOLITY_MODEL_PRIOR_PROJECT_DIR=\"${project_dir}\"" > "$bash_project_config_path"
echo "export ECOEVOLITY_MODEL_PRIOR_BIN_DIR=\"${project_dir}/bin\"" >> "$bash_project_config_path"
echo "    Created \"$bash_project_config_path\""

# Load modules
echo "Loading modules..."
source ./modules-to-load.sh >/dev/null 2>&1 || echo "    No modules loaded"

echo "Cloning and building ecoevolity..."
git clone https://github.com/phyletica/ecoevolity.git
(
    cd ecoevolity
    git checkout -b testing "$ecoevolity_commit"
    ./build.sh --prefix "$project_dir"
    echo "    Commit $ecoevolity_commit of ecoevolity successfully built and installed"
    cd ..
)

if [ -n "$(command -v conda)" ]
then
    eval "$(conda shell.bash hook)"
    conda env create -f conda-environment.yml
    conda activate ecoevolity-model-prior-project
fi
