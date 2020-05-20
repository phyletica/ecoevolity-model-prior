#! /bin/bash

set -e

ecoevolity_commit="c1685dfa"

# Get path to directory of this script
project_dir="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

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
