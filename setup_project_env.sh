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

# Python 3 is needed to setup the python environment, but the module is not
# needed once it is setup
module load python/3.6.4 >/dev/null 2>&1 || echo ""

python3 -m venv project-python-env
source project-python-env/bin/activate
pip3 install --upgrade pip
pip3 install wheel
pip3 install -r python-requirements.txt
