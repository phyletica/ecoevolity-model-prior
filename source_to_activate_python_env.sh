# Get path to directory of this file
project_dir="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ -f "${project_dir}/project-python-env/bin/activate" ]
then
    source "${project_dir}/project-python-env/bin/activate"
else
    echo "ERROR: Project python environment doesn't exist."
    echo "       You probably need to run the project setup script."
fi
