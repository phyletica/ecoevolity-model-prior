#! /bin/bash

set -e

if [ -n "$PBS_JOBNAME" ]
then
    cd $PBS_O_WORKDIR
fi

source "${project_dir}/modules-to-load.sh" >/dev/null 2>&1 || echo "    No modules loaded"

if [ -n "$(command -v conda)" ]
then
    eval "$(conda shell.bash hook)"
    conda activate ecoevolity-model-prior-project
fi

project_dir=".."
bin_dir="${project_dir}/bin"
submission_executable="${bin_dir}/psub"
extra_args=()
expected_nsamples=1501
burnin=501
restrict_nodes=''

usage () {
    echo ""
    echo "Usage:"
    echo "  $0 [ OPTIONS ] <PATH-TO-SIM-DIR-1> [ <PATH-TO-SIM-DIR-2> ... ]"
    echo ""
    echo "Required positional argument:"
    echo "  PATH-TO-SIM-DIR  Path to diretory containing simcoevolity output"
    echo "                   files to be analyzed with ecoevolity."
    echo "Optional arguments:"
    echo "  -h|--help        Show help message and exit."
    echo "  -s               Number of MCMC samples that should be found in"
    echo "                   each log file of each analysis."
    echo "                   Default: $expected_nsamples."
    echo "                   Default: $wtime."
    echo "  --burnin         Number of MCMC samples to be ignored as burnin"
    echo "                   from the beginning of every chain."
    echo "                   Default: $burnin."
    echo "  --nsub           Use 'nsub' submission script."
    echo "                   Default is to use 'psub' script."
    echo "  -r|--restrict    Restrict job to lab nodes."
    echo ""
}

# process args

if [ "$(echo "$@" | grep -c "=")" -gt 0 ]
then
    echo "ERROR: Do not use '=' for arguments. For example, use"
    echo "'--burnin 501' instead of '--burnin=501'."
    usage
    exit 1
fi

while [ "$1" != "" ]
do
    case $1 in
        -h| --help)
            usage
            exit
            ;;
        -s)
            shift
            expected_nsamples="$1"
            shift
            ;;
        --burnin)
            shift
            burnin="$1"
            shift
            ;;
        -r| --restrict)
            shift
            restrict_nodes=1
            ;;
        --nsub)
            shift
            submission_executable="${bin_dir}/nsub"
            ;;
        *)
            extra_args+=( "$1" )
    esac
    shift
done

if [ ! -x "$submission_executable" ]
then
    echo "ERROR: No executable '$submission_executable'"
    exit 1
fi

# Make sure there are positional arguments
if [ ! "${#extra_args[*]}" -gt 1 ]
then
    echo "ERROR: At least one argument must be provided; paths to a sim directories"
    usage
    exit 1
fi

python_opts="-s $expected_nsamples --burnin $burnin"
psub_opts=""
if [ -n "$restrict_nodes" ]
then
    psub_opts="-r ${psub_opts}"
fi

for batch_dir in "${extra_args[@]}"
do
    "$submission_executable" $psub_opts ./parse_sim_results.py $python_opts "$batch_dir"
done
