#!/bin/bash

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

usage () {
    echo ""
    echo "Usage:"
    echo "  $0 [ OPTIONS ] <PATH-TO-SIM-DIR>"
    echo ""
    echo "Required positional argument:"
    echo "  PATH-TO-SIM-DIR  Path to diretory containing ecoeovlity output of "
    echo "                   analyses of data sets simulated with simcoevolity."
    echo "Optional arguments:"
    echo "  -h|--help        Show help message and exit."
    echo "  -t|--walltime   Max time limit for job."
    echo "                  Default: 00:30:00."
    echo "  -r|--restrict   Restrict job to lab nodes."
    echo "  --nsub          Use 'nsub' submission script."
    echo "                  Default is to use 'psub' script."
    echo ""
}

# process args
submission_executable="${ECOEVOLITY_MODEL_PRIOR_BIN_DIR}/psub"
extra_args=()
restrict_nodes=''
wtime='00:30:00'

if [ "$(echo "$@" | grep -c "=")" -gt 0 ]
then
    echo "ERROR: Do not use '=' for arguments. For example, use"
    echo "'--nlines 1502' instead of '--nlines=1502'."
    exit 1
fi

while [ "$1" != "" ]
do
    case $1 in
        -h| --help)
            usage
            exit
            ;;
        -t| --walltime)
            shift
            wtime="$1"
            shift
            ;;
        -r| --restrict)
            shift
            restrict_nodes=1
            ;;
        --nsub)
            shift
            submission_executable="${ECOEVOLITY_MODEL_PRIOR_BIN_DIR}/nsub"
            ;;
        *)
            extra_args+=( "$1" )
    esac
    shift
done

# Make sure there is exactly one positional argument
if [ ! "${#extra_args[*]}" = 1 ]
then
    echo "ERROR: One argument must be provided; a path to a sim directory"
    usage
    exit 1
fi

# Path to the batch directory should be the only argument
batch_dir="${extra_args[0]}"
# Remove any trailing slashes from the path
batch_dir="$(echo "$batch_dir" | sed 's:/*$::')"

# Make sure the argument is a valid directory
if [ ! -d "$batch_dir" ]
then
    echo "ERROR: Path is not a valid directory: $batch_dir"
    usage
    exit 1
fi

# Make sure the directory is a batch directory
if [ ! "$(echo "$batch_dir" | grep -c -E "batch[0-9]{1,4}$")" -gt 0 ]
then
    echo "ERROR: The path provided doesn't seem to be a sim batch directory:"
    echo "    $batch_dir"
    usage
    exit 1
fi

psub_flags="-t ${wtime}"
if [ -n "$restrict_nodes" ]
then
    psub_flags="-r ${psub_flags}"
fi

echo "Beginning to vet and consolidate sim analysis files..."
reruns=()
for qsub_path in ${batch_dir}/*pairs-*-sim-*-config-run-*-qsub.sh
do
    to_run="${qsub_path/-qsub.sh/}"
    run_number="${to_run##*-}"
    qsub_file_name="$(basename "$qsub_path")"
    dir_path="$(dirname "$qsub_path")"
    base_prefix="${qsub_file_name%-run-*}"
    file_prefix="run-${run_number}-${base_prefix}"
    prefix="${dir_path}/${file_prefix}"
    out_file="${prefix}.yml.out"
    state_log="${prefix}-state-run-1.log"
    op_log="${prefix}-operator-run-1.log"

    # Consolidate state logs if run was restarted 
    extra_run_number=2
    while [ -e "${prefix}-state-run-${extra_run_number}.log" ]
    do
        mv "${prefix}-state-run-${extra_run_number}.log" "$state_log"
        ((++extra_run_number))
    done

    # Consolidate operator logs if run was restarted 
    extra_run_number=2
    while [ -e "${prefix}-operator-run-${extra_run_number}.log" ]
    do
        mv "${prefix}-operator-run-${extra_run_number}.log" "$op_log"
        ((++extra_run_number))
    done

    if [ ! -e "$out_file" ] 
    then
        echo "No stdout: $qsub_path" 
        reruns+=( "$qsub_path" )
        continue
    fi

    if [ ! -e "$state_log" ] 
    then
        echo "No state log: $qsub_path" 
        reruns+=( "$qsub_path" )
        continue
    fi

    runtime_line_count="$(grep -c "Runtime:" "$out_file")"
    if [ "$runtime_line_count" != 1 ]
    then 
        echo "Incomplete stdout: $qsub_path" 
        reruns+=( "$qsub_path" )
        continue
    fi

    nlines="$(wc -l "$state_log" | awk '{print $1}')"
    if [ "$nlines" != "$expected_nlines" ] 
    then
        echo "Incomplete log: $qsub_path" 
        reruns+=( "$qsub_path" )
        continue
    fi

    seed_line="$(grep "seed" "$qsub_path")"
    after_seed="${seed_line##*--seed}"
    expected_seed="$(echo ${after_seed%%--*} | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    seed_line="$(grep -i "seed" "$out_file")"
    seed="$(echo ${seed_line##*:} | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    
    if [ "$expected_seed" != "$seed" ]
    then
        echo "Bad seed: $qsub_path"
        reruns+=( "$qsub_path" )
        continue
    fi
done

if [ "${#reruns[*]}" = 0 ]
then
    echo "All analyses are complete and clean!"
    exit 0
fi

echo "Submitting analyses to queue..."
for qsub_path in "${reruns[@]}"
do
    dir_path="$(dirname "$qsub_path")"
    file_name="$(basename "$qsub_path")"

    (
        cd "$dir_path"
        ls "$file_name"

        prefix="${file_name/-qsub\.sh/}"
        run_number="${prefix##*run-}"
        sim_base="${prefix%-run-*}"

        op_log_file="run-${run_number}-${sim_base}-operator-run-1.log"
        state_log_file="run-${run_number}-${sim_base}-state-run-1.log"
        stdout_file="run-${run_number}-${sim_base}.yml.out"

        if [ -e "$op_log_file" ]
        then
            rm "$op_log_file"
        fi

        if [ -e "$state_log_file" ]
        then
            rm "$state_log_file"
        fi

        if [ -e "$stdout_file" ]
        then
            rm "$stdout_file"
        fi

        $submission_executable $psub_flags "$file_name"
    )
done
