#!/bin/bash

set -e

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
    echo "  -l|--nlines      The number of lines expected in each state log file."
    echo "                   Default: 1502"
    echo ""
}

# process args
expected_nlines=1502
extra_args=()

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
        -l| --nlines)
            shift
            expected_nlines="$1"
            ;;
        * )
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

echo "Re-starting failed analyses..."
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

        psub -t "01:20:00" "$file_name"
    )
done
