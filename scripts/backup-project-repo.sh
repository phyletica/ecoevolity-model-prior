#! /bin/bash

set -e

# Make sure this script was called from within its directory
script_dir="$(pwd)"
if [ "$(basename "$script_dir")" != "scripts" ]
then
    echo "ERROR: this script must be called from the 'ecoevolity-model-prior/scripts' directory"
    exit 1
fi
project_dir="$(dirname "$script_dir")"
if [ "$(basename "$project_dir")" != "ecoevolity-model-prior" ]
then
    echo "ERROR: this script must be called from the 'ecoevolity-model-prior/scripts' directory"
    exit 1
fi
parent_dir="$(dirname "$project_dir")"

git_commit="$(git rev-parse --short HEAD)"
time_stamp="$(date '+%Y-%m-%d-%H-%M-%S')"
backup_dir="${time_stamp}-${git_commit}-ecoevolity-model-prior"
backup_path="aubox_phyletica_data:ecoevolity-model-prior/${backup_dir}"

rclone mkdir "$backup_path"
(
    cd "$parent_dir"
    echo "Starting rclone copy from the following directory: $(pwd)"
    echo "Starting rclone copy with the following command:"
    echo "  rclone copy ./ecoevolity-model-prior \"$backup_path\""
    rclone copy ./ecoevolity-model-prior "$backup_path"
    echo "Rclone copy finished!"
)
