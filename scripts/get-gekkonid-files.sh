#! /bin/bash

set -e

write_hline() {
    echo "########################################################################"
}

write_banner_msg() {
    echo ""
    write_hline
    echo "$@"
    write_hline
}

# Remove the temporary directory created for Gekgo repo
clean_up() {
  test -d "${tmp_dir}/.git" && rm -rf "${tmp_dir}/.git"
  test -d "$tmp_dir" && rm -r "$tmp_dir"
}

# Ensure temp dir gets deleted on exit
trap clean_up EXIT
# Make temp dir for cloning Gekgo repo
# The or ("||") syntax is to support different flavors of `mktemp`; Darwin's
# mktemp behaves differently than Gnu's.
tmp_dir="$(mktemp -d 2>/dev/null || mktemp -d -t 'tmp_gekgo')"

# Get path to directory of this script
script_dir="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
project_dir="$(dirname "$script_dir")"

nex2yml_path="${project_dir}/bin/nex2yml"

data_dir="${project_dir}/data"

gekgo_dir="$tmp_dir"
write_banner_msg "Cloning v2.0.0 of gekgo repo..."
git clone --depth 1 --branch "v2.0.0" "git@github.com:phyletica/gekgo.git" "$gekgo_dir"

gekgo_config_dir="${gekgo_dir}/data/genomes/msg/ecoevolity-configs"
gekgo_data_dir="${gekgo_dir}/data/genomes/msg/alignments"
gekgo_out_dir="${gekgo_dir}/data/genomes/msg/ecoevolity-output"
config_path="${gekgo_config_dir}/cyrtodactylus-conc5-rate200.yml"

write_banner_msg "Converting gekgo nexus files to yaml allele count files..."
"$nex2yml_path" --relax-missing-sites --relax-triallelic-sites "$config_path"

write_banner_msg "Moving yaml allele count files into data directory..."
for yml_path in "$gekgo_data_dir"/*.nex.yml
do
    yml_name="$(basename $yml_path)"
    yml_out_name="$(echo "$yml_name" | sed -e "s/\.nex//g")"
    yml_out_path="${data_dir}/${yml_out_name}"
    mv "$yml_path" "$yml_out_path"
done
