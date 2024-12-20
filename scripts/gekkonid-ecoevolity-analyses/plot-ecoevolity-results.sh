#!/bin/bash

function run_summary_tools () {
    time_plot_width="$plot_width"
    gzip -d -k run-?-${ecoevolity_config_prefix}-state-run-1.log.gz
    pyco-sumtimes -f -w "$time_plot_width" --violin -y "$time_ylabel" -x "$time_xlabel" -b $burnin "${label_array[@]}" -p "${plot_dir}/pyco-sumtimes-${ecoevolity_config_prefix}-" run-?-${ecoevolity_config_prefix}-state-run-1.log
    pyco-sumsizes -f -w "$plot_width" --violin --base-font-size $size_base_font -y "$size_ylabel" -b $burnin "${label_array[@]}" -p "${plot_dir}/pyco-sumsizes-${ecoevolity_config_prefix}-" run-?-${ecoevolity_config_prefix}-state-run-1.log
    if [ -e "${plot_dir}/sumcoevolity-${ecoevolity_config_prefix}-sumcoevolity-results-nevents.txt" ]
    then
        rm "${plot_dir}/sumcoevolity-${ecoevolity_config_prefix}-sumcoevolity-results-nevents.txt"
    fi
    if [ -e "${plot_dir}/sumcoevolity-${ecoevolity_config_prefix}-sumcoevolity-results-model.txt" ]
    then
        rm "${plot_dir}/sumcoevolity-${ecoevolity_config_prefix}-sumcoevolity-results-model.txt"
    fi
    if [ -z "$sumco_seed" ]
    then
        sumco_seed=7
    fi
    "$sumco_exe_path" --seed "$sumco_seed" -b $burnin -n 1000000 -p "${plot_dir}/sumcoevolity-${ecoevolity_config_prefix}-" -c "${config_dir}/${ecoevolity_config_prefix}.yml" run-?-${ecoevolity_config_prefix}-state-run-1.log
    pyco-sumevents -f -w "$plot_width" --bf-font-size $bf_font_size -p "${plot_dir}/pyco-sumevents-${ecoevolity_config_prefix}-" --full-prob-axis --legend-in-plot "${plot_dir}/sumcoevolity-${ecoevolity_config_prefix}-sumcoevolity-results-nevents.txt"
    rm run-?-${ecoevolity_config_prefix}-state-run-1.log
}

set -e

label_array=()
convert_labels_to_array() {
    local concat=""
    local t=""
    label_array=()

    for word in $@
    do
        local len=`expr "$word" : '.*"'`

        [ "$len" -eq 1 ] && concat="true"

        if [ "$concat" ]
        then
            t+=" $word"
        else
            word=${word#\"}
            word=${word%\"}
            label_array+=("$word")
        fi

        if [ "$concat" -a "$len" -gt 1 ]
        then
            t=${t# }
            t=${t#\"}
            t=${t%\"}
            label_array+=("$t")
            t=""
            concat=""
        fi
    done
}

burnin=101

current_dir="$(pwd)"
function return_on_exit () {
    cd "$current_dir"
}
trap return_on_exit EXIT

# Get path to project directory
project_dir="$( cd ../.. && pwd )"

if [ -n "$(command -v conda)" ]
then
    eval "$(conda shell.bash hook)"
    conda activate ecoevolity-model-prior-project
fi

ecoevolity_output_dir="${project_dir}/ecoevolity-gekkonid-outputs"
config_dir="${project_dir}/ecoevolity-configs"
sumco_exe_path="${project_dir}/bin/sumcoevolity"
sumco_seed=7

plot_base_dir="${project_dir}/results/gekkonid-plots"
if [ ! -d "$plot_base_dir" ]
then
    mkdir -p "$plot_base_dir"
fi

labels='-l "Bohol0" "Bohol"
-l "CamiguinSur0" "Camiguin Sur"
-l "root-Bohol0" "Bohol-Camiguin Sur Root"
-l "Palawan1" "Palawan"
-l "Kinabalu1" "Borneo"
-l "root-Palawan1" "Palawan-Borneo Root"
-l "Samar2" "Samar"
-l "Leyte2" "Leyte"
-l "root-Samar2" "Samar-Leyte Root"
-l "Luzon3" "Luzon 1"
-l "BabuyanClaro3" "Babuyan Claro"
-l "root-Luzon3" "Luzon-Babuyan Claro Root"
-l "Luzon4" "Luzon 2"
-l "CamiguinNorte4" "Camiguin Norte"
-l "root-Luzon4" "Luzon-Camiguin Norte Root"
-l "Polillo5" "Polillo"
-l "Luzon5" "Luzon 3"
-l "root-Polillo5" "Polillo-Luzon Root"
-l "Panay6" "Panay"
-l "Negros6" "Negros"
-l "root-Panay6" "Panay-Negros Root"
-l "Sibuyan7" "Sibuyan"
-l "Tablas7" "Tablas"
-l "root-Sibuyan7" "Sibuyan-Tablas Root"
-l "BabuyanClaro8" "Babuyan Claro"
-l "Calayan8" "Calayan"
-l "root-BabuyanClaro8" "Babuyan Claro-Calayan Root"
-l "SouthGigante9" "S. Gigante"
-l "NorthGigante9" "N. Gigante"
-l "root-SouthGigante9" "S. Gigante-N. Gigante Root"
-l "Lubang11" "Lubang"
-l "Luzon11" "Luzon"
-l "root-Lubang11" "Lubang-Luzon Root"
-l "MaestreDeCampo12" "Maestre De Campo"
-l "Masbate12" "Masbate"
-l "root-MaestreDeCampo12" "Maestre De Campo-Masbate Root"
-l "Panay13" "Panay 1"
-l "Masbate13" "Masbate"
-l "root-Panay13" "Panay-Masbate Root"
-l "Negros14" "Negros"
-l "Panay14" "Panay 2"
-l "root-Negros14" "Negros-Panay Root"
-l "Sabtang15" "Sabtang"
-l "Batan15" "Batan"
-l "root-Sabtang15" "Sabtang-Batan Root"
-l "Romblon16" "Romblon"
-l "Tablas16" "Tablas"
-l "root-Romblon16" "Romblon-Tablas Root"
-l "CamiguinNorte17" "Camiguin Norte"
-l "Dalupiri17" "Dalupiri"
-l "root-CamiguinNorte17" "Camiguin Norte-Dalupiri Root"'

convert_labels_to_array $labels

time_ylabel="Island pair"
time_xlabel="Divergence time (substitutions/site)"
size_ylabel="Population"

bf_font_size=2.0

size_base_font=8.0

plot_width=7.0

config_prefixes=( "cyrtodactylus-conc5-rate200" "cyrtodactylus-pyp5-rate200" "cyrtodactylus-unif5-rate200" )

for ecoevolity_config_prefix in "${config_prefixes[@]}"
do
    input_dir="${ecoevolity_output_dir}/${ecoevolity_config_prefix}"
    plot_dir="${plot_base_dir}/${ecoevolity_config_prefix}"
    if [ ! -d "$plot_dir" ]
    then
        mkdir "$plot_dir"
    fi
    cd "$input_dir"

    run_summary_tools

    cd "$plot_dir"
    
    for p in pyco-*.pdf
    do
        pdfcrop "$p" "$p"
    done
done

cd "$current_dir"
