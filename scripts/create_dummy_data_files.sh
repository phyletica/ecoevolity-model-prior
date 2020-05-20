#!/bin/bash

nspecies=2
ngenomes=4
nchars=500000

data_dir="../data"
mkdir -p "$data_dir"

# Generate 500k dummy alignments
i=1
while [ "$i" -lt 11 ]
do
    prefix="c${i}sp"
    comp_str="0${i}"
    if [ ${#i} -gt 1 ]
    then
        comp_str="$i"
    fi
    outfile="${data_dir}/comp${comp_str}-${nspecies}species-${ngenomes}genomes-${nchars}chars.txt"
    ./generate_dummy_data_file.py \
        --nspecies "$nspecies" \
        --ngenomes "$ngenomes" \
        --ncharacters "$nchars" \
        --prefix "$prefix" \
        > "$outfile"
    i=`expr $i + 1`
done
