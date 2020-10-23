#! /bin/bash

set -e


paths="$(ls ../results/plots/*.pdf)"

for f in $paths
do
    filesuffix="${f##*-}"
    if [ "$filesuffix" = "cropped.pdf" ]
    then
        continue
    fi
    
    echo "Cropping $f"
    n=${f/\.pdf/-cropped\.pdf}
    pdfcrop --margins 20 $f $n
done

paths="$(ls ../results/plots/*-nevents-cropped.pdf)"

for f in $paths
do
    filesuffix="${f##*-}"
    if [ "$filesuffix" = "rasterized.pdf" ]
    then
        continue
    fi
    
    echo "Rasterizing and compressing $f"
    n=${f/\.pdf/-rasterized\.pdf}
    convert -density 600 -compress jpeg -quality 80 $f $n
done
