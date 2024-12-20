#! /usr/bin/env bash 

set -e

for tex_path in grid*.tex
do
    path_prefix="${tex_path/\.tex/}"
    pdf_path="${path_prefix}.pdf"
    cropped_path="${path_prefix}-cropped.pdf"

    latexmk -C "$tex_path" && \
        latexmk -pdf "$tex_path" && \
        pdfcrop "$pdf_path" "$cropped_path" && \
        latexmk -C "$tex_path"
done
