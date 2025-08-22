#!/bin/bash
# clean_xelatex_temp.sh
# Recursively delete common XeLaTeX/LaTeX temp files in current and subdirectories

find . -type f \( \
    -name '*.aux' -o \
    -name '*.log' -o \
    -name '*.out' -o \
    -name '*.toc' -o \
    -name '*.bbl' -o \
    -name '*.blg' -o \
    -name '*.fdb_latexmk' -o \
    -name '*.fls' -o \
    -name '*.synctex.gz' -o \
    -name '*.nav' -o \
    -name '*.snm' -o \
    -name '*.vrb' -o \
    -name '*.xdv' \
\) -print -delete
