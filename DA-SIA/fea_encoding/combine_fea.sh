#!/bin/bash
#$ -N combine_fea
#$ -S /bin/bash
#$ -cwd
#$ -o logs/UGE$JOB_ID.o
#$ -j y
#$ -l virtual_free=64G

SCRIPT_DIR="/grid/siepel/home_norepl/mo/dom_adapt/fea_encoding"

echo "_START_$(date)"

${SCRIPT_DIR}/combine_fea.py $1 $2 $3

echo "_EXITSTAT_$?"
echo "_END_$(date)"
exit