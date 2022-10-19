#!/bin/bash
#$ -N combine_fea_meta
#$ -S /bin/bash
#$ -cwd
#$ -o UGE$JOB_ID.o
#$ -j y
#$ -l m_mem_free=64G

GITPATH="/grid/siepel/home_norepl/mo/dom_adapt/popgen-dom-adapt/DA-SIA/fea_encoding"

echo "_START_$(date)"

${GITPATH}/combine_fea_meta.py $1 $2 $3

echo "_EXITSTAT_$?"
echo "_END_$(date)"
exit