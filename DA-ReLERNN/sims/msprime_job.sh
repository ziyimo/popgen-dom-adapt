#!/bin/bash
#$ -N msprime_rho_sim
#$ -S /bin/bash
#$ -cwd
#$ -o logs/UGE$JOB_ID.o
#$ -j y
#$ -l m_mem_free=8G
#$ -l virtual_free=20G

echo "_START_$(date)"
echo "Job Command: $@"

"$@"

echo "_EXITSTAT_$?"
echo "_END_$(date)"
exit