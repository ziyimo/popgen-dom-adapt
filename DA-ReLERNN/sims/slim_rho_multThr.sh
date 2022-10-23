#!/bin/bash
#$ -N SLiM_rho
#$ -S /bin/bash
#$ -cwd
#$ -o logs/$JOB_ID_$TASK_ID.o
#$ -e logs/$JOB_ID_$TASK_ID.e
#$ -l m_mem_free=2G
#$ -l virtual_free=16G

## specify at submit time
# -t 1-3
# -tc 3

GITPATH="/grid/siepel/home_norepl/mo/dom_adapt/popgen-dom-adapt"

echo "_START_$(date)"

COND="bkgd" # ("reg" or "bkgd")
GENELEN=$1
TAG=$2
RUNS=$3

${GITPATH}/DA-ReLERNN/sims/slim_rho_sims.py --G $GENELEN $COND $TAG $((SGE_TASK_ID-1)) $RUNS
echo "_EXITSTAT_$?"
echo "_END_$(date)"
exit
