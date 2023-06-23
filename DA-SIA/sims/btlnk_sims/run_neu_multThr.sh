#!/bin/bash
#$ -N msprime_array
#$ -S /bin/bash
#$ -cwd
#$ -o logs/$JOB_ID_$TASK_ID.o
#$ -e logs/$JOB_ID_$TASK_ID.e
#$ -l m_mem_free=2G

## These should be passed in while submitting the job
# -t 1-100
# -tc 50

echo "_START_$(date)"

GITDIR="/grid/siepel/home_norepl/mo/dom_adapt/popgen-dom-adapt/DA-SIA/sims/btlnk_sims"


NBTLNK=$1 # bottleneck size
RUNS=$2 # no of new runs PER THREAD
LASTIDX=$3
NOIDVL=$4
HNDL=$5

OUTPREF=${HNDL}_Nb${NBTLNK}/${HNDL}_Nb${NBTLNK}

${GITDIR}/neu_sims.py $((LASTIDX+(SGE_TASK_ID-1)*RUNS+1)) $((LASTIDX+SGE_TASK_ID*RUNS+1)) $NOIDVL $NBTLNK $OUTPREF

echo "_EXITSTAT_$?"
echo "_END_$(date)"
exit
