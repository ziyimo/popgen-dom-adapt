#!/bin/bash
#$ -N SLiM_array
#$ -S /bin/bash
#$ -cwd
#$ -o survey_logs/$JOB_ID_$TASK_ID.o
#$ -e survey_logs/$JOB_ID_$TASK_ID.e
#$ -l m_mem_free=2G

## at submission
# -t 1-60
# -tc 30

echo "_START_$(date)"

# module load GCC/9.3.0
# module load OpenMPI/4.0.3-GCC-9.3.0
# module load GSL/2.6-GCC-9.3.0

SLIMDIR="/grid/siepel/home_norepl/mo/SLiM_4"
GITDIR="/grid/siepel/home_norepl/mo/dom_adapt/popgen-dom-adapt/DA-SIA/sims/btlnk_sims"

NBTLNK=$1 # bottleneck size
RUNS=$2 # no of new runs PER THREAD

OUTPREF="tmp/sft_Nb${NBTLNK}_${SGE_TASK_ID}"

for sim in $(seq 1 $RUNS); do
    ${GITDIR}/sft_init.py 0.001 0.02 -1 -1 $NBTLNK $OUTPREF
    INIT_RTCD=$?
    echo "_init_EXITSTAT_${INIT_RTCD}"
    ${SLIMDIR}/slim -s $(tr -cd "[:digit:]" < /dev/urandom | head -c 10) -t -m \
    -d "pref='${OUTPREF}'" "${GITDIR}/sft_btlnk_survey.slim"
    SLIM_RTCD=$?
    echo "_SLiM_EXITSTAT_${SLIM_RTCD}"

    rm ${OUTPREF}_init*
done

echo "_END_$(date)"
exit
