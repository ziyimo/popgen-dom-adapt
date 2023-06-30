#!/bin/bash
#$ -N SLiM_array
#$ -S /bin/bash
#$ -cwd
#$ -o logs/$JOB_ID_$TASK_ID.o
#$ -e logs/$JOB_ID_$TASK_ID.e
#$ -l m_mem_free=2G

## These should be passed in while submitting the job
# -t 1-100
# -tc 50

echo "_START_$(date)"

# module load GCC/9.3.0
# module load OpenMPI/4.0.3-GCC-9.3.0
# module load GSL/2.6-GCC-9.3.0

SLIMDIR="/grid/siepel/home_norepl/mo/SLiM_4"
GITDIR="/grid/siepel/home_norepl/mo/dom_adapt/popgen-dom-adapt/DA-SIA/sims/xtreme_sims"


NBTLNK=$1 # bottleneck size
RUNS=$2 # no of new runs PER THREAD
LASTIDX=$3
NOIDVL=$4
HNDL=$5
SC_MIN=$6 # 0.001
SC_MAX=$7 # 0.02

OUTPREF=${HNDL}_Nb${NBTLNK}/${HNDL}_Nb${NBTLNK}
SCRIPT="${GITDIR}/neu_xtreme_cont.slim"

for sim in $(seq 1 $RUNS); do
    RUN_ID=$((LASTIDX+(SGE_TASK_ID-1)*RUNS+sim))
    if [ -f ${OUTPREF}_${RUN_ID}_samp.trees ]; then
        echo "$RUN_ID was successful, SKIPPING"
        continue
    fi
    ATTEMPTS=0
    while :
    do
        ((ATTEMPTS=ATTEMPTS+1))
        if ((ATTEMPTS > 20)); then break ; fi

        ${GITDIR}/msprime_init.py n $SC_MIN $SC_MAX $NBTLNK ${OUTPREF}_${RUN_ID}
        INIT_RTCD=$?
        echo "${RUN_ID}_init_EXITSTAT_${INIT_RTCD}"
        if ((INIT_RTCD != 0)); then continue ; fi

        ${SLIMDIR}/slim -s $(tr -cd "[:digit:]" < /dev/urandom | head -c 10) -t -m \
        -d "pref='${OUTPREF}_${RUN_ID}'" $SCRIPT | tee logs/${JOB_ID}_${SGE_TASK_ID}.buf
        SLIM_RTCD=${PIPESTATUS[0]}
        echo "${RUN_ID}_SLiM_EXITSTAT_${SLIM_RTCD}"
        if ((SLIM_RTCD != 0)); then continue ; fi
        
        #/usr/bin/time -f "RSS=%M elapsed=%E"
        ${GITDIR}/samp_tree.py ${OUTPREF}_${RUN_ID} $NOIDVL n
        REC_RTCD=$?
        echo "${RUN_ID}_samp_EXITSTAT_${REC_RTCD}"
        if ((REC_RTCD != 0)); then continue ; fi

        grep "%%" logs/${JOB_ID}_${SGE_TASK_ID}.buf >> ${HNDL}_Nb${NBTLNK}_${SGE_TASK_ID}.meta
        echo "${RUN_ID}_SUCCESS" >> ${HNDL}_Nb${NBTLNK}_${SGE_TASK_ID}.0exit
        break
    done
    # delete the intermediate tree file (takes up too much space)
    rm ${OUTPREF}_${RUN_ID}_init*
    rm ${OUTPREF}_${RUN_ID}_slim.trees

done

rm logs/${JOB_ID}_${SGE_TASK_ID}.buf

echo "_END_$(date)"
exit
