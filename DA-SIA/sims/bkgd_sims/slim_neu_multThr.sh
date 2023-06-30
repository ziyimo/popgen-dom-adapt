#!/bin/bash
#$ -N SLiM_array
#$ -S /bin/bash
#$ -cwd
#$ -o logs/$JOB_ID_$TASK_ID.o
#$ -e logs/$JOB_ID_$TASK_ID.e
#$ -l m_mem_free=2G
#$ -l virtual_free=16G

## These should be passed in while submitting the job
# -t 1-100
# -tc 50

echo "_START_$(date)"

SLIMDIR="/grid/siepel/home_norepl/mo/SLiM_3.7"
GITPATH="/grid/siepel/home_norepl/mo/dom_adapt/popgen-dom-adapt"

SCRIPT=$1
NOCHR=$2
HNDL=$3
OUTPREF=${HNDL}/${HNDL}
LASTIDX=$4
RUNS=$5 # no of new runs PER THREAD

for sim in $(seq 1 $RUNS); do
    RUN_ID=$((LASTIDX+(SGE_TASK_ID-1)*RUNS+sim))
    if [ -f ${OUTPREF}_${RUN_ID}_samp.trees ]; then
        echo "$RUN_ID was successful, SKIPPING"
        continue
    fi
    while :
    do
		${SLIMDIR}/slim -s $(tr -cd "[:digit:]" < /dev/urandom | head -c 10) \
		-d "N=10000" -d "L=1e5" -d "G=1e4" -d "rho=1.25e-8" -d "mu=1.25e-8" \
        -d "outPref='${OUTPREF}_${RUN_ID}_temp'" \
        ${GITPATH}/DA-SIA/sims/$SCRIPT

        SLIM_RTCD=$?
        echo "${RUN_ID}_SLiM_EXITSTAT_${SLIM_RTCD}"
        if ((SLIM_RTCD != 0)); then continue ; fi
        #if ((SLIM_RTCD == 0)); then break ; fi
        ATTEMPTS=0
        while :
        do
            #/usr/bin/time -f "RSS=%M elapsed=%E"
            ${GITPATH}/DA-SIA/sims/recap.py --N $NOCHR ${OUTPREF}_${RUN_ID}_temp.trees ${OUTPREF}_${RUN_ID}_samp.trees
            REC_RTCD=$?
            echo "${RUN_ID}_recap_EXITSTAT_${REC_RTCD}"
            if ((REC_RTCD == 0)); then
                echo "${RUN_ID}_SUCCESS" >> ${HNDL}_${SGE_TASK_ID}.0exit
                break 2
            fi
            echo "Recap attempt:${ATTEMPTS} FAILED"
            ((ATTEMPTS=ATTEMPTS+1))
            if ((ATTEMPTS > 20)); then break ; fi
        done
    done
    # delete the intermediate tree file (takes up too much space)
    rm ${OUTPREF}_${RUN_ID}_temp.trees
done

echo "_END_$(date)"
exit
