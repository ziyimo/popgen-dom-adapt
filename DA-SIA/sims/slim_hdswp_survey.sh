#!/bin/bash
#$ -N SLiM_array
#$ -S /bin/bash
#$ -cwd
#$ -o $JOB_ID_$TASK_ID.o
#$ -e $JOB_ID_$TASK_ID.e
#$ -l m_mem_free=2G

## at submission
# -t 1-60
# -tc 30

echo "_START_$(date)"

# module load EBModules
# module load GCC/9.3.0
# module load OpenMPI/4.0.3-GCC-9.3.0
# module load GSL/2.6-GCC-9.3.0

SLIMDIR="/grid/siepel/home_norepl/mo/SLiM_3.7"
GITPATH="/grid/siepel/home_norepl/mo/dom_adapt/popgen-dom-adapt"

RUNS=$1 # no of new runs PER THREAD
MUTGENEL=$2 # ealiest gen (before present) to introduce mutation
MUTGENLT=$3 # latest gen (before present) to introduce mutation
SCMIN=$4 # min sel. coef
SCMAX=$5 # max sel. coef
SCRIPT=$6

for sim in $(seq 1 $RUNS); do
	while :
	do
		${SLIMDIR}/slim -s $(tr -cd "[:digit:]" < /dev/urandom | head -c 10) -t -m \
		-d "N=10000" -d "L=1e5" -d "G=1e4" -d "rho=1.25e-8" -d "mu=1.25e-8" \
		-d "mutgenbp_early=$MUTGENEL" -d "mutgenbp_late=$MUTGENLT" \
		-d "s_min=$SCMIN" -d "s_max=$SCMAX" \
		-d "outPref='dummy'" ${GITPATH}/DA-SIA/sims/$SCRIPT
		SLIM_RTCD=$?
		echo "${sim}_SLiM_EXITSTAT_${SLIM_RTCD}"
		if ((SLIM_RTCD == 0)); then break ; fi
	done
done

echo "_END_$(date)"
exit
