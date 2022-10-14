#!/bin/bash
#$ -N ts2Feature_arr
#$ -S /bin/bash
#$ -cwd
#$ -o logs/$JOB_ID_$TASK_ID.o
#$ -e logs/$JOB_ID_$TASK_ID.e
#$ -l virtual_free=16G

## Specify at submission
# -t 1-200
# -tc 100

echo "_START_$(date)"

# module load GCC/9.3.0
# module load OpenMPI/4.0.3-GCC-9.3.0
# module load GSL/2.6-GCC-9.3.0

SCRIPT_DIR="/grid/siepel/home_norepl/mo/dom_adapt/fea_encoding"

THR=$SGE_TASK_ID

NOSIMS=$1
TOTTHR=$2

INPREF=$3 # complete prefix, including directory and file name prefix
TTYPE=$4 # example: `tru.trees`, `inf.trees` or `inf.trees.tsz`
OUTPREF=$5
TAXA=128

if [[ -f ${OUTPREF}_fea_${THR}.npy && -f ${OUTPREF}_idx_${THR}.npy ]]; then
	echo "THR${THR} COMPLETED; SKIPPING"
else
	echo "THR${THR} MISSING; RUNNING"
	## NORMAL RUN ###
	${SCRIPT_DIR}/ts2feature.py $NOSIMS $THR $TOTTHR $INPREF $TTYPE $OUTPREF $TAXA
	echo "_EXITSTAT_$?"
	#################
fi

echo "_END_$(date)"
exit
