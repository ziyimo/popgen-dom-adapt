#!/bin/bash
#$ -N ts2Feature_arr
#$ -S /bin/bash
#$ -cwd
#$ -o logs/$JOB_ID_$TASK_ID.o
#$ -e logs/$JOB_ID_$TASK_ID.e
#$ -l virtual_free=32G

## Specify at submission
# -t 1-200
# -tc 100

echo "_START_$(date)"

SCRIPT_DIR="/grid/siepel/home_norepl/mo/dom_adapt/popgen-dom-adapt/DA-SIA/fea_encoding"

THR=$SGE_TASK_ID
TOTTHR=$1

MODE=$2 # <`n`/`s`>
META=$3 # <no_sims/meta_file_path>

INPREF=$4 # complete prefix, including directory and file name prefix
TTYPE=$5 # example: `tru.trees`, `inf.trees` or `inf.trees.tsz`
OUTPREF=$6
TAXA=128

if [[ -f ${OUTPREF}_fea_${THR}.npy && -f ${OUTPREF}_meta_${THR}.npy ]]; then
	echo "THR${THR} COMPLETED; SKIPPING"
else
	echo "THR${THR} MISSING; RUNNING"
	## NORMAL RUN ###
	${SCRIPT_DIR}/ts2feature.py $MODE $META $THR $TOTTHR $INPREF $TTYPE $OUTPREF $TAXA
	echo "_EXITSTAT_$?"
	#################
fi

echo "_END_$(date)"
exit
