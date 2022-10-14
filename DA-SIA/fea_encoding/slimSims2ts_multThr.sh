#!/bin/bash
#$ -N slim2ts_arr
#$ -S /bin/bash
#$ -cwd
#$ -o logs/$JOB_ID_$TASK_ID.o
#$ -e logs/$JOB_ID_$TASK_ID.e
#$ -l virtual_free=25G

## Specify with qsub
# -t 1-50
# -tc 50

echo "_START_$(date)"

# also remember to load R ``module load R/4.0.5-foss-2020a``

SCRIPT_DIR="/grid/siepel/home_norepl/mo/dom_adapt/popgen-dom-adapt/DA-SIA/fea_encoding"

NE=$1
NOSIMS=$2
TOTTHR=$3 # makes patching easier
INPREF=$4
OUTPREF=$5

SCALE=1
THR=$SGE_TASK_ID

${SCRIPT_DIR}/slimSims2ts.py $NE $SCALE $NOSIMS $THR $TOTTHR $INPREF $OUTPREF
echo "_EXITSTAT_$?"

echo "_END_$(date)"
exit
