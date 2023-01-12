#!/bin/bash
#$ -N RELATE_array
#$ -S /bin/bash
#$ -cwd
#$ -o $JOB_ID_$TASK_ID.o
#$ -e $JOB_ID_$TASK_ID.e
#$ -l m_mem_free=8G

## Specify at submit time, match # of line in WINS file
# -t 1-16

echo "_START_$(date)"

#GITPATH='/grid/siepel/home_norepl/mo'
RELATE_PATH="/grid/siepel/home_norepl/mo/relate_v1.0.17_x86_64_static"

mapfile -t WIN_LS < $1
IDX=$((SGE_TASK_ID-1))
WIN=${WIN_LS[$IDX]}

cd $WIN
echo Running Relate on $WIN

${RELATE_PATH}/bin/Relate \
	--mode All \
	-m 2.5e-8 \
	-N 376176 \
	--haps ${WIN}.haps \
	--sample ${WIN}.sample \
	--map ${WIN}.map \
	-o ${WIN}
RELATE_EXIT=$?

if [$RELATE_EXIT -ne 0]; then
    echo "${WIN}_RELATE_EXITSTAT_${RELATE_EXIT}_END_$(date)"
    exit
fi

# re-estimate branch lengths
${RELATE_PATH}/scripts/EstimatePopulationSize/EstimatePopulationSize.sh \
            -i ${WIN} \
            -m 2.5e-8 \
            --poplabels ${WIN}.poplabels \
            --threshold 10 \
            -o ${WIN}_popsize
POPSIZE_EXITSTAT=$?

if [$POPSIZE_EXITSTAT -ne 0]; then
    echo "${WIN}_POPSIZE_EXITSTAT_${POPSIZE_EXITSTAT}_END_$(date)"
    exit
fi

# re-estimate branch length for ENTIRE genealogy
${RELATE_PATH}/scripts/EstimatePopulationSize/EstimatePopulationSize.sh \
            -i ${WIN} \
            -m 2.5e-8 \
            --poplabels ${WIN}.poplabels \
            --threshold 0 \
            --coal ${WIN}_popsize.coal \
            --num_iter 1 \
            -o ${WIN}_wg
WG_EXITSTAT=$?

if [$WG_EXITSTAT -ne 0]; then
    echo "${WIN}_WG_EXITSTAT_${WG_EXITSTAT}_END_$(date)"
    exit
fi

${RELATE_PATH}/bin/RelateFileFormats \
	--mode ConvertToTreeSequence \
	-i ${WIN}_wg \
	-o ${WIN}_wg
TS_CONVERSION_EXITSTAT=$?

if [$TS_CONVERSION_EXITSTAT -ne 0]; then
    echo "${WIN}_TS_CONVERSION_EXITSTAT_${TS_CONVERSION_EXITSTAT}_END_$(date)"
    exit
fi

echo "${WIN}_SUCCESS_END_$(date)"
exit
