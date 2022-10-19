#!/bin/bash
#$ -N eval_mod
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l m_mem_free=100G
#$ -l virtual_free=150G
#$ -l gpu=2

# module purge
module load EBModules
module load OpenMPI/4.0.5-gcccuda-2020b
module load TensorFlow/2.4.1-fosscuda-2020b

CODEPATH="/grid/siepel/home_norepl/mo/dom_adapt/DL_archs"
DATAPATH="/grid/siepel/home_norepl/mo/dom_adapt/sims_SIA/fea_ds"

echo "_START_$(date)"

${CODEPATH}/$1 $2 ${DATAPATH}/$3 ${DATAPATH}/$4 benchmark/$5 # for sc mods
echo "_EXITSTAT_$?"

echo "_END_$(date)"
exit
