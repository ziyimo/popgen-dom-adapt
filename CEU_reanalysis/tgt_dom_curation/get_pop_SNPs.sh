#!/bin/bash

echo "_START_$(date)"

POPFILE=$1 # has to be local, i.e. in the current working directory
CHR=$2 # note that non autosomes have different file naming from autosomes
OUTDIR=$3

DATAPATH='/usr/data/1000G'

# --keep <filename>
# Provide files containing a list of individuals to either include or exclude in subsequent analysis. Each individual ID (as defined in the VCF headerline) should be included on a separate line. No header line is expected.

vcftools --vcf ${DATAPATH}/ALL.chr${CHR}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf --keep $POPFILE --remove-indels --min-alleles 2 --max-alleles 2 --mac 1 --recode --recode-INFO AA --out ${OUTDIR}/${POPFILE}_chr${CHR}_gt
echo "chr${CHR}_RECODE_EXITSTAT_$?"
# Maybe keep the hotspot variants for ARG inference purpose  --bed wg_mask.bed
# Downstream filter: 1) pass global mask, 2) minimum # of variants (survey histogram to determine threshold)

bgzip ${OUTDIR}/${POPFILE}_chr${CHR}_gt.recode.vcf
echo "chr${CHR}_BGZIP_EXITSTAT_$?"

tabix -p vcf ${OUTDIR}/${POPFILE}_chr${CHR}_gt.recode.vcf.gz
echo "chr${CHR}_TABIX_EXITSTAT_$?"

echo "_END_$(date)"