#!/usr/bin/env python3

import sys
import subprocess
import numpy as np
import tskit

#RELATE_PATH = '/grid/siepel/home_norepl/mo/relate_v1.1.2_x86_64_static/'
RELATE_PATH = '/grid/siepel/home_norepl/mo/relate_v1.0.17_x86_64_static/'
ii32MAX = np.iinfo(np.int32).max

def run_RELATE(pp, gtm, Ne, rho_cMpMb=1.25, mut_rate="2.5e-8"):
    '''pp- physical position; gtm: genotype matrix; var_ppos: physical position of the variant
        var_ppos = -1 : default to skip RELATE selection inference, returns pval=0.1 << deprecated
        rho_cMpMb, mut_rate: default to human
    '''
    # create RELATE input files

    with open("temp.haps", 'w') as hapF:
        for i in range (len(pp)):
            print(str(1), "SNP"+str(i+1), int(pp[i]), "A", "T", *gtm[i], sep=" ", file=hapF)
    # Unpacking Argument Lists: *args

    no_dips = round(gtm.shape[1]/2)
    with open("temp.sample", 'w') as sampF:
        print('ID_1', 'ID_2', 'missing', file=sampF)
        print('0', '0', '0', file=sampF)
        for i in range(no_dips):
            print('UNR'+str(i+1), 'UNR'+str(i+1), 0, file=sampF)

    with open("temp.poplabels", 'w') as popF:
        print('sample', 'population', 'group', 'sex', file=popF)
        for i in range(no_dips):
            print('UNR'+str(i+1), 'CEU', 'EUR', 1, file=popF)

    with open("temp.map", 'w') as mapF:
        ppos = 0
        rdist = 0
        print('pos', 'COMBINED_rate', 'Genetic_map', file=mapF)
        print(int(ppos), rho_cMpMb, rdist, file=mapF)
        
        for loc in pp:
            next_ppos = loc
            rdist = rdist + rho_cMpMb/1e6*(next_ppos - ppos)
            ppos = next_ppos
            print(int(ppos), rho_cMpMb, rdist, file=mapF)

    relate_cmd = [RELATE_PATH+"bin/Relate", "--mode", "All",
            "-m", mut_rate,
            "-N", Ne,
            #"--coal", "/grid/siepel/home_norepl/mo/arg-selection/GBR_discoal_dem.coal", # supply actual demography, overwrites -N
            "--haps", "temp.haps",
            "--sample", "temp.sample",
            "--map", "temp.map",
            "-o", "temp",
            #"--memory", "8", # default is 5 GB
            "--seed", "1"]

    popsize_cmd = [RELATE_PATH+"scripts/EstimatePopulationSize/EstimatePopulationSize.sh",
            "-i", "temp",
            "-m", mut_rate,
            "--poplabels", "temp.poplabels",
            "--threshold", "10",
            #"--num_iter", "10", # default is 5
            "-o", "temp_popsize",
            "--seed", "1"]

    wg_cmd = [RELATE_PATH+"scripts/EstimatePopulationSize/EstimatePopulationSize.sh",
            "-i", "temp",
            "-m", mut_rate,
            "--poplabels", "temp.poplabels",
            "--threshold", "0",
            "--coal", "temp_popsize.coal",
            "--num_iter", "1",
            "-o", "temp_wg"]

    conv_cmd = [RELATE_PATH+"bin/RelateFileFormats", "--mode", "ConvertToTreeSequence",
            "-i", "temp_wg",
            "-o", "temp_wg"]

    #clean_cmd = [RELATE_PATH+"bin/Relate", "--mode", "Clean", "-o", "temp"]

    loop_cnt = 0
    # run RELATE
    while True:
        loop_cnt += 1
        if loop_cnt > 20:
            print("Milestone: attempt exceeds 20, ABORTS!!", file=sys.stderr, flush=True)
            subprocess.call("rm -r temp*", shell=True)
            return None

        print("Milestone: Running RELATE pipeline, try_", loop_cnt, sep='', flush=True)
        relate_cmd[-1] = str(np.random.randint(ii32MAX))
        popsize_cmd[-1] = str(np.random.randint(ii32MAX))
        print(" ".join(relate_cmd), flush=True)
        relate_proc = subprocess.run(relate_cmd)
        if relate_proc.returncode != 0: continue

        # re-estimate branch lengths
        popsize_proc = subprocess.run(popsize_cmd)
        if popsize_proc.returncode != 0: continue
        # output: _popsize.pdf, _popsize.anc.gz, _popsize.mut.gz, _popsize.dist, _popsize.coal. _popsize.bin, _popsize_avg.rate

        # re-estimate branch length for ENTIRE genealogy
        wg_proc = subprocess.run(wg_cmd)
        if wg_proc.returncode != 0: continue

        # convert to tree-sequence
        conv_proc = subprocess.run(conv_cmd)
        if conv_proc.returncode != 0: continue
        # if the conversion throws <time[parent] must be greater than time[child]> error, rerun the last resampling step
        print("Milestone: RELATE pipeline success", flush=True)
        break

    # load tree sequence
    ts_inferred = tskit.load("temp_wg.trees")

    # clean-up
    subprocess.call("rm -r temp*", shell=True)

    return ts_inferred