#!/usr/bin/env python3

import sys, os
#sys.path.append(os.path.join(os.path.dirname(__file__), "../..")) # directory of module fea_util
from contextlib import contextmanager

import msprime, tskit
import numpy as np
import tszip

from RELATE_util import run_RELATE
from utils import get_site_ppos

helpMsg = '''
        usage: $./slimSims2ts.py <Ne> <scale> <max_no_sims> <thr> <tot_thr> <inPref> <outPref>
            Take slim simulation output, run RELATE and save both true and inferred tree sequence file for each sim
            <inPref> and <outPref> should be directory names!
'''


def sim2ts(ts_mut, mu, rho_cMpMb, N0):

    ppos_ls = get_site_ppos(ts_mut)
    GTM = ts_mut.genotype_matrix()
    
    mask = (ppos_ls != -1)
    ppos_ls = ppos_ls[mask]
    GTM = GTM[mask, :]

    ts_inf = run_RELATE(ppos_ls, GTM, str(2*N0), rho_cMpMb=rho_cMpMb, mut_rate=str(mu))

    return ts_inf # inferred tree-seqs

# @contextmanager is just an easier way of saying cd = contextmanager(cd)
@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

def main(args):
    if len(args) != 8:    #7 arguments
        return helpMsg

    parent_dir = os.getcwd() # directory where the job is submitted

    scaled_mu = 1.25e-8
    scaled_rho = 1.25e-8
    scaled_N0 = int(args[1])

    scale = int(args[2])
    no_sims = int(args[3])
    thr = int(args[4]) # should be 1-indexed
    tot_thr = int(args[5])
    inPref = args[6]
    outPref = args[7]

    mu = scaled_mu/scale
    rho_cMpMb = scaled_rho/scale*100*1e6 # 1cM = 1e-2 crossover
    N0 = scaled_N0*scale

    tasks = no_sims//tot_thr
    a_idx = (thr-1)*tasks # inclusive
    if thr == tot_thr:
        b_idx = no_sims # exclusive
    else:
        b_idx = thr*tasks # exclusive
    # indices are 0-based

    print("Processing: [", a_idx, b_idx, ")", flush=True)

    wd = outPref+'/'+'RELATE_temp_'+str(thr)
    os.mkdir(wd, 0o755)

    log_f = open(outPref+"_"+str(thr)+".log", 'a')
    with cd(wd):
        for r_idx in range(a_idx, b_idx):

            ID = r_idx + 1 # convert 0-based index to 1-based index
            sim_path = parent_dir+"/"+inPref+"/"+inPref+"_"+str(ID)+"_samp.trees"

            if not os.path.isfile(sim_path):
                print(ID, "SKIPPED", file=log_f, flush=True)
                continue

            outFP = parent_dir+"/"+outPref+"/"+outPref+"_"+str(ID)
            if os.path.isfile(outFP+"_tru.trees") and os.path.isfile(outFP+"_inf.trees.tsz"):
                print(ID, "EXISTED", file=log_f, flush=True)
                continue

            print("Input:", sim_path, flush=True)
            ts_samp = tskit.load(sim_path)
            if ts_samp.num_sites > 1:
                ts_samp = ts_samp.delete_sites(np.nonzero(ts_samp.tables.sites.position != 50000)[0])
            ts_tru = msprime.mutate(ts_samp, rate=scaled_mu, keep=True) # deprecated, but supported indefinitely
            # while True:
            #     try:
            #         ts_tru = msprime.sim_mutations(ts_samp, rate=scaled_mu, keep=True)
            #         break
            #     except msprime._msprime.LibraryError:
            #         # see: https://github.com/tskit-dev/msprime/discussions/2089
            #         print("Mutations overlap, try again.", flush=True)
            #         continue

            ts_inf = sim2ts(ts_tru, mu, rho_cMpMb, N0)

            if ts_inf is None:
                print(ID, "FAILED", file=log_f, flush=True) # after 20 Relate attempts
            else:
                ts_tru.dump(outFP+"_tru.trees")
                tszip.compress(ts_inf, outFP+"_inf.trees.tsz")
                print(ID, "SUCCESS", file=log_f, flush=True)

    log_f.close()
    os.rmdir(wd)

    return 0

sys.exit(main(sys.argv))
