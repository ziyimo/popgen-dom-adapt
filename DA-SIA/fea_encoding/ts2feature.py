#!/usr/bin/env python3

import sys, os
#sys.path.append(os.path.join(os.path.dirname(__file__), "../..")) # directory of module utils

import numpy as np
import dendropy
import tskit
import tszip

from utils import *

# MACRO: simulated region [0, 1e5)
#NRS = 1 # of genealogies represented in the feature
# N.b. use only center genealogy for prototyping

helpMsg = '''
        usage: $./ts2feature.py <no_sims> <thr> <tot_thr> <in_pref> <tree_type> <outPref> <no_taxa>
        meta_file is .npz file
        <in_pref> should be the complete prefix, including directory and file name prefix
        <tree_type> example: `tru.trees`, `inf.trees` or `inf.trees.tsz`
        (e.g. SLiM_trial_neu_ts/SLiM_trial_neu_ts)
'''

def main(args):
    if len(args) != 8:    #7 arguments
        return helpMsg

    ## new simulations: neutral sim also have a variant at center

    no_sims = int(args[1])
    thr = int(args[2]) # should be 1-indexed
    tot_thr = int(args[3])
    inPref = args[4]
    tree_type = args[5]
    outPref = args[6]
    no_taxa = int(args[7])

    ## worry about meta later, only do classification for now

    tasks = no_sims//tot_thr
    a_idx = (thr-1)*tasks # inclusive
    if thr == tot_thr:
        b_idx = no_sims # exclusive
    else:
        b_idx = thr*tasks # exclusive
    # indices are 0-based

    print("Processing: [", a_idx, b_idx, ")", flush=True)

    fea_df = np.empty((0, 3, no_taxa-1, no_taxa-1), dtype=int) # dataframe containing the features
    idx_arr = np.empty(0, dtype=int) # just save the index of the feature to look up meta

    for r_idx in range(a_idx, b_idx):
        ID = r_idx + 1 # convert 0-based index to 1-based index
        ts_path = inPref+"_"+str(ID)+"_"+tree_type
        if not os.path.isfile(ts_path): continue

        if tree_type[-3:] == "tsz":
            ts_eg = tszip.decompress(ts_path)
        else:
            ts_eg = tskit.load(ts_path)

        GTM = ts_eg.genotype_matrix()
        var_pos = get_site_ppos(ts_eg)

        vOI_gt = GTM[var_pos==50000, :].flatten()
        if np.sum(vOI_gt) == 0: # sweep site monomorphic, possibly due to Relate artifact
            print(">>>", r_idx, "/", b_idx, ts_path, ": MONO", flush=True)
            continue # discard simulation
        vOI_pos = 50000

        print(">>>", r_idx, "/", b_idx, ts_path, ":", vOI_pos, flush=True)
        fea_mtx = gen2feature(ts_eg, vOI_pos, vOI_gt, no_taxa)

        fea_df = np.concatenate((fea_df, np.array([fea_mtx])))
        idx_arr = np.append(idx_arr, ID)

    print(fea_df.shape, idx_arr.shape)
    np.save(outPref+"_fea_"+str(thr), fea_df)
    np.save(outPref+"_idx_"+str(thr), idx_arr)

    return 0

sys.exit(main(sys.argv))
