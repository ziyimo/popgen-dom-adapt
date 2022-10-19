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
        usage: $./ts2feature.py <`n`/`s`> <no_sims/meta_file_path> <thr> <tot_thr> <in_pref> <tree_type> <outPref> <no_taxa>
        meta_file is .npy file
        <in_pref> should be the complete prefix, including directory and file name prefix
        <tree_type> example: `tru.trees`, `inf.trees` or `inf.trees.tsz`
        (e.g. SLiM_trial_neu_ts/SLiM_trial_neu_ts)
'''

def main(args):
    if len(args) != 9:    #8 arguments
        return helpMsg

    mode = args[1]
    if mode == 'n':
        no_sims = int(args[2])
    elif mode == 's':
        metaF = args[2]
    else:
        return helpMsg

    thr = int(args[3]) # should be 1-indexed
    tot_thr = int(args[4])
    inPref = args[5]
    tree_type = args[6]
    outPref = args[7]
    no_taxa = int(args[8])

    if mode == 's':
        meta_sorted = np.load(metaF) # new npy format
        idx_ls = meta_sorted[:, 0]
        sc_ls = meta_sorted[:, 1]
        onset_ls = meta_sorted[:, 2]
        caf_ls = meta_sorted[:, 3]
        no_sims = meta_sorted.shape[0]

    tasks = no_sims//tot_thr
    a_idx = (thr-1)*tasks # inclusive
    if thr == tot_thr:
        b_idx = no_sims # exclusive
    else:
        b_idx = thr*tasks # exclusive
    # indices are 0-based

    print("Processing: [", a_idx, b_idx, ")", flush=True)

    fea_df = np.empty((0, 3, no_taxa-1, no_taxa-1), dtype=np.int32) # dataframe containing the features
    meta_df = np.empty((0, 5)) # dataframe containing the meta-data [idx, sc, onset, AF, var_pos]

    for r_idx in range(a_idx, b_idx):
        if mode == 'n':
            ID = r_idx + 1 # convert 0-based index to 1-based index
        elif mode == 's':
            ID = int(idx_ls[r_idx]) # retrieve 1-based index from metadata
        ts_path = inPref+"_"+str(ID)+"_"+tree_type
        if not os.path.isfile(ts_path): continue

        if tree_type[-3:] == "tsz":
            ts_eg = tszip.decompress(ts_path)
        else:
            ts_eg = tskit.load(ts_path)

        GTM = ts_eg.genotype_matrix()
        var_pos = get_site_ppos(ts_eg)

        if mode == 'n':
            samp_idx = samp_var(GTM, var_pos)
            vOI_gt = GTM[samp_idx, :].flatten()
            vOI_pos = var_pos[samp_idx]
            sc = 0
            onset = 0
            AF = np.mean(vOI_gt)
        elif mode == 's':
            vOI_gt = GTM[var_pos==50000, :].flatten()
            if np.sum(vOI_gt) == 0: # sweep site monomorphic, possibly due to Relate artifact
                print(">>>", r_idx, "/", b_idx, ts_path, ": MONO", flush=True)
                continue # discard simulation
            vOI_pos = 50000
            sc = sc_ls[r_idx]
            onset = onset_ls[r_idx]
            AF = caf_ls[r_idx]

        print(">>>", r_idx, "/", b_idx, ts_path, ":", vOI_pos, flush=True)
        fea_mtx = gen2feature(ts_eg, vOI_pos, vOI_gt, no_taxa)

        fea_df = np.concatenate((fea_df, np.array([fea_mtx], dtype=np.int32)))
        meta_df = np.vstack((meta_df, [ID, sc, onset, AF, vOI_pos]))

    print(fea_df.shape, meta_df.shape)
    np.save(outPref+"_fea_"+str(thr), fea_df)
    np.save(outPref+"_meta_"+str(thr), meta_df)

    return 0

sys.exit(main(sys.argv))
s