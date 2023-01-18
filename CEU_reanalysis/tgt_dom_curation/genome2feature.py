#!/usr/bin/env python3

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../DA-SIA/fea_encoding")) # directory of module utils

import re
import numpy as np
import dendropy
import tskit
from tqdm import trange

from utils import encode

helpMsg = '''
        usage: $./genome2feature.py <ts_path> <outPref>
'''

def main(args):
    if len(args) != 3:    #2 arguments
        return helpMsg

    ts_path = args[1]
    outPref = args[2]

    chrom = int(re.split(r'[r_]', outPref)[-3])

    ts = tskit.load(ts_path)
    
    no_trs = ts.num_trees
    no_taxa = ts.num_samples
    no_sites = ts.num_sites

    pos_ls = ts.sites_position.astype(int)
    GTM = ts.genotype_matrix()
    assert len(pos_ls) == no_sites
    assert GTM.shape == (no_sites, no_taxa)

    DAF = GTM.sum(axis=1)/no_taxa
    mask = np.logical_and(DAF > 0.05, DAF < 0.95)
    cand_idx = mask.nonzero()[0]
    pass_sites = len(cand_idx)
    assert mask.sum() == pass_sites

    no_samps = np.minimum(1000, pass_sites)
    samp_idx = np.random.choice(cand_idx, no_samps, replace=False)
    samp_idx.sort()

    fea_df = np.empty((no_samps, 3, no_taxa-1, no_taxa-1), dtype=np.int32) # dataframe containing the features
    samp_pos = pos_ls[samp_idx]
    samp_gt = GTM[samp_idx]

    for i in trange(no_samps):
        focal_gen = ts.at(samp_pos[i])
        var_der = list(map(str, np.nonzero(samp_gt[i])[0] + 1))
        
        F, W, R = encode(focal_gen.newick(precision=5), no_taxa, var_der)

        fea_df[i] = np.stack((F.T+np.tril(F, -1), W.T+np.tril(W, -1), R.T+np.tril(R, -1)))

    np.save(outPref+"_fea", fea_df)
    np.save(outPref+"_gt", samp_gt)
    np.save(outPref+"_pos", np.vstack((chrom*np.ones(no_samps, dtype=int), samp_pos)))

    print("#chrom\ttaxa\ttrees\tsites\tpassed_sites\tsampled_sites")
    print(f"##{chrom}\t{no_taxa}\t{no_trs}\t{no_sites}\t{pass_sites}\t{no_samps}", flush=True)

    return 0

sys.exit(main(sys.argv))