#!/usr/bin/env python3

import sys, os
import tskit
import pyslim
import numpy as np

helpMsg = '''
        usage: $./samp_tree.py <pref> <N_chrs>

        - samples from the tree-sequence output from SLiM simulations
'''

def main(args):
    if len(args) != 3:    #2 arguments
        return helpMsg

    pref = args[1]
    #scale = int(args[2])
    N = int(args[2])
    #out_path = args[5]+".trees"

    #mu = 1e-9*scale
    #rho = 1e-8*scale

    ts = tskit.load(f"{pref}_slim.trees")
    #rts = ts.recapitate(recombination_rate=rho, Ne=N_anc/scale)
    print(f"Max roots: {max(t.num_roots for t in ts.trees())}")

    sampN = np.random.choice(ts.samples(), size=N, replace=False)
    sts = ts.simplify(samples=sampN)

    print(f"Simplify: {ts.num_samples}/{ts.num_individuals} -> {sts.num_samples}/{sts.num_individuals}")

    sts.dump(f"{pref}_samp.trees")

    return 0

sys.exit(main(sys.argv))