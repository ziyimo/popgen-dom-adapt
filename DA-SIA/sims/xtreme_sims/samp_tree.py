#!/usr/bin/env python3

import sys, os
import tskit
import pyslim
import numpy as np

helpMsg = '''
        usage: $./samp_tree.py <pref> <N_chrs> <`n` or `s`>

        - samples from the tree-sequence output from SLiM simulations
'''

def main(args):
    if len(args) != 4:    #3 arguments
        return helpMsg

    pref = args[1]
    #scale = int(args[2])
    N = int(args[2])
    #out_path = args[5]+".trees"
    mode = args[3]

    #mu = 1e-9*scale
    #rho = 1e-8*scale

    ts = tskit.load(f"{pref}_slim.trees")
    assert max(t.num_roots for t in ts.trees()) == 1
    sim_len = ts.sequence_length

    if mode == 'n':
        center_gen = ts.at(sim_len/2)
        subtree_size = np.empty((len(list(center_gen.nodes())), 2), dtype=int) # node_id | total leaves
        for idx, nd in enumerate(center_gen.nodes()):
            for samp_nd in center_gen.samples(nd):
                assert center_gen.population(samp_nd) == 0
            subtree_size[idx] = [nd, center_gen.num_samples(nd)] ## samples, not individuals

        intNode_AF = subtree_size[:,1]/ts.num_samples

        while True:
            try:
                AF_thr = np.random.uniform(0.25, 0.95)
                mut_nd_idx = np.abs(intNode_AF - AF_thr).argmin()

                AF_init = intNode_AF[mut_nd_idx]
                if AF_init == 1: continue
                mut_nd = subtree_size[mut_nd_idx, 0]

                init_tables = ts.dump_tables()
                init_tables.sites.clear()
                init_tables.mutations.clear()

                mut_site_id = init_tables.sites.add_row(position=sim_len/2, ancestral_state="A")
                init_tables.mutations.add_row(site=mut_site_id, node=mut_nd, derived_state="T", time=int(np.ceil(ts.node(mut_nd).time)), metadata={'mutation_list':[pyslim.default_slim_metadata("mutation_list_entry")]})

                ts = init_tables.tree_sequence() # ts with mutation overlaid

            except tskit.LibraryError:
                print("ts conversion ERROR!")
            else:
                print("Neu mutation @ node:", ts.node(mut_nd))
                print(subtree_size[mut_nd_idx])
                break


    #rts = ts.recapitate(recombination_rate=rho, Ne=N_anc/scale)

    sampN = np.random.choice(ts.samples(), size=N, replace=False)
    sts = ts.simplify(samples=sampN)

    print(f"Simplify: {ts.num_samples}/{ts.num_individuals} -> {sts.num_samples}/{sts.num_individuals}")

    sts.dump(f"{pref}_samp.trees")

    return 0

sys.exit(main(sys.argv))