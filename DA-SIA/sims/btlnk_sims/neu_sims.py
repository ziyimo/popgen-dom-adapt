#!/usr/bin/env python3

import sys, os
import msprime
import tskit
import numpy as np
#from tqdm import tqdm

helpMsg = '''
        usage: $./neu_sims.py <start_idx> <end_idx> <N_indvls> <N_btlnk> <out_pref>

'''

def main(args):
    if len(args) != 6:    # 5 arguments
        return helpMsg

    a_idx = int(args[1]) # inclusive
    b_idx = int(args[2]) # exclusive
    N = int(args[3])
    N_btnk = int(args[4])
    handle = args[5]

    scale = 1
    Ne = 10000
    #mu = 1e-9*scale
    rho = 1.25e-8*scale
    sim_len = 1e5
    t_end = 1000 # bottleneck end time
    t_start = 2000 # bottleneck start time    

    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=int(Ne/scale))
    demography.add_population_parameters_change(time=int(t_end/scale), population="A", initial_size=int(N_btnk/scale))
    demography.add_population_parameters_change(time=int(t_start/scale), population="A", initial_size=int(Ne/scale))

    print(f">>>Simulating: [{a_idx}, {b_idx})", flush=True)
    for sim_idx in range(a_idx, b_idx):

        if os.path.isfile(f"{handle}_{sim_idx}_samp.trees"):
            print(f">>>SKIPPED{sim_idx} - already exists", flush=True)
            continue

        ts_init = msprime.sim_ancestry(samples=N,
                                       demography=demography,
                                       sequence_length=sim_len,
                                       recombination_rate=rho)

        center_gen = ts_init.at(sim_len/2)
        subtree_size = np.empty((len(list(center_gen.nodes())), 2), dtype=int) # node_id | total leaves
        for idx, nd in enumerate(center_gen.nodes()):
            for samp_nd in center_gen.samples(nd):
                assert center_gen.population(samp_nd) == 0
            subtree_size[idx] = [nd, center_gen.num_samples(nd)] ## samples, not individuals

        intNode_AF = subtree_size[:,1]/(2*N)

        while True:
            try:
                AF_thr = np.random.uniform(0.25, 0.95)
                mut_nd_idx = np.abs(intNode_AF - AF_thr).argmin()

                AF_init = intNode_AF[mut_nd_idx]
                if AF_init == 1: continue
                mut_nd = subtree_size[mut_nd_idx, 0]

                init_tables = ts_init.dump_tables()
                init_tables.sites.add_row(position=sim_len/2, ancestral_state="A")
                init_tables.mutations.add_row(site=0, node=mut_nd, derived_state="T", time=int(np.ceil(ts_init.node(mut_nd).time)))

                ts = init_tables.tree_sequence() # ts with mutation overlaid

            except:
                print("ts conversion ERROR!")
            else:
                print("Mutation @ node:", ts_init.node(mut_nd))
                print(subtree_size[mut_nd_idx])
                break

        assert max(t.num_roots for t in ts.trees()) == 1
        print(len(ts.samples()))

        ts.dump(f"{handle}_{sim_idx}_samp.trees")
        print(f">>>SIM{sim_idx} - Region_len:{sim_len}\tAF:{AF_init}", flush=True)

    return 0

sys.exit(main(sys.argv))