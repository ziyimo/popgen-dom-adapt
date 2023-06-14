#!/usr/bin/env python3

import sys, os
import msprime
import pyslim
import tskit
import numpy as np

helpMsg = '''
        usage: $./sft_init.py <sc_min> <sc_max> <genbp_min> <genbp_max> <Ne_bottleneck> <out_pref>

        Put genbp minmax as `-1` to dynamically tune mutgen
'''

scale = 1
sim_len = 5e3 # can be shortened for surveys
rho = 1e-8*scale
Ne = 10000

def main(args):
    if len(args) != 7:    # 6 arguments
        return helpMsg

    sc_min = float(args[1])
    sc_max = float(args[2])
    rec = int(args[3])
    anc = int(args[4])
    if rec == -1 and anc == -1: # TBD
        rec = (10**0.7)*(SC**(-1))
        anc = (10**1.3)*(SC**(-1))

    t_end = 1000 # bottleneck end time
    t_start = 2000 # bottleneck start time
    N_btnk = int(args[5])

    pref = args[6]

    logunif = bool(np.random.randint(2))
    if logunif:
        SC = np.exp(np.random.uniform(np.log(sc_min*scale), np.log(sc_max*scale)))
    else:
        SC = np.random.uniform(sc_min*scale, sc_max*scale)

    mutgen = t_end

    while mutgen in [t_start, t_end]: # exclude edge cases
        logunif = bool(np.random.randint(2))
        if logunif:
            mutgen = int(np.exp(np.random.uniform(np.log(rec), np.log(anc))))
        else:
            mutgen = np.random.randint(rec, anc) # scaled generation of selection onset (before present)

    demography = msprime.Demography()
    extant_indvs = int(Ne/scale)
    if mutgen >= t_start:
        demography.add_population(name="A", initial_size=extant_indvs)
    elif mutgen < t_end:
        demography.add_population(name="A", initial_size=extant_indvs)
        demography.add_population_parameters_change(time=int((t_end-mutgen)/scale), population="A", initial_size=int(N_btnk/scale))
        demography.add_population_parameters_change(time=int((t_start-mutgen)/scale), population="A", initial_size=int(Ne/scale))
    else:
        extant_indvs = int(N_btnk/scale)
        demography.add_population(name="A", initial_size=extant_indvs)
        demography.add_population_parameters_change(time=int((t_start-mutgen)/scale), population="A", initial_size=int(Ne/scale))


    ts_init = msprime.sim_ancestry(samples=extant_indvs,
                               demography=demography,
                               sequence_length=sim_len,
                               recombination_rate=rho)

    center_gen = ts_init.at(sim_len/2)
    subtree_size = np.empty((len(list(center_gen.nodes())), 2), dtype=int) # node_id | total leaves
    for idx, nd in enumerate(center_gen.nodes()):
        for samp_nd in center_gen.samples(nd):
            assert center_gen.population(samp_nd) == 0
        subtree_size[idx] = [nd, center_gen.num_samples(nd)] ## samples, not individuals

    intNode_AF = subtree_size[:,1]/(2*extant_indvs)

    AF_thr = np.random.uniform(0.01, 0.1)
    mut_nd_idx = np.abs(intNode_AF - AF_thr).argmin()

    AF_init = intNode_AF[mut_nd_idx]
    mut_nd = subtree_size[mut_nd_idx, 0]

    print("Mutation @ node:", ts_init.node(mut_nd))
    print(subtree_size[mut_nd_idx])

    init_tables = ts_init.dump_tables()
    init_tables.sites.add_row(position=sim_len/2, ancestral_state="A")
    init_tables.mutations.add_row(site=0, node=mut_nd, derived_state="T", time=int(np.ceil(ts_init.node(mut_nd).time)))

    ts_init_mut = init_tables.tree_sequence()
    ts_init_mut = pyslim.annotate(ts_init_mut, model_type="WF", tick=1, stage="late")

    ts_init_mut.dump(f"{pref}_init.trees")
    np.savetxt(f"{pref}_init.param", [sim_len, SC, mutgen, AF_init, t_start, t_end, N_btnk], fmt='%f')
    print(f"Region_len:{sim_len}\tSC:{SC}\tmutgen:{mutgen}\tAF_init:{AF_init}")

    return 0

sys.exit(main(sys.argv))