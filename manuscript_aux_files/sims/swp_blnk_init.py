#!/usr/bin/env python3

import sys, os
import msprime
import pyslim
import tskit
import numpy as np

helpMsg = '''
        usage: $./swp_blnk_init.py <`n` or `s`> <sc_min> <sc_max> <Ne_bottleneck> <out_pref>
'''

scale = 1
sim_len = 1e5 # can be shortened for surveys
rho = 1e-7
Ne = 10000

def log_log_lin(sc_vec, k, b):
    return 10**b*np.power(sc_vec, -k)

def main(args):
    if len(args) != 6:    # 5 arguments
        return helpMsg

    mode = args[1]
    sc_min = float(args[2])
    sc_max = float(args[3])

    t_end = 1000 # bottleneck end time
    t_start = 2000 # bottleneck start time
    N_btnk = int(args[4])

    pref = args[5]

    logunif = np.random.uniform()
    if logunif < 0.1: # << tune proportion here
        SC = np.exp(np.random.uniform(np.log(sc_min*scale), np.log(sc_max*scale)))
    else:
        SC = np.random.uniform(sc_min*scale, sc_max*scale)

    rec = log_log_lin(SC, 1, 0.5)
    anc = np.minimum(log_log_lin(SC, 1, 1.1), 8000)
    if N_btnk < 5000:
        rec = np.minimum(rec, 1000)

    mutgen = t_end
    while mutgen in [t_start, t_start+1, t_end, t_end+1]: # exclude edge cases
        logunif = np.random.uniform()
        if logunif < 0: # << tune proportion here
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

    if mode == "n":
        ts_init = pyslim.annotate(ts_init, model_type="WF", tick=1, stage="late")

        ts_init.dump(f"{pref}_init.trees")
        np.savetxt(f"{pref}_init.param", [sim_len, SC, mutgen, -1, t_start, t_end, N_btnk], fmt='%f')
        print(f"Region_len:{sim_len}\tSC:{SC}\tmutgen:{mutgen}\tAF_init:N/A")

        return 0

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