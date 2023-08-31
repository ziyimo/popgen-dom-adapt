#!/usr/bin/env python3

import sys, os
import msprime
import pyslim
import tskit
import numpy as np

helpMsg = '''
        usage: $./swp_IwM_init.py <sc_min> <sc_max> <genbp_min> <genbp_max> <out_pref>

        Put genbp minmax as `-1` to dynamically tune mutgen
'''

N_B = 30281
N_C = 10100
N_BC = 33980
N_ABC = 11593

tau_BC = 845
tau_ABC = 17937
mig_B2C = 0.0002148

def main(args):
    if len(args) != 6:    #5 arguments
        return helpMsg

    scale = 1

    sc_min = float(args[1])
    sc_max = float(args[2])
    pref = args[5]

    sim_len = 1e5 # can be shortened for surveys

    #mu = 1e-9*scale
    rho = 1e-8*scale
    # buf_gen = 100 # soft sweep sims can't have this

    # logunif = bool(np.random.randint(2))
    # if logunif:
    #     SC = np.exp(np.random.uniform(np.log(sc_min*scale), np.log(sc_max*scale)))
    # else:
    #     SC = np.random.uniform(sc_min*scale, sc_max*scale)
    SC = np.random.uniform(sc_min*scale, sc_max*scale)

    rec = int(args[3])
    anc = int(args[4])
    if rec == -1 and anc == -1:
        rec = (10**0.6)*(SC**(-1))
        anc = (10**1.18)*(SC**(-1))
        logunif = bool(np.random.randint(2))
        if logunif:
            mutgen = int(np.exp(np.random.uniform(np.log(rec), np.log(anc))))
        else:
            mutgen = np.random.randint(rec, anc) # scaled generation of selection onset (before present)

    while True:
        mutgen = np.random.randint(rec, anc)
        if mutgen != tau_BC and mutgen != (tau_BC+1): break

    demography = msprime.Demography()
    if mutgen > tau_BC:

        demography.add_population(name="BC", initial_size=int(N_BC/scale))
        demography.add_population_parameters_change(time=int(tau_ABC/scale)-mutgen, population="BC", initial_size=N_ABC)

        ts_init = msprime.sim_ancestry(samples={"BC":int(N_BC/scale)},
                                       demography=demography,
                                       sequence_length=sim_len,
                                       recombination_rate=rho)

        center_gen = ts_init.at(sim_len/2)
        nd_lfs = np.empty((len(list(center_gen.nodes())), 2), dtype=int)
        for idx, nd in enumerate(center_gen.nodes()):
            nd_lfs[idx] = [nd, center_gen.num_samples(nd)] ## samples, not individuals

        DAF_ls = nd_lfs[:, 1]/(2*N_BC/scale)
        AF_thr = np.random.uniform(0.01, 0.1)
        mut_nd_idx = np.abs(DAF_ls - AF_thr).argmin()

        AF_init = nd_lfs[mut_nd_idx, 1]/(2*N_BC/scale)
        AF_B = np.nan
        AF_tot = AF_init
        mut_nd = nd_lfs[mut_nd_idx, 0]

        print("Mutation @ node:", ts_init.node(mut_nd))
        print(nd_lfs[mut_nd_idx])

        init_tables = ts_init.dump_tables()
        init_tables.sites.add_row(position=sim_len/2, ancestral_state="A")
        init_tables.mutations.add_row(site=0, node=mut_nd, derived_state="T", time=int(np.ceil(ts_init.node(mut_nd).time)))

        pyslim.annotate_defaults_tables(init_tables, model_type="WF", slim_generation=1)

    else:

        demography.add_population(name="B", initial_size=int(N_B/scale))
        demography.add_population(name="C", initial_size=int(N_C/scale))
        demography.add_population(name="BC", initial_size=int(N_BC/scale))

        demography.set_migration_rate(source="C", dest="B", rate=mig_B2C)

        demography.add_population_split(time=int(tau_BC/scale)-mutgen, derived=["B", "C"], ancestral="BC")
        demography.add_population_parameters_change(time=int(tau_ABC/scale)-mutgen, population="BC", initial_size=N_ABC)

        ts_init = msprime.sim_ancestry(samples={"B":int(N_B/scale), "C":int(N_C/scale)},
                                       demography=demography,
                                       sequence_length=sim_len,
                                       recombination_rate=rho)

        center_gen = ts_init.at(sim_len/2)
        pop_noSamps = np.empty((len(list(center_gen.nodes())), 5), dtype=int) # node_id | population | total leaves | B leaves | C leaves
        for idx, nd in enumerate(center_gen.nodes()):
            pop_samp_count = [0, 0] # B_cnt, C_cnt
            for samp_nd in center_gen.samples(nd):
                pop_samp_count[center_gen.population(samp_nd)] += 1
            pop_noSamps[idx] = [nd, center_gen.population(nd), center_gen.num_samples(nd)] + pop_samp_count ## samples, not individuals

        AF_isl = pop_noSamps[:, 4]/(2*N_C/scale)
        AF_thr = np.random.uniform(0.01, 0.1)
        mut_nd_idx = np.abs(AF_isl - AF_thr).argmin()

        AF_init = pop_noSamps[mut_nd_idx, 4]/(2*N_C/scale)
        AF_B = pop_noSamps[mut_nd_idx, 3]/(2*N_B/scale)
        AF_tot = pop_noSamps[mut_nd_idx, 2]/(2*(N_B+N_C)/scale)
        mut_nd = pop_noSamps[mut_nd_idx, 0]

        print("Mutation @ node:", ts_init.node(mut_nd))
        print(pop_noSamps[mut_nd_idx])

        init_tables = ts_init.dump_tables()
        init_tables.sites.add_row(position=sim_len/2, ancestral_state="A")
        init_tables.mutations.add_row(site=0, node=mut_nd, derived_state="T", time=int(np.ceil(ts_init.node(mut_nd).time)))

        pyslim.annotate_defaults_tables(init_tables, model_type="WF", slim_generation=1)
        init_tables.populations.clear()

        for j in range(max(init_tables.nodes.population)):
            md = {
                "slim_id": j,
                "selfing_fraction": 0.0,
                "female_cloning_fraction": 0.0,
                "male_cloning_fraction": 0.0,
                "sex_ratio": 0.0,
                "bounds_x0": 0.0,
                "bounds_x1": 0.0,
                "bounds_y0": 0.0,
                "bounds_y1": 0.0,
                "bounds_z0": 0.0,
                "bounds_z1": 0.0,
                "migration_records": []
            }
            init_tables.populations.add_row(metadata=md)
        init_tables.populations.add_row()

    ts_init = pyslim.load_tables(init_tables)

    ts_init.dump(f"{pref}_init.trees")
    np.savetxt(f"{pref}_init.param", [sim_len, SC, mutgen, AF_init, AF_B, AF_tot], fmt='%f')
    print(f"Region_len:{sim_len}\tSC:{SC}\tmutgen:{mutgen}\tAF (C, B, tot):{AF_init}/{AF_B}/{AF_tot}")

    return 0

sys.exit(main(sys.argv))