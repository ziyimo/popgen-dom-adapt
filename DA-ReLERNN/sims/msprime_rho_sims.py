#!/usr/bin/env python3

import sys

import numpy as np
import msprime
from numpy import euler_gamma
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

helpMsg = '''
    usage: $./msprime_sims.py <# individuals> <max # sites> <total sims> <label>

'''

## MACROS ##

# Adrion et al.
rho_max = 6.25e-8
mu_min = 1.875e-8
mu_max = 3.125e-8
sim_len = 3e5

def H(n):
    return euler_gamma + np.log(n) + 0.5/n - 1./(12*n**2) + 1./(120*n**4)

def sort_min_diff(amat):
    '''Adapted from https://github.com/flag0010/pop_gen_cnn/blob/master/selection/extract.final.dataset.py
    '''
    nbrs = NearestNeighbors(n_neighbors=amat.shape[0], metric='manhattan').fit(amat)
    distances, indices = nbrs.kneighbors(amat)
    smallest = np.argmin(distances.sum(axis=1))
    return amat[indices[smallest]]

def run_sim(N_indvls):
    
    mu = np.random.uniform(mu_min, mu_max)
    rho = np.random.uniform(0, rho_max)
    #ts = msprime.sim_ancestry(N_indvls, demography=CEU_dem, recombination_rate=rho, sequence_length=sim_len)
    ts = msprime.sim_ancestry(N_indvls, population_size=10000, recombination_rate=rho, sequence_length=sim_len)
    mts = msprime.sim_mutations(ts, rate=mu)

    S = mts.num_sites
    theta_W = S/H(N_indvls*2-1)
    
    GTM = mts.genotype_matrix().T
    VP = np.array([s.position for s in mts.sites()], dtype='float32')
    sorted_GTM = sort_min_diff(GTM)
    
    assert sorted_GTM.shape == (N_indvls*2, S) and VP.shape[0] == S, "No. sites mismatch!"

    # padding
    if S >= max_sites_ret:
        sorted_GTM = sorted_GTM[:, (S//2-max_sites_ret//2):(S//2+max_sites_ret//2)]
        VP = VP[(S//2-max_sites_ret//2):(S//2+max_sites_ret//2)]
    else:
        sorted_GTM = np.pad(sorted_GTM, [(0, 0), (0, max_sites_ret-S)])
        VP = np.pad(VP, (0, max_sites_ret-S))
    
    packed_GTM = np.packbits(sorted_GTM, axis=1)
    
    return [mu, rho, S, theta_W/(4*mu*sim_len)], packed_GTM, VP

def main(args):
    global max_sites_ret #, CEU_dem

    if len(args) != 5:    #4 argument(s)
        return helpMsg

    no_indvl = int(args[1])
    max_sites_ret = int(args[2])
    tot_sims = int(args[3])
    #seed = int(args[4])
    handle = args[4]
    
    # rng = np.random.default_rng(seed)
    # sim_seed = rng.integers(low=1, high=2**31, size=(tot_sims, 2)) # ancestry and mutation seed

    # OOA_dem = np.loadtxt("OOA_expansion.txt", dtype=int)
    # CEU_dem = msprime.Demography()
    # CEU_dem.add_population(name="CEU", initial_size=OOA_dem[0, 1])
    # for t in range(1, OOA_dem.shape[0]):
    #     CEU_dem.add_population_parameters_change(time=OOA_dem[t, 0], population="CEU", initial_size=OOA_dem[t, 1])

    stats_df = np.empty((tot_sims, 4))
    gtm_df = np.empty((tot_sims, no_indvl*2, max_sites_ret//8), dtype=np.uint8)
    varPos_df = np.empty((tot_sims, max_sites_ret), dtype=np.float32)

    with open(f"{handle}.log", "w") as f:
        for idx in tqdm(range(tot_sims)):
            stats_df[idx], gtm_df[idx], varPos_df[idx] = run_sim(no_indvl)
            print(stats_df[idx], file=f, flush=True)

    np.savez_compressed(f"{handle}_gtsims", stats=stats_df, gtm=gtm_df, varPos=varPos_df)
        
if __name__ == "__main__":
    sys.exit(main(sys.argv))