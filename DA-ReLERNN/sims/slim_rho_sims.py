#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess

import msprime
import numpy as np
import pyslim
import tskit
from sklearn.neighbors import NearestNeighbors

SLIM_PATH = '/grid/siepel/home_norepl/mo/SLiM_3.7'
SCRIPT_PATH = '/grid/siepel/home_norepl/mo/dom_adapt/popgen-dom-adapt/DA-ReLERNN/sims'


def H(n):
    return np.euler_gamma + np.log(n) + 0.5/n - 1./(12*n**2) + 1./(120*n**4)


def sort_min_diff(amat):
    '''Adapted from https://github.com/flag0010/pop_gen_cnn/blob/master/selection/extract.final.dataset.py
    '''
    nbrs = NearestNeighbors(n_neighbors=amat.shape[0], metric='manhattan').fit(amat)
    distances, indices = nbrs.kneighbors(amat)
    smallest = np.argmin(distances.sum(axis=1))
    return amat[indices[smallest]]


def main(args):
    for sim in range(args.runs):
        run_id = args.last_idx + args.thr*args.runs + sim

        if os.path.exists(f'{args.sim_tag}/{args.sim_tag}_{run_id}.npz'):
            print(f'>> Simulation {run_id} already exists, SKIPPING.', flush=True)
            continue

        mu = np.random.uniform(args.mu_min, args.mu_max)
        rho = np.random.uniform(0, args.rho_max)

        slim_cmd = [f'{SLIM_PATH}/slim', '-t', '-m',
                    '-d', f'N={args.Ne}', '-d', f'L={int(args.L)}', '-d', f'G={int(args.G)}',
                    '-d', f'mu={mu}', '-d', f'rho={rho}',
                    '-d', f'outPref="{args.sim_tag}/{args.sim_tag}_{run_id}_temp"',
                    f'{SCRIPT_PATH}/{args.mode}_rho.slim']

        loop_cnt = 0
        while True:
            loop_cnt += 1
            print(f">> SIM:{run_id}, ATTEMPT:{loop_cnt}", flush=True)
            slim_proc = subprocess.run(slim_cmd)
            if slim_proc.returncode == 0:
                break

        ts = pyslim.update(tskit.load(f'{args.sim_tag}/{args.sim_tag}_{run_id}_temp.trees'))

        ts_recap = pyslim.recapitate(ts, recombination_rate=rho, ancestral_Ne=args.Ne)
        sampN = np.random.choice(ts_recap.samples(), size=args.N, replace=False)
        ts_samp = ts_recap.simplify(samples=sampN)

        print(f">> Max roots: {max(t.num_roots for t in ts.trees())}->{max(t.num_roots for t in ts_samp.trees())}")
        print(f">> Sample size: {ts.num_samples}/{ts.num_individuals} -> {ts_samp.num_samples}/{ts_samp.num_individuals}")

        mts = msprime.sim_mutations(ts_samp, rate=mu, keep=False)
        S = mts.num_sites
        theta_W = S/H(args.N-1)

        GTM = mts.genotype_matrix().T
        VP = np.array([s.position for s in mts.sites()], dtype='float32')
        sorted_GTM = sort_min_diff(GTM)

        assert sorted_GTM.shape == (args.N, S) and VP.shape[0] == S, "No. sites mismatch!"

        # padding
        if S >= args.max_sites:
            sorted_GTM = sorted_GTM[:, :args.max_sites]
            VP = VP[:args.max_sites]
        else:
            sorted_GTM = np.pad(sorted_GTM, [(0, 0), (0, args.max_sites-S)])
            VP = np.pad(VP, (0, args.max_sites-S))

        packed_GTM = np.packbits(sorted_GTM, axis=1)

        np.savez_compressed(f'{args.sim_tag}/{args.sim_tag}_{run_id}',
                            GTM=packed_GTM, VP=VP, theta_W=theta_W, mu=mu, rho=rho, S=S)
        print(f'%%\t{run_id}\t{mu}\t{rho}\t{S}\t{theta_W/(4*mu*args.L)}', flush=True)
        os.remove(f'{args.sim_tag}/{args.sim_tag}_{run_id}_temp.trees')

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run recombination rate simulations with SLiM.')
    parser.add_argument('--mu_min', type=float, help='min. mutation rate', default=1.875e-8)
    parser.add_argument('--mu_max', type=float, help='max. mutation rate', default=3.125e-8)
    parser.add_argument('--rho_max', type=float, help='recombination rate', default=6.25e-8)
    parser.add_argument('--Ne', type=int, help='effective population size', default=10000)
    parser.add_argument('--L', type=float, help='chromosome length', default=3e5)
    parser.add_argument('--G', type=float, help='gene length', default=1e4)
    parser.add_argument('--N', type=int, help='number of samples', default=32)
    parser.add_argument('--max_sites', type=int, help='max. # of sites to keep', default=800)

    parser.add_argument('--last_idx', type=int, help='index of the last pre-existing simulations', default=0)
    parser.add_argument(
        'mode', type=str, help='mode of the simulation, regular or with background selection', choices=['reg', 'bkgd'])
    parser.add_argument('sim_tag', type=str, help='simulation tag')
    parser.add_argument('thr', type=int, help='0-based thread #')
    parser.add_argument('runs', type=int, help='# of runs per thread')
    args = parser.parse_args()

    main(args)
