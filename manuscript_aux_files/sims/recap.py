#!/usr/bin/env python3

import argparse
import tskit
import msprime
import numpy as np
import pyslim


def main(args):

    ts = pyslim.update(tskit.load(args.in_path))
    ts_recap = pyslim.recapitate(ts, recombination_rate=args.rho, ancestral_Ne=args.Ne)
    sampN = np.random.choice(ts_recap.samples(), size=args.N, replace=False)
    ts_samp = ts_recap.simplify(samples=sampN)

    print(f">> Max roots: {max(t.num_roots for t in ts.trees())}->{max(t.num_roots for t in ts_samp.trees())}")
    print(f">> Sample size: {ts.num_samples}/{ts.num_individuals} -> {ts_samp.num_samples}/{ts_samp.num_individuals}")

    ts_samp.dump(args.out_path)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recapitate and downsample SLiM tree-sequence.')
    parser.add_argument('--mu', type=float, help='mutation rate', default=1.25e-8)
    parser.add_argument('--rho', type=float, help='recombination rate', default=1.25e-8)
    parser.add_argument('--Ne', type=int, help='effective population size', default=10000)
    parser.add_argument('--N', type=int, help='number of samples', default=128)

    parser.add_argument('in_path', help='path to input tree-sequence')
    parser.add_argument('out_path', help='path to output tree-sequence')
    args = parser.parse_args()

    main(args)
