#!/usr/bin/env python3

import math
import os
import subprocess
import sys

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../DA-SIA/fea_encoding"))
from utils import *

helpMsg = '''
    usage: $./discoal_sim2fea.py <`swp` or `neu`> <handle> <s_low-s_high> <f_low-f_high> <c_low-c_high> <NOCHR> <thr> <sims_p_thr>

    Simulation de novo
    Run RELATE and output features of true genealogies
    <s_low-s_high>: range of selection coefficient
    <f_low-f_high>: range of starting sweep frequency (0-0: hard sweep)
    <c_low-c_high>: range of ending sweep frequency

    Outputs:
        "discoal_<handle>_<NOCHR>_<`swp` or `neu`>_<thr>.npz"
'''

## MACRO ##
DISCOAL_PATH = '/grid/siepel/home_norepl/mo/discoal-master'
DEM_PATH = '/grid/siepel/home_norepl/mo/dom_adapt/popgen-dom-adapt/CEU_reanalysis/sims/discoal_CEU.txt'

## Simulation parameters ##
N_0 = 188088
SWP_POS = 0.5 #beneficial mutation positions (0-1)
length = 100000 #length of the simulated region
mu_range = [1.25e-8, 2.5e-8]
rho_range = [1.25e-8, 2.5e-8]
SIGMA = 4 # time discretization for sweep simulation range: (4, 400),dt=(1/sigma*N)

def parse_hdswp_cmd(cmd_line):
    
    args = cmd_line.split()
    N_curr = int(args[args.index('-N') + 1])
    alpha = float(args[args.index('-a') + 1])
    CAF = float(args[args.index('-c') + 1])
    if '-f' in args:
        SAF = float(args[args.index('-f') + 1])
    else:
        SAF = 0
    selcoef = alpha/(2*N_curr)
    
    return selcoef, SAF, CAF

def discoal2fea(discoal_file, cat, c_low, c_high, no_taxa, soft_flag):

    with open(discoal_file, "r") as discoalF:
        if cat == 'swp':
            SC, SAF, CAF = parse_hdswp_cmd(discoalF.readline().strip())
        elif cat == 'neu':
            SC = 0
            SAF = 0

        read_GT = False
        seek_onset = False # seek sweep onset
        onset_gen = -1
        for line in discoalF:
            if line[:8] == "segsites":
                segsites = int(line.strip().split()[1])
                gtm = np.empty((0, segsites), dtype=np.int8)
                continue
            if line[:9] == "positions":
                var_pos = np.array(line.strip().split()[1:], dtype=float)
                read_GT = True
                continue
            if line[:4] == "Freq":
                seek_onset = True
                continue
            if seek_onset:
                gbp_der_anc = line.strip().split()
                if len(gbp_der_anc) == 3:
                    onset_gen = float(gbp_der_anc[0]) # in coalc. unit
                    if soft_flag and float(gbp_der_anc[1]) < SAF: # 1st time point going backward
                        seek_onset = False
                elif onset_gen != -1:
                    seek_onset = False
                continue
            if read_GT:
                gtm = np.vstack((gtm, np.fromstring(line.strip(), dtype=np.int8) - ord("0")))

    gtm = np.transpose(gtm)

    try:
        if cat == 'neu':
            samp_idx = samp_var(gtm, var_pos, c_low, c_high, 0.4, 0.6)
            foc_var_pos = var_pos[samp_idx]
            CAF = np.mean(gtm[samp_idx])

        elif cat == 'swp':
            foc_var_pos = SWP_POS
            samp_idx = np.nonzero(var_pos == foc_var_pos)[0][0]
            

        vOI_gt = gtm[samp_idx].flatten()
        assert len(vOI_gt) == no_taxa, f"Check genotype dimension ({len(vOI_gt)})!"

        tar_pos = length*foc_var_pos
        left = 0

        with open(discoal_file, "r") as discoalF:
            for line in discoalF:
                if line[0] == "[":
                    intvl, nwk_str = line.strip().split("]")
                    intvl = int(intvl[1:])
                    if left + intvl > tar_pos: # not sure if 0- or 1- based coordinates
                        F, W, R = encode(nwk_str, no_taxa, list(map(str, np.nonzero(vOI_gt)[0])))
                        break
                    left += intvl
    except KeyError as err:
        print("KeyError: ", err)
        print(f"samp_idx: {samp_idx}, foc_var_pos: {foc_var_pos}, CAF: {CAF}")
        print(vOI_gt, list(map(str, np.nonzero(vOI_gt)[0])))
        sys.exit(1)
 
    return SC, SAF, CAF, onset_gen, foc_var_pos, segsites, np.array((F, W*4*N_0, R), dtype=np.int32)

def main(args):
    if len(args) != 9:    #8 argument(s)
        return helpMsg
    
    mode = args[1] # <`swp` or `neu`>
    handle = args[2]
    s_low, s_high = list(map(float, args[3].split("-")))
    f_low, f_high = list(map(float, args[4].split("-")))
    c_low, c_high = list(map(float, args[5].split("-")))
    no_chrs = int(args[6])
    thr = int(args[7]) # one-based
    no_sims = int(args[8])

    with open(DEM_PATH, "r") as inF:
        discoal_dem = inF.readline().strip().split()
    
    print(f"discoal_{handle}_{no_chrs}_{mode}_{thr}: {no_sims} sims", flush=True)

    SC_arr = np.concatenate((np.random.uniform(s_low, s_high, size=int(no_sims/2)),
                             np.exp(np.random.uniform(np.log(s_low), np.log(s_high), size=int(no_sims/2)))))
    mu_arr = np.random.uniform(mu_range[0], mu_range[1], size=no_sims)
    rho_arr = np.random.uniform(rho_range[0], rho_range[1], size=no_sims)

    if f_low == 0 and f_high == 0:
        SAF_arr = np.zeros(no_sims)
        soft = False
    else:
        SAF_arr = np.random.uniform(f_low, f_high, size=no_sims) # Starting AF
        soft = True
    CAF_arr = np.random.uniform(c_low, c_high, size=no_sims) # Current AF
    onset_arr = np.empty(no_sims)
    foc_var_pos_arr = np.empty(no_sims)
    site_cnt_arr = np.empty(no_sims, dtype=np.int32)

    fea_df = np.empty((no_sims, 3, no_chrs-1, no_chrs-1), dtype=np.int32)

    for samp in tqdm(range(no_sims)):
        #if os.stat(discoalF_path).st_size == 0: continue
        theta = 4*N_0*mu_arr[samp]*length
        R = 4*N_0*rho_arr[samp]*length

        temp_discoalF = f"discoal_temp/discoal_{handle}_{no_chrs}_{mode}_{thr}_{samp}.discoal"
        discoal_cmd = [f"{DISCOAL_PATH}/discoal", str(no_chrs), "1", str(length),
            "-t", str(theta), "-r", str(R), "-T"]

        if mode == 'swp':
            sel = 2*N_0*SC_arr[samp]
            discoal_cmd += ["-c", str(CAF_arr[samp]), "-ws", "0", "-a", str(sel),
            "-N", str(N_0), "-i", str(SIGMA), "-x", str(SWP_POS)]
            if soft:
                discoal_cmd += ["-f", str(SAF_arr[samp])]

        discoal_cmd += discoal_dem

        loop_cnt = 0
        while True:
            loop_cnt += 1
            print(f"SIM:{samp}, ATTEMPT:{loop_cnt}", flush=True)
            with open(temp_discoalF, "w") as outF:
                discoal_proc = subprocess.run(discoal_cmd, stdout=outF)
            if discoal_proc.returncode == 0: break

        SC, SAF, CAF, onset_gen, foc_var_pos, no_sites, fea = discoal2fea(temp_discoalF, mode, c_low, c_high, no_chrs, soft)
        
        if mode == 'neu':
            SC_arr[samp] = SC
            CAF_arr[samp] = CAF
            SAF_arr[samp] = SAF
        elif mode == 'swp':
            assert math.isclose(SC, SC_arr[samp], rel_tol=1e-3), f"Check s: {SC} != {SC_arr[samp]}"
            assert math.isclose(CAF, CAF_arr[samp], rel_tol=1e-3), f"Check f: {CAF} != {CAF_arr[samp]}"
            assert math.isclose(SAF, SAF_arr[samp], rel_tol=1e-3), f"Check f_init: {SAF} != {SAF_arr[samp]}"

        #print(SC, f"{SAF}=>{CAF}", ";", SC_arr[samp], f"{SAF_arr[samp]}=>{CAF_arr[samp]}", ";", onset_gen) # sanity check
        onset_arr[samp] = onset_gen
        foc_var_pos_arr[samp] = foc_var_pos
        site_cnt_arr[samp] = no_sites
        fea_df[samp] = fea

        os.remove(temp_discoalF) # comment out for debugging

    np.savez_compressed(f"discoal_out/discoal_{handle}_{no_chrs}_{mode}_{thr}",
        SC=SC_arr, SAF=SAF_arr, CAF=CAF_arr, onset=onset_arr, foc_var_pos=foc_var_pos_arr, site_cnt=site_cnt_arr, fea=fea_df)

    print("Shape:", SC_arr.shape, SAF_arr.shape, CAF_arr.shape, onset_arr.shape, foc_var_pos_arr.shape, site_cnt_arr.shape, fea_df.shape, flush=True)

sys.exit(main(sys.argv))
