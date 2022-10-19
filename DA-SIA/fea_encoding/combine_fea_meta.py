#!/usr/bin/env python3

import sys
import gc
import numpy as np
import os.path

from numpy.lib.format import open_memmap
from tqdm import tqdm

helpMsg = '''
        usage: $./combine_fea_meta.py <PATH_pref> <total threads> <oTAG>
            Combine <PATH_pref>_meta_*.npy => <oTAG>_meta.npy
                    <PATH_pref>_fea_*.npy  => <oTAG>_fea.npy
'''

## MANUALLY DEFINE MACRO ##
no_taxa = 128 # of taxa

def main(args):
    if len(args) != 4:    #3 arguments
        return helpMsg

    path_pref = args[1]
    no_threads = int(args[2])
    oTAG = args[3]

    #fea_df = np.empty((0, 3*NRS+1, no_taxa-1, no_taxa-1), dtype=int) # dataframe containing the features
    meta_df = np.empty((0, 5)) # dataframe containing the meta-data [idx, sc, onset, AF, var_pos]
    fea_df_size = np.zeros(no_threads, dtype=int)

    for t in range(1, no_threads+1):
        #fea_df = np.concatenate((fea_df, np.load(path_pref+'_fea_'+str(t)+'.npy').astype(int)))
        meta_df = np.concatenate((meta_df, np.load(path_pref+'_meta_'+str(t)+'.npy')))
        fea_df_size[t-1] = np.load(path_pref+'_fea_'+str(t)+'.npy', mmap_mode='r').shape[0]

        print(t, ":", np.sum(fea_df_size), meta_df.shape, flush=True)
        gc.collect()

    fea_df = open_memmap(oTAG+'_fea.npy', mode='w+', dtype=np.int32, shape=(np.sum(fea_df_size), 3, no_taxa-1, no_taxa-1))

    mem_ptr = 0
    pbar = tqdm(total=no_threads)
    for i in range(no_threads):
        fea_df[mem_ptr:(mem_ptr+fea_df_size[i])] = np.load(path_pref+'_fea_'+str(i+1)+'.npy').astype(np.int32)
        mem_ptr += fea_df_size[i]
        pbar.update()
        gc.collect()

    #np.save(oTAG+'_fea', fea_df)
    fea_df.flush()
    np.save(oTAG+'_meta', meta_df)

    print(path_pref+':', mem_ptr, fea_df.shape, meta_df.shape)
    pbar.close()

    return 0

sys.exit(main(sys.argv))