#!/usr/bin/env python3

import os
import numpy as np
import dendropy
import tskit

os.environ["NUMEXPR_MAX_THREADS"]="272"

def get_site_ppos(ts):
    var_ppos_ls = []
    prev_pos = 0
    for site in ts.sites():
        site_pos = int(site.position)
        if site_pos <= prev_pos:
            if prev_pos == 49999:
                var_ppos_ls.append(-1) # flag indicating this site should be removed
                continue
            else:
                site_pos = prev_pos + 1
        var_ppos_ls.append(site_pos)
        prev_pos = site_pos
    return np.array(var_ppos_ls)

def discrete_pos(pos_ls, reg_len=1e5):
    ''' convert coordinate and resolve rounding duplicate
    pos_ls should be (0, 1)
    TO BE TESTED
    '''
    ppos = pos_ls*reg_len
    ppos = np.round(ppos).astype(int)

    for p in range(1, len(ppos)):
        if ppos[p] <= ppos[p-1]:
            ppos[p] = ppos[p-1] + 1

    return ppos

def samp_var(geno_mtx, pos_ls, AF_min=0.05, AF_max=0.95, left=40000, right=60000): # careful when picking window, may not contain mut. if too small
    AF = np.mean(geno_mtx, axis=1)
    target = np.random.uniform(AF_min, AF_max)
    for idx in np.argsort(np.abs(AF - target)):
        if pos_ls[idx] > left and pos_ls[idx] < right: # make sure the ends are covered by gene trees
            return idx

def encode(nwk_str, no_taxa, der_ls=None):

    dpy_tr = dendropy.Tree.get(data=nwk_str, schema="newick", rooting="default-rooted")

    dpy_tr.calc_node_ages(ultrametricity_precision=1e-01) # assuming branch length in generations
    u_vec = np.zeros(no_taxa) # vector of (internal) node ages, last element always 0

    nd_lab = 0
    for int_nd in dpy_tr.ageorder_node_iter(include_leaves=False, filter_fn=None, descending=True):
        int_nd.label = nd_lab # integer label
        u_vec[nd_lab] = int_nd.age
        nd_lab += 1

    F_mtx = np.zeros((no_taxa-1, no_taxa-1), dtype=int)
    W_mtx = np.zeros((no_taxa-1, no_taxa-1))

    # populate W matrix
    for i in range(no_taxa-1):
        for j in range(i+1):
            W_mtx[i, j] = u_vec[j] - u_vec[i+1]

    for head_node in dpy_tr.preorder_node_iter():
        if head_node.label == 0: # root node
            #print("I am root")
            continue
        if head_node.is_leaf():
            edg_head = no_taxa - 1
        else:
            edg_head = head_node.label
            
        edg_tail = head_node.parent_node.label
        
        #print(edg_head, edg_tail)
        for j in range(edg_tail, edg_head):
            for i in range(j, edg_head):
                F_mtx[i, j] += 1

    if der_ls is None:
        return F_mtx, W_mtx

    R_mtx = np.zeros((no_taxa-1, no_taxa-1), dtype=int) # encoding of derived lineages only
    der_mrca = dpy_tr.mrca(taxon_labels=der_ls)

    for head_node in der_mrca.preorder_iter():
        #print(head_node)
        if head_node == der_mrca: # "root" node
            #print("I am root")
            continue
        if head_node.is_leaf():
            edg_head = no_taxa - 1
        else:
            edg_head = head_node.label
            
        edg_tail = head_node.parent_node.label
        
        #print(edg_head, edg_tail)
        for j in range(edg_tail, edg_head):
            for i in range(j, edg_head):
                R_mtx[i, j] += 1        
    
    return F_mtx, W_mtx, R_mtx

def gen2feature(ts, var_ppos, vOI_gt, no_taxa): # only extract feature for focal genealogy
    focal_gen = ts.at(var_ppos)
    vOI_der = list(map(str, np.nonzero(vOI_gt)[0] + 1))

    F, W, R = encode(focal_gen.newick(precision=5), no_taxa, vOI_der)

    return np.stack((F.T+np.tril(F, -1), W.T+np.tril(W, -1), R.T+np.tril(R, -1)))

def ts2feature(ts, var_ppos, vOI_gt, NRS, no_taxa):
    '''
    intervals: dataframe with start and end position of genealogies
    trees: list of newick strings
    var_ppos: integer physical position of focal variant
    vOI_gt: genotype of focal variant
    NRS: # of non-recombining segments (i.e. trees) represented in the feature
    no_ft: # of flanking trees for feature extraction
    no_taxa: # of taxa
    '''
    no_ft = NRS//2

    trees = []
    intervals = np.empty((0,2), int)
    for tree in ts.trees():
        left, right = map(int, tree.interval)
        intervals = np.vstack((intervals, [left, right]))
        trees = np.append(trees, tree.newick(precision=5))

    intervals[-1, 1] = 1e5 # force last tree to cover the rest of the region, hard coded

    mtx_encoding = np.empty((3*NRS+1, no_taxa-1, no_taxa-1), dtype=int)

    c_idx = np.where((intervals[:,0]<=var_ppos) & (intervals[:,1]>var_ppos))[0][0]
    window = range(c_idx-no_ft, c_idx+no_ft+1)
    s_indices = np.take(np.arange(len(trees)), window, mode='clip') # mode='clip' or 'wrap'

    for cnt, st_idx in enumerate(s_indices):
        #st = dendropy.Tree.get(data=trees[st_idx], schema="newick")
        end = intervals[st_idx, 1]
        begin = intervals[st_idx, 0]

        length = end - begin

        if st_idx == c_idx and cnt == no_ft:
            # N.B. (per .newick() doc for tskit) By default, leaf nodes are labelled with their numerical ID + 1
            vOI_der = list(map(str, np.nonzero(vOI_gt)[0] + 1))
            F, W, R = encode(trees[st_idx], no_taxa, vOI_der)
            mtx_encoding[3*cnt:(3*cnt+4)] = np.stack((length*np.ones((no_taxa-1, no_taxa-1), dtype=int), F.T+np.tril(F, -1), W.T+np.tril(W, -1), R.T+np.tril(R, -1)))
        else:
            F, W = encode(trees[st_idx], no_taxa)
            shift = int(cnt>no_ft)
            mtx_encoding[(3*cnt+shift):(3*cnt+shift+3)] = np.stack((length*np.ones((no_taxa-1, no_taxa-1), dtype=int), F.T+np.tril(F, -1), W.T+np.tril(W, -1)))

    return mtx_encoding