#!/usr/bin/env python3

import sys # command line args: sys.argv
#import time
#import gc
#from tqdm import tqdm

import numpy as np
import tensorflow as tf

print(tf.__version__)
print(tf.config.experimental.list_physical_devices('GPU'), flush=True)

from SIA_sc_GRL import *

src_swp = sys.argv[1]
tgt_swp = sys.argv[2]

tot_epoch = int(sys.argv[3])
batch_size = int(sys.argv[4])
bce_wgt = float(sys.argv[5])

tgt_size_prop = float(sys.argv[6])

out_pref = sys.argv[7]

# preparing training data generator

with np.load(f"{src_swp}_splitIdx.npz") as splitIdx:
    src_swp_idx = splitIdx["TRAIN"]
with np.load(f"{tgt_swp}_splitIdx.npz") as splitIdx:
    tgt_swp_idx = splitIdx["TRAIN"]
    val_swp_idx = splitIdx["VAL"]

if tgt_size_prop < 1.0:
    tgt_swp_idx = np.random.choice(tgt_swp_idx, int(len(tgt_swp_idx)*tgt_size_prop), replace=False)

print(f"src size: {len(src_swp_idx)}\ntgt size: {len(tgt_swp_idx)}", flush=True)

DataGen = XYseq(f"{src_swp}_fea.npy", f"{src_swp}_meta.npy", src_swp_idx,
                f"{tgt_swp}_fea.npy", tgt_swp_idx,
                batch_size)

# load validation data
val_X = np.load(f"{tgt_swp}_fea.npy", mmap_mode='r')[val_swp_idx]
val_X = val_X*fea_scaling()
val_X = np.moveaxis(val_X, 1, -1)

val_Y_pred = np.load(f"{tgt_swp}_meta.npy")[val_swp_idx][:, 1]

val_Y_discr = -1*np.ones(len(val_swp_idx))

# initilize model
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    DA_class = create_GRL_mod(bce_wgt)

## Callbacks ##
erly_stp = keras.callbacks.EarlyStopping(monitor='val_predictor_loss', patience=20)
md_ckpt = keras.callbacks.ModelCheckpoint(filepath=out_pref+'_{epoch:02d}-{val_loss:.2e}',
  save_best_only=True, save_weights_only=False, monitor='val_predictor_loss', verbose=0)

## Train ##
history = DA_class.fit(DataGen,
  validation_data=(val_X, {"predictor":val_Y_pred, "discriminator":val_Y_discr}),
  callbacks=[erly_stp, md_ckpt], batch_size=batch_size, epochs=tot_epoch, verbose=2)

#np.save(f"{out_pref}_hist.npy", history)