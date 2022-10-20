#!/usr/bin/env python3

import sys # command line args: sys.argv
#import time
#import gc
#from tqdm import tqdm

import numpy as np
import tensorflow as tf

print(tf.__version__)
print(tf.config.experimental.list_physical_devices('GPU'), flush=True)

from SIA_GRL import *

src_neu = sys.argv[1]
src_swp = sys.argv[2]

tgt_neu = sys.argv[3]
tgt_swp = sys.argv[4]

tgt_swp_prop = float(sys.argv[5])

tot_epoch = int(sys.argv[6])
batch_size = int(sys.argv[7])

out_pref = sys.argv[8]

# preparing training data generator

with np.load(f"{src_neu}_splitIdx.npz") as splitIdx:
    src_neu_idx = splitIdx["TRAIN"]
with np.load(f"{src_swp}_splitIdx.npz") as splitIdx:
    src_swp_idx = splitIdx["TRAIN"]
with np.load(f"{tgt_neu}_splitIdx.npz") as splitIdx:
    tgt_neu_idx = splitIdx["TRAIN"]
    val_neu_idx = splitIdx["VAL"]
with np.load(f"{tgt_swp}_splitIdx.npz") as splitIdx:
    tgt_swp_idx = splitIdx["TRAIN"]
    val_swp_idx = splitIdx["VAL"]

DataGen = XYseq(f"{src_neu}_fea.npy", src_neu_idx,
                f"{src_swp}_fea.npy", src_swp_idx,
                f"{tgt_neu}_fea.npy", tgt_neu_idx,
                f"{tgt_swp}_fea.npy", tgt_swp_idx,
                batch_size, tgt_swp_prop)

# load validation data
val_X = np.concatenate((np.load(f"{tgt_neu}_fea.npy", mmap_mode='r')[val_neu_idx], 
                       np.load(f"{tgt_swp}_fea.npy", mmap_mode='r')[val_swp_idx]))
val_X = val_X*fea_scaling()
val_X = np.moveaxis(val_X, 1, -1)
val_Y_class = np.concatenate((np.zeros(len(val_neu_idx)), np.ones(len(val_swp_idx))))
val_Y_discr = -1*np.ones(len(val_neu_idx)+len(val_swp_idx))

# initilize model
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    DA_class = create_GRL_mod()

## Callbacks ##
erly_stp = keras.callbacks.EarlyStopping(monitor='val_classifier_loss', patience=20)
md_ckpt = keras.callbacks.ModelCheckpoint(filepath=out_pref+'_{epoch:02d}-{val_loss:.2e}',
  save_best_only=True, save_weights_only=False, monitor='val_classifier_loss', verbose=0)

## Train ##
history = DA_class.fit(DataGen,
  validation_data=(val_X, {"classifier":val_Y_class, "discriminator":val_Y_discr}),
  callbacks=[erly_stp, md_ckpt], batch_size=batch_size, epochs=tot_epoch, verbose=2)

#np.save(f"{out_pref}_hist.npy", history)