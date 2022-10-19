#!/usr/bin/env python3

import sys # command line args: sys.argv
#import time
#import gc
#from tqdm import tqdm

import numpy as np
import tensorflow as tf

print(tf.__version__)
print(tf.config.experimental.list_physical_devices('GPU'), flush=True)

from SIA_base import *

src_neu = sys.argv[1]
src_swp = sys.argv[2]

tot_epoch = int(sys.argv[3])
batch_size = int(sys.argv[4])

out_pref = sys.argv[5]

# preparing training data generator

with np.load(f"{src_neu}_splitIdx.npz") as splitIdx:
    src_neu_idx = splitIdx["TRAIN"]
    val_neu_idx = splitIdx["VAL"]
with np.load(f"{src_swp}_splitIdx.npz") as splitIdx:
    src_swp_idx = splitIdx["TRAIN"]
    val_swp_idx = splitIdx["VAL"]

srcDataGen = XYseq(f"{src_neu}_fea.npy", f"{src_swp}_fea.npy", src_neu_idx, src_swp_idx, batch_size)

# load validation data
val_X = np.concatenate((np.load(f"{src_neu}_fea.npy", mmap_mode='r')[val_neu_idx], 
                       np.load(f"{src_swp}_fea.npy", mmap_mode='r')[val_swp_idx]))
val_X = val_X*fea_scaling()
val_X = np.moveaxis(val_X, 1, -1)
val_Y = np.concatenate((np.zeros(len(val_neu_idx)), np.ones(len(val_swp_idx))))

# initilize model
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    convnet_class = create_convnet()

## Callbacks ##
erly_stp = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
md_ckpt = keras.callbacks.ModelCheckpoint(filepath=out_pref+'_{epoch:02d}-{val_loss:.2e}',
  save_best_only=True, save_weights_only=False, monitor='val_loss', verbose=0)

## Train ##
history = convnet_class.fit(srcDataGen,
  validation_data=(val_X, val_Y),
  callbacks=[erly_stp, md_ckpt], batch_size=batch_size, epochs=tot_epoch, verbose=2)

np.save(f"{out_pref}_hist.npy", history)