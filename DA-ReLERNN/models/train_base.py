#!/usr/bin/env python3

import sys # command line args: sys.argv
import gc

import numpy as np
import tensorflow as tf

print(tf.__version__)
print(tf.config.experimental.list_physical_devices('GPU'), flush=True)

from ReLERNN_mods import *

TRAIN_pref = sys.argv[1]
VAL_pref = sys.argv[2]
TAG = sys.argv[3]
CONT = sys.argv[4]
tot_epoch = int(sys.argv[5])
batch_size = int(sys.argv[6])

# Training data generator
srcDataGen = GTimgSequence(f"{TRAIN_pref}_train.npz", batch_size)

# Load validation data
with np.load(f"{VAL_pref}_val.npz") as val_npz:    
  val_bgtm = val_npz["gtm"]
  val_pos = val_npz["pos"]/sim_len # scale position
  val_rho = val_npz["meta"][:, 1]/rho_max # scale rho to [0, 1]
val_bgtm = np.unpackbits(val_bgtm, axis=2)
val_bgtm = np.transpose(val_bgtm, axes=(0, 2, 1)) # Transposition for RNN

print("GTM_VAL\tPOS_VAL\tRHO_VAL", flush=True)
print(val_bgtm.shape, val_pos.shape, val_rho.shape, flush=True)

print("TRAIN_SIZE\tNO_BATCH", flush=True)
print(srcDataGen.dsize, srcDataGen.no_batch, flush=True)

if CONT == "new":
  print(">>>CREATE NEW MODEL", flush=True)
  # mirrored_strategy = tf.distribute.MirroredStrategy()
  # with mirrored_strategy.scope():
  merged_model = ReLERNN_base()
else:
  print(f">>>LOAD FROM EXISTING MODEL: {CONT}", flush=True)
  merged_model = tf.keras.models.load_model(CONT)

## Callbacks ##
erly_stp = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
md_ckpt = keras.callbacks.ModelCheckpoint(filepath=TAG+'_{epoch:02d}-{val_loss:.2e}',
  save_best_only=True, save_weights_only=False, monitor='val_loss', verbose=0)

## Train using generator ##
history = merged_model.fit(srcDataGen, validation_data=([val_bgtm, val_pos], val_rho),
  callbacks=[erly_stp, md_ckpt], epochs=tot_epoch, verbose=2)

# rho_val_pred = merged_model.predict([val_bgtm, val_pos])
# print(np.transpose(np.vstack((val_rho, rho_val_pred)))) # sanity check
