#!/usr/bin/env python3

import sys # command line args: sys.argv

import numpy as np
import tensorflow as tf

print(tf.__version__)
print(tf.config.experimental.list_physical_devices('GPU'), flush=True)

import numpy as np
from tensorflow import keras # use tf.keras
from ReLERNN_mods import *

mod_path = sys.argv[1]
dat_path = sys.argv[2]
out_pref = sys.argv[3]

if "DA" in mod_path:
    trained_mod = keras.models.load_model(mod_path,
        custom_objects={'custom_bce': custom_bce, 'custom_mse': custom_mse})
else:
    trained_mod = keras.models.load_model(mod_path)

# Load test data
with np.load(dat_path) as test_npz:    
  test_bgtm = test_npz["gtm"]
  test_pos = test_npz["pos"]/sim_len # scale position
  Y_true = test_npz["meta"][:, 1]/rho_max # scale rho to [0, 1]
test_bgtm = np.unpackbits(test_bgtm, axis=2)
test_bgtm = np.transpose(test_bgtm, axes=(0, 2, 1)) # Transposition for RNN

print("GTM_TEST\tPOS_TEST\tRHO_TEST", flush=True)
print(test_bgtm.shape, test_pos.shape, Y_true.shape, flush=True)

Y_pred = trained_mod.predict([test_bgtm, test_pos], verbose=1)

np.savez_compressed(out_pref, Y_true=Y_true, Y_pred=Y_pred)