#!/usr/bin/env python3

import sys # command line args: sys.argv

import numpy as np
import tensorflow as tf

print(tf.__version__)
print(tf.config.experimental.list_physical_devices('GPU'), flush=True)

import numpy as np
from tensorflow import keras # use tf.keras
from SIA_sc_GRL import *

mod_path = sys.argv[1]
dat_pref = sys.argv[2]
out_pref = sys.argv[3]

if "DA" in mod_path:
    trained_mod = keras.models.load_model(mod_path,
        custom_objects={'custom_bce': custom_bce, 'custom_mse': custom_mse})
else:
    trained_mod = keras.models.load_model(mod_path)

with np.load(f"{dat_pref}_splitIdx.npz") as swpIdx:
    test_X = np.load(f"{dat_pref}_fea.npy", mmap_mode='r')[swpIdx["TEST"]]    
    test_X = test_X*fea_scaling()
    test_X = np.moveaxis(test_X, 1, -1)

    test_Y = np.load(f"{dat_pref}_meta.npy")[swpIdx["TEST"]][:, 1]

Y_pred = trained_mod.predict(test_X, verbose=1)

np.savez_compressed(out_pref, Y_true=test_Y, Y_pred=Y_pred)