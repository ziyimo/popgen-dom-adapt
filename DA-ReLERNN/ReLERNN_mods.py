#!/usr/bin/env python3

import gc

import numpy as np
import tensorflow as tf
from tensorflow import keras  # use tf.keras
from tensorflow.keras.layers import (GRU, LSTM, Activation, BatchNormalization,
                                     Bidirectional, Dense, Dropout, Flatten,
                                     Input, Layer, concatenate)
from tensorflow.keras.losses import binary_crossentropy, mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence

# MACROS
rho_max = 6.25e-8
sim_len = 3e5
no_sites = 800
no_samps = 32


class GTmaskedSequence(Sequence):
  def __init__(self, src_path, tar_path, batch_size):
    with np.load(src_path) as src_npz:    
      self.src_bgtm = src_npz["gtm"]
      self.src_pos = src_npz["pos"]/sim_len # scale position
      self.src_rho = src_npz["meta"][:, 1]/rho_max # scale rho to [0, 1]

    with np.load(tar_path) as tar_npz:
      self.tar_bgtm = tar_npz["gtm"]
      self.tar_pos = tar_npz["pos"]/sim_len # scale position

    self.batch_size = batch_size

    src_size = self.src_bgtm.shape[0]
    tgt_size = self.tar_bgtm.shape[0]

    #assert self.pos.shape == (self.dsize, no_sites) and self.rho.shape[0] == self.dsize, "Check data dimensions!"

    self.no_batch = int(np.floor(np.minimum(src_size, tgt_size) / self.batch_size)) # model sees training sample at most once per epoch
    self.src_pred_idx = np.arange(src_size)
    self.src_discr_idx = np.arange(src_size)
    self.tgt_discr_idx = np.arange(tgt_size)

    np.random.shuffle(self.src_pred_idx)
    np.random.shuffle(self.src_discr_idx)
    np.random.shuffle(self.tgt_discr_idx)

  def __len__(self):
    return self.no_batch

  def __getitem__(self, idx):
    pred_batch_idx = self.src_pred_idx[idx*self.batch_size:(idx+1)*self.batch_size]
    discrSrc_batch_idx = self.src_discr_idx[idx*(self.batch_size//2):(idx+1)*(self.batch_size//2)]
    discrTgt_batch_idx = self.tgt_discr_idx[idx*(self.batch_size//2):(idx+1)*(self.batch_size//2)]

    batch_X1 = np.concatenate((self.src_bgtm[pred_batch_idx],
                          self.src_bgtm[discrSrc_batch_idx],
                          self.tar_bgtm[discrTgt_batch_idx]))
    batch_X1 = np.unpackbits(batch_X1, axis=2)
    batch_X1 = np.transpose(batch_X1, axes=(0, 2, 1)) # Transposition for RNN

    batch_X2 = np.concatenate((self.src_pos[pred_batch_idx],
                          self.src_pos[discrSrc_batch_idx],
                          self.tar_pos[discrTgt_batch_idx]))

    batch_Y_pred = np.concatenate((self.src_rho[pred_batch_idx],
                                    -1*np.ones(len(discrSrc_batch_idx)),
                                    -1*np.ones(len(discrTgt_batch_idx))))

    batch_Y_discr = np.concatenate((-1*np.ones(len(pred_batch_idx)),
                                    np.zeros(len(discrSrc_batch_idx)),
                                    np.ones(len(discrTgt_batch_idx))))

    assert batch_X1.shape[0] == self.batch_size*2, batch_X1.shape[0]
    assert batch_X2.shape[0] == self.batch_size*2, batch_X2.shape[0]
    assert batch_Y_pred.shape == batch_Y_discr.shape, (batch_Y_pred, batch_Y_discr)

    return (batch_X1, batch_X2), {"predictor":batch_Y_pred, "discriminator":batch_Y_discr}

  def on_epoch_end(self):
    np.random.shuffle(self.src_pred_idx)
    np.random.shuffle(self.src_discr_idx)
    np.random.shuffle(self.tgt_discr_idx)
    gc.collect()

## GRL implementation from: https://stackoverflow.com/questions/56841166/how-to-implement-gradient-reversal-layer-in-tf-2-0
@tf.custom_gradient
def grad_reverse(x):
  y = tf.identity(x)
  def custom_grad(dy):
    return -dy
  return y, custom_grad

class GradReverse(Layer):
  def __init__(self):
    super().__init__()

  def call(self, x):
    return grad_reverse(x)

def custom_bce(y_true, y_pred):
  # The model will be trained using this loss function, which is identical to normal BCE
  # except when the label for an example is -1, that example is masked out for that task.
  # This allows for examples to only impact loss backpropagation for one of the two tasks.
  y_pred = tf.boolean_mask(y_pred, tf.not_equal(y_true, -1))
  y_true = tf.boolean_mask(y_true, tf.not_equal(y_true, -1))
  return binary_crossentropy(y_true, y_pred)

def custom_mse(y_true, y_pred):
  y_pred = tf.boolean_mask(y_pred, tf.not_equal(y_true, -1))
  y_true = tf.boolean_mask(y_true, tf.not_equal(y_true, -1))
  return mean_squared_error(y_true, y_pred)

def ReLERNN_DA():

  gt_inputs = Input(shape=(no_sites, no_samps))
  model = Bidirectional(LSTM(256,return_sequences=True))(gt_inputs)
  model = Bidirectional(LSTM(128,return_sequences=False))(model)
  model = Dense(256, use_bias=False)(model)
  model = BatchNormalization()(model)
  model = Activation("relu")(model)
  #model = Dropout(0.35)(model)

  #----------------------------------------------------

  position_inputs = Input(shape=(no_sites,))
  m2 = Dense(256, activation='relu')(position_inputs)

  #----------------------------------------------------

  model = concatenate([model,m2])

  # regression branch
  task_pred = Dense(256, use_bias=False)(model)
  task_pred = BatchNormalization()(task_pred)
  task_pred = Activation("relu")(task_pred)
  #task_pred = Dropout(0.35)(task_pred)
  task_pred = Dense(64, activation='relu')(task_pred)
  out_pred = Dense(1, activation='relu', name='predictor')(task_pred)

  # domain discriminator branch
  dom_discr = GradReverse()(model)
  dom_discr = Dense(256, use_bias=False)(dom_discr)
  dom_discr = BatchNormalization()(dom_discr)
  dom_discr = Activation("relu")(dom_discr)
  #dom_discr = Dropout(0.35)(dom_discr)
  dom_discr = Dense(64, activation='relu')(dom_discr)
  out_discr = Dense(1, activation='sigmoid', name='discriminator')(dom_discr)

  GRL_model = Model(inputs=[gt_inputs, position_inputs], outputs=[out_pred, out_discr])
  GRL_model.compile(optimizer='adam',
                    loss=[custom_mse, custom_bce],
                    loss_weights = [1, 5e-3], # equal weighing of two tasks (empirically)
                    metrics={'predictor': ['mae', tf.keras.metrics.RootMeanSquaredError()], 'discriminator': 'accuracy'})

  return GRL_model
