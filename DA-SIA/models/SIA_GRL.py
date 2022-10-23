import numpy as np

import tensorflow as tf
from tensorflow import keras # use tf.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Flatten, Activation, Dropout, Dense, Reshape, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D # UpSampling2D, Conv2DTranspose
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras.losses import binary_crossentropy

# Just 3 stacked matrices for prototyping

def fea_scaling(taxa_cnt = 128, max_gen = 1e5):
  scale_mtx = np.empty((3, taxa_cnt-1, taxa_cnt-1))

  scale_mtx[0] = 1/taxa_cnt # F
  scale_mtx[1] = 1/max_gen # W
  scale_mtx[2] = 1/taxa_cnt # R
          
  return scale_mtx

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

def custom_loss(y_true, y_pred):
  # The model will be trained using this loss function, which is identical to normal BCE
  # except when the label for an example is -1, that example is masked out for that task.
  # This allows for examples to only impact loss backpropagation for one of the two tasks.
  y_pred = tf.boolean_mask(y_pred, tf.not_equal(y_true, -1))
  y_true = tf.boolean_mask(y_true, tf.not_equal(y_true, -1))
  return binary_crossentropy(y_true, y_pred)

def create_GRL_mod():

  input_dims = [127, 127, 3] # channel last, use np.moveaxis() to reformat data
  inputs = Input(shape=input_dims)

  # Conv layer #1
  enc_f = Conv2D(128, [127, 3], data_format='channels_last') # kernel size = [height, width]
  h_layer = enc_f(inputs)
  h_layer = Activation('relu')(h_layer)
  h_layer = MaxPool2D(pool_size=(1, 2), padding='valid', data_format='channels_last')(h_layer)
  #h_layer = Dropout(dropout_rate)(h_layer)

  # Conv layer #2
  enc_f = Conv2D(256, [3, 3], data_format='channels_first', padding='valid')
  h_layer = enc_f(h_layer)
  h_layer = Activation('relu')(h_layer)
  h_layer = MaxPool2D(pool_size=(4, 4), padding='valid', data_format='channels_first')(h_layer)
  #h_layer = Dropout(dropout_rate)(h_layer)

  # remember dimension
  #[_, cflat, wflat, hflat] = h_layer.shape.as_list() # channels first

  h_layer = Flatten()(h_layer)

  # Dense layer #1
  # classification branch
  h_class = Dense(1024, use_bias=False)(h_layer)
  h_class = BatchNormalization()(h_class)
  h_class = Activation('relu')(h_class)
  #h_class = Dropout(dropout_rate)(h_class)

  # Dense layer #2
  h_class = Dense(512, activation="relu")(h_class)
  #h_class = Dropout(dropout_rate)(h_class)

  # classification output
  out_class = Dense(1, activation='sigmoid', name='classifier')(h_class)

  ### domain discriminator
  discriminator = GradReverse()(h_layer)
  discriminator = Dense(1024, use_bias=False)(discriminator)
  discriminator = BatchNormalization()(discriminator)
  discriminator = Activation('relu')(discriminator)

  discriminator = Dense(512, activation="relu")(discriminator)
  out_disc = Dense(1, activation='sigmoid', name='discriminator')(discriminator)

  ### GRL model done###
  GRL_model = Model(inputs=inputs, outputs=[out_class, out_disc])
  GRL_model.compile(optimizer='adam',
                    loss=[custom_loss, custom_loss],
                    loss_weights = [1, 1], # equal weighing of two tasks
                    metrics=['accuracy'])
  ######

  return GRL_model

class XYseq(Sequence):
  def __init__(self, src_neu_file, src_neu_trnidx,
                     src_swp_file, src_swp_trnidx,
                     tgt_neu_file, tgt_neu_trnidx,
                     tgt_swp_file, tgt_swp_trnidx,
                     batch_size, tgt_swp_prop=0.5):
    self.offset = 10**6
    #self.memmap_neu = np.load(neu_file, mmap_mode='r')
    #self.memmap_swp = np.load(swp_file, mmap_mode='r')
    self.src_neu = np.load(src_neu_file).astype(np.int32)
    self.src_swp = np.load(src_swp_file).astype(np.int32)

    self.tgt_neu = np.load(tgt_neu_file).astype(np.int32)
    self.tgt_swp = np.load(tgt_swp_file).astype(np.int32)

    if tgt_swp_prop == 0:
      tgt_swp_trnidx = np.array([], dtype=int)
    elif tgt_swp_prop == 1:
      tgt_neu_trnidx = np.array([], dtype=int)
    elif tgt_swp_prop != 0.5:
      if tgt_swp_prop > 0.5:
        neu_sz = int(225000/tgt_swp_prop*(1-tgt_swp_prop))
        tgt_neu_trnidx = np.random.choice(tgt_neu_trnidx, size=neu_sz, replace=False)
      elif tgt_swp_prop < 0.5:
        swp_sz = int(225000/(1-tgt_swp_prop)*tgt_swp_prop)
        tgt_swp_trnidx = np.random.choice(tgt_swp_trnidx, size=swp_sz, replace=False)

    self.scaler = fea_scaling()
    self.batch_size = batch_size
    self.srcidx_map = np.concatenate((src_neu_trnidx, src_swp_trnidx+self.offset))
    self.tgtidx_map = np.concatenate((tgt_neu_trnidx, tgt_swp_trnidx+self.offset))

    src_size = len(self.srcidx_map)
    tgt_size = len(self.tgtidx_map)

    self.no_batch = int(np.floor(np.minimum(src_size, tgt_size) / self.batch_size)) # model sees training sample at most once per epoch

    self.src_class_idx = np.arange(src_size)
    self.src_discr_idx = np.arange(src_size)
    self.tgt_discr_idx = np.arange(tgt_size)

    np.random.shuffle(self.src_class_idx)
    np.random.shuffle(self.src_discr_idx)
    np.random.shuffle(self.tgt_discr_idx)

  def __len__(self):
    return self.no_batch

  def __getitem__(self, idx):
    class_batch_idx = self.src_class_idx[idx*self.batch_size:(idx+1)*self.batch_size]
    class_batch_data = self.srcidx_map[class_batch_idx]

    class_neu_idx = class_batch_data[class_batch_data < self.offset]
    class_swp_idx = class_batch_data[class_batch_data >= self.offset] - self.offset

    discrSrc_batch_idx = self.src_discr_idx[idx*(self.batch_size//2):(idx+1)*(self.batch_size//2)]
    discrSrc_batch_data = self.srcidx_map[discrSrc_batch_idx]

    discrSrc_neu_idx = discrSrc_batch_data[discrSrc_batch_data < self.offset]
    discrSrc_swp_idx = discrSrc_batch_data[discrSrc_batch_data >= self.offset] - self.offset

    discrTgt_batch_idx = self.tgt_discr_idx[idx*(self.batch_size//2):(idx+1)*(self.batch_size//2)]
    discrTgt_batch_data = self.tgtidx_map[discrTgt_batch_idx]

    discrTgt_neu_idx = discrTgt_batch_data[discrTgt_batch_data < self.offset]
    discrTgr_swp_idx = discrTgt_batch_data[discrTgt_batch_data >= self.offset] - self.offset

    batch_X = np.concatenate((self.src_neu[class_neu_idx], self.src_swp[class_swp_idx],
                              self.src_neu[discrSrc_neu_idx], self.src_swp[discrSrc_swp_idx],
                              self.tgt_neu[discrTgt_neu_idx], self.tgt_swp[discrTgr_swp_idx]))

    batch_X = batch_X*self.scaler
    batch_Y_class = np.concatenate((np.zeros(len(class_neu_idx)), np.ones(len(class_swp_idx)),
                                    -1*np.ones(len(discrSrc_neu_idx)), -1*np.ones(len(discrSrc_swp_idx)),
                                    -1*np.ones(len(discrTgt_neu_idx)), -1*np.ones(len(discrTgr_swp_idx))))
    batch_Y_discr = np.concatenate((-1*np.ones(len(class_neu_idx)), -1*np.ones(len(class_swp_idx)),
                                    np.zeros(len(discrSrc_neu_idx)), np.zeros(len(discrSrc_swp_idx)),
                                    np.ones(len(discrTgt_neu_idx)), np.ones(len(discrTgr_swp_idx))))

    batch_X = np.moveaxis(batch_X, 1, -1) # change format to "channel-last"

    assert batch_X.shape[0] == self.batch_size*2, batch_X.shape[0]
    assert batch_Y_class.shape == batch_Y_discr.shape, (batch_Y_class, batch_Y_discr)

    #print(batch_idx, data_idx, neu_idx, swp_idx, batch_Y) #debugging
    return batch_X, {"classifier":batch_Y_class, "discriminator":batch_Y_discr}
    
  def on_epoch_end(self):
    np.random.shuffle(self.src_class_idx)
    np.random.shuffle(self.src_discr_idx)
    np.random.shuffle(self.tgt_discr_idx)