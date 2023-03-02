import numpy as np
import tensorflow as tf
from tensorflow import keras  # use tf.keras
from tensorflow.keras.layers import (  # UpSampling2D, Conv2DTranspose
    Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input,
    Layer, MaxPool2D, Reshape)
from tensorflow.keras.losses import binary_crossentropy, mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import Sequence


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

def create_GRL_mod(bce_weight):

  dropout_rate = 0.2

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
  # regression branch
  h_class = Dense(1024, use_bias=False)(h_layer)
  h_class = BatchNormalization()(h_class)
  h_class = Activation('relu')(h_class)
  h_class = Dropout(dropout_rate)(h_class)

  # Dense layer #2
  h_class = Dense(512, activation="relu")(h_class)
  h_class = Dropout(dropout_rate)(h_class)

  # classification output
  out_continuous = Dense(1, activation='linear', name='predictor')(h_class)

  ### domain discriminator
  discriminator = GradReverse()(h_layer)
  discriminator = Dense(1024, use_bias=False)(discriminator)
  discriminator = BatchNormalization()(discriminator)
  discriminator = Activation('relu')(discriminator)

  discriminator = Dense(512, activation="relu")(discriminator)
  out_disc = Dense(1, activation='sigmoid', name='discriminator')(discriminator)

  ### GRL model done###
  GRL_model = Model(inputs=inputs, outputs=[out_continuous, out_disc])
  GRL_model.compile(optimizer='adam',
                    loss=[custom_mse, custom_bce],
                    loss_weights = [1, bce_weight], # equal weighing of two tasks
                    metrics={'predictor': ['mae', tf.keras.metrics.RootMeanSquaredError()], 'discriminator': 'accuracy'})
  ######

  return GRL_model

class XYseq(Sequence):
  def __init__(self, src_swp_fea, src_swp_meta, src_swp_trnidx,
                     tgt_swp_file, tgt_swp_trnidx,
                     batch_size):

    self.src_fea = np.load(src_swp_fea).astype(np.int32)
    #self.src_idx = np.load(src_swp_idx).astype(np.int32)
    self.src_sc = np.load(src_swp_meta)[:, 1]

    self.tgt_fea = np.load(tgt_swp_file).astype(np.int32)

    self.scaler = fea_scaling()
    self.batch_size = batch_size
    self.srcidx_map = src_swp_trnidx
    self.tgtidx_map = tgt_swp_trnidx

    src_size = len(self.srcidx_map)
    tgt_size = len(self.tgtidx_map)

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
    pred_batch_data = self.srcidx_map[pred_batch_idx]

    discrSrc_batch_idx = self.src_discr_idx[idx*(self.batch_size//2):(idx+1)*(self.batch_size//2)]
    discrSrc_batch_data = self.srcidx_map[discrSrc_batch_idx]

    discrTgt_batch_idx = self.tgt_discr_idx[idx*(self.batch_size//2):(idx+1)*(self.batch_size//2)]
    discrTgt_batch_data = self.tgtidx_map[discrTgt_batch_idx]

    batch_X = np.concatenate((self.src_fea[pred_batch_data],
                              self.src_fea[discrSrc_batch_data],
                              self.tgt_fea[discrTgt_batch_data]))
    batch_X = batch_X*self.scaler
    batch_X = np.moveaxis(batch_X, 1, -1) # change format to "channel-last"

    batch_Y_pred = np.concatenate((self.src_sc[pred_batch_data],
                                    -1*np.ones(len(discrSrc_batch_data)),
                                    -1*np.ones(len(discrTgt_batch_data))))

    batch_Y_discr = np.concatenate((-1*np.ones(len(pred_batch_data)),
                                    np.zeros(len(discrSrc_batch_data)),
                                    np.ones(len(discrTgt_batch_data))))

    assert batch_X.shape[0] == self.batch_size*2, batch_X.shape[0]
    assert batch_Y_pred.shape == batch_Y_discr.shape, (batch_Y_pred, batch_Y_discr)

    #print(batch_idx, data_idx, neu_idx, swp_idx, batch_Y) #debugging
    return batch_X, {"predictor":batch_Y_pred, "discriminator":batch_Y_discr}
    
  def on_epoch_end(self):
    np.random.shuffle(self.src_pred_idx)
    np.random.shuffle(self.src_discr_idx)
    np.random.shuffle(self.tgt_discr_idx)
