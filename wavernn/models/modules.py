import tensorflow as tf
import numpy as np
import math
ln256 = math.log(256)


def mu_law_expanding(x):
  """
  x:[-1, 1]
  return y:[-1, 1]
  """
  with tf.variable_scope("MuLawExpanding"):
    return tf.truediv((tf.exp(tf.abs(x)*ln256)-1.0), 255.0)*tf.sign(x)


def dequantize(x):
  """
  :param x: tf.int32 [0, 255]
  :return: tf.float32 [-1, 1]
  """
  with tf.variable_scope("Dequantize"):
    x = tf.cast(x, tf.float32)
    return tf.truediv(2*(x + 0.5), 256.0)-1.0


def sequence_mask(input_lengths, max_len=None, expand=True):
  if max_len is None:
    max_len = tf.reduce_max(input_lengths)

  if expand:
    return tf.expand_dims(tf.sequence_mask(input_lengths, max_len, dtype=tf.float32), axis=-1)
  return tf.sequence_mask(input_lengths, max_len, dtype=tf.float32)


def MaskedCrossEntropyLoss(outputs, targets, lengths=None, mask=None, max_len=None, mask_enabled=False):
  # One hot encode targets (outputs.shape[-1] = hparams.quantize_channels)
  targets = tf.cast(targets, dtype=tf.int32)
  targets_ = tf.one_hot(targets, depth=tf.shape(outputs)[-1])
  with tf.control_dependencies([tf.assert_equal(tf.shape(outputs), tf.shape(targets_))]):
    losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=targets_)

  if mask_enabled:
    if lengths is None and mask is None:
      raise RuntimeError('Please provide either lengths or mask')

    # [batch_size, time_length]
    if mask is None:
      mask = sequence_mask(lengths, max_len, False)

    with tf.control_dependencies([tf.assert_equal(tf.shape(mask), tf.shape(losses))]):
      masked_loss = losses * mask
    return tf.reduce_sum(masked_loss) / tf.count_nonzero(masked_loss, dtype=tf.float32)

  else:
    return tf.reduce_mean(losses)



def DiscretizedMixtureLogisticLoss(outputs, targets, hparams, lengths=None, mask=None, max_len=None):
  if lengths is None and mask is None:
    raise RuntimeError('Please provide either lengths or mask')

  # [batch_size, time_length, 1]
  if mask is None:
    mask = sequence_mask(lengths, max_len, True)

  # [batch_size, time_length, dimension]
  ones = tf.ones([tf.shape(mask)[0], tf.shape(mask)[1], tf.shape(targets)[-1]], tf.float32)
  mask_ = mask * ones

  losses = discretized_mix_logistic_loss(outputs, targets, num_classes=hparams.quantize_channels,
    log_scale_min=hparams.log_scale_min, reduce=False)

  with tf.control_dependencies([tf.assert_equal(tf.shape(losses), tf.shape(targets))]):
    return tf.reduce_sum(losses * mask_) / tf.reduce_sum(mask_)


class Affine:
  """
  similar to the tf.nn.Dense
  y = Wx + b
  or
  y = relu(Wx + b)
  """
  def __init__(self, units_in, units_out, use_bias=True, activation=None, name=None):
    """
    :param units_in:
    :param units_out:
    :param use_bias:
    :param activation: relu or None
    :param name:
    """
    super(Affine, self).__init__()
    name = 'Affine' if name is None else name
    with tf.variable_scope(name) as scope:
      self._units_in = units_in
      self._units_out = units_out
      self._use_bias = use_bias

      weight_shape = [units_in, units_out]
      self._weight = tf.get_variable(
        name='kernel_{}'.format(name),
        shape=weight_shape,
        dtype=tf.float32)

      if use_bias:
        self._bias = tf.get_variable(
          name='bias_{}'.format(name),
          shape=(units_out,),
          initializer=tf.zeros_initializer(),
          dtype=tf.float32)

      self._activation = activation
      self.scope = scope

  def __call__(self, inputs):
    #batch_size = tf.shape(inputs)[0]
    with tf.variable_scope(self.scope):
      shape = tf.shape(inputs)
      shape = tf.strided_slice(shape, [0], [-1], [1])
      inputs = tf.reshape(inputs, [-1, self._units_in])
      output = tf.matmul(inputs, self._weight)
      if self._use_bias:
        output = output + self._bias
      shape = tf.concat([shape, [self._units_out]], axis=0)
      output = tf.reshape(output, shape)
      if self._activation is not None:
        output = self._activation(output)
      return output


class BNReluConv1D:
  """
  The important part of encoder
  """
  def __init__(self, kernel_size, channels_in, channels_out, use_bias=True, padding='VALID', idx=0, name=None):
    """
    :param kernel_size:
    :param channels_in:
    :param channels_out:
    :param use_bias:
    :param padding:
    :param name:
    """
    super(BNReluConv1D, self).__init__()
    name = 'BN_Relu_Conv1D' + str(idx) if name is None else name
    with tf.variable_scope(name) as scope:
      # Create variables
      kernel_shape = (kernel_size, channels_in, channels_out)

      self._kernel = tf.get_variable(
        name='kernel_{}'.format(name),
        shape=kernel_shape,
        dtype=tf.float32)

      if use_bias:
        self._bias = tf.get_variable(
          name='bias_{}'.format(name),
          shape=(channels_out,),
          initializer=tf.zeros_initializer(),
          dtype=tf.float32)

      self._channels_out = channels_out
      self._channels_in = channels_in
      self._padding = padding
      self._use_bias = use_bias
      self._epsilon = 0.000009999999747378752
      self.scope = scope

  def __call__(self, inputs, is_training):
    with tf.variable_scope(self.scope):
      bn_output = tf.contrib.layers.batch_norm(inputs, scale=True, center=True, trainable=True,
                                               is_training=is_training, epsilon=self._epsilon)
      relu_output = tf.nn.relu(bn_output)
      conv1d_output = tf.nn.conv1d(relu_output, self._kernel, stride=1, padding=self._padding, data_format='NWC')

      if self._use_bias:
        conv1d_output = tf.nn.bias_add(conv1d_output, self._bias)

      return conv1d_output


class UpSamplingByRepetition:
  """
  adjust the resolution of mels to that of time
  """
  def __init__(self, multiples=400, channels=80, name=None):
    """
    :param multiples: Repetition multiples
    :param channels: mels channels
    :param name:
    """
    super(UpSamplingByRepetition, self).__init__()
    name = 'UpSamplingByRepetition' if name is None else name
    with tf.variable_scope(name) as scope:
      self._multiples = multiples
      self._channels = channels
      self.scope = scope

  def __call__(self, inputs):
    with tf.variable_scope(self.scope):
      multi_inputs = tf.tile(inputs, [1, 1, self._multiples])
      time_length = tf.shape(inputs)[1] * self._multiples
      multi_inputs = tf.reshape(multi_inputs, [-1, time_length, self._channels])
      return multi_inputs


class Encoder:
  """
  to encode the mels features before it is fed into the WaveRNN networks
  """
  def __init__(self, hparams, GRUCell=False, name=None):
    super(Encoder, self).__init__()
    name = 'Encoder' if name is None else name
    with tf.variable_scope(name) as scope:
      self._hparams = hparams
      self._mel_affine = Affine(units_in=hparams.num_mels, units_out=hparams.num_channels)

      # Residual convolutions
      self._conv_layers = []
      for i in range(hparams.num_layers):
        self._conv_layers.append(BNReluConv1D(kernel_size=hparams.kernel_size, channels_in=hparams.num_channels,
                                              channels_out=hparams.num_channels, idx=i))

      self._upsampling = UpSamplingByRepetition(multiples=hparams.multiples, channels=hparams.num_channels)
      self._GRUCell = GRUCell
      self.scope = scope

  def __call__(self, inputs, is_training=False):
    with tf.variable_scope(self.scope):
      mel_spectrums = tf.add(inputs, self._hparams.mel_bias)
      mel_spectrums = tf.truediv(mel_spectrums, self._hparams.mel_scale)

      mel_spec_proj = self._mel_affine(mel_spectrums)
      mel_spec_tanh = tf.tanh(mel_spec_proj)
      bn_conv_ouput = mel_spec_tanh
      for layer in self._conv_layers:
        bn_conv_ouput = layer(bn_conv_ouput, is_training)

      with tf.variable_scope('Residual'):
        output_no_pad = mel_spec_tanh[:, self._hparams.padding:-self._hparams.padding, :]
        residual_output = output_no_pad + bn_conv_ouput

      output = self._upsampling(residual_output)
      if self._GRUCell:
        batch_size = tf.shape(output)[0]
        time_length = tf.shape(output)[1]
        num_channels = self._hparams.num_channels
        output = tf.reshape(output, [batch_size, time_length, num_channels])
    return output

  def initialize(self, inputs, is_training=False):
      return self.__call__(inputs, is_training)
