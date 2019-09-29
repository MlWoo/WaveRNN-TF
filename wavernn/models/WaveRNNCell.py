from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.contrib.util import loader
from tensorflow.python.layers import base as base_layer
from tensorflow.contrib.rnn import LayerRNNCell, RNNCell
from tensorflow.contrib.rnn.ops import gen_gru_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.platform import resource_loader
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.framework import ops
import tensorflow as tf
from wavernn.models.modules import Affine
import os
import pdb


_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

TF_PATH  = tf.__path__
print("TF path =====>", TF_PATH)
contrib_rnn_path = os.path.join(TF_PATH[0], "contrib/rnn/python/ops")
_gru_ops_so = loader.load_op_library(
    resource_loader.get_path_to_datafile(contrib_rnn_path + "/_gru_ops.so"))


@ops.RegisterGradient("GRUBlockCellMask")
def _GRUBlockCellMaskGrad(op, *grad):
  r"""Gradient for GRUBlockCell.

  Args:
    op: Op for which the gradient is defined.
    *grad: Gradients of the optimization function wrt output
      for the Op.

  Returns:
    d_x: Gradients wrt to x
    d_h: Gradients wrt to h
    d_w_ru: Gradients wrt to w_ru
    d_w_c: Gradients wrt to w_c
    d_b_ru: Gradients wrt to b_ru
    d_b_c: Gradients wrt to b_c

  Mathematics behind the Gradients below:
  ```
  d_c_bar = d_h \circ (1-u) \circ (1-c \circ c)
  d_u_bar = d_h \circ (h-c) \circ u \circ (1-u)

  d_r_bar_u_bar = [d_r_bar d_u_bar]

  [d_x_component_1 d_h_prev_component_1] = d_r_bar_u_bar * w_ru^T

  [d_x_component_2 d_h_prevr] = d_c_bar * w_c^T

  d_x = d_x_component_1 + d_x_component_2

  d_h_prev = d_h_prev_component_1 + d_h_prevr \circ r + u
  ```
  Below calculation is performed in the python wrapper for the Gradients
  (not in the gradient kernel.)
  ```
  d_w_ru = x_h_prevr^T * d_c_bar

  d_w_c = x_h_prev^T * d_r_bar_u_bar

  d_b_ru = sum of d_r_bar_u_bar along axis = 0

  d_b_c = sum of d_c_bar along axis = 0
  ```
  """
  x, h_prev, w_ru, w_c, b_ru, b_c = op.inputs
  r, u, c, _ = op.outputs
  _, _, _, d_h = grad

  d_x, d_h_prev, d_c_bar, d_r_bar_u_bar = gen_gru_ops.gru_block_cell_grad(
      x, h_prev, w_ru, w_c, b_ru, b_c, r, u, c, d_h)

  x_h_prev = array_ops.concat([x, h_prev], 1)
  d_w_ru = math_ops.matmul(x_h_prev, d_r_bar_u_bar, transpose_a=True)
  d_b_ru = nn_ops.bias_add_grad(d_r_bar_u_bar)

  x_h_prevr = array_ops.concat([x, h_prev * r], 1)
  d_w_c = math_ops.matmul(x_h_prevr, d_c_bar, transpose_a=True)
  d_b_c = nn_ops.bias_add_grad(d_c_bar)

  return d_x, d_h_prev, d_w_ru, d_w_c, d_b_ru, d_b_c

class GRUBlockCellMask(LayerRNNCell):
  r"""Block GRU cell implementation.

  Deprecated: use GRUBlockCellV2 instead.

  The implementation is based on:  http://arxiv.org/abs/1406.1078
  Computes the GRU cell forward propagation for 1 time step.

  This kernel op implements the following mathematical equations:

  Biases are initialized with:

  * `b_ru` - constant_initializer(1.0)
  * `b_c` - constant_initializer(0.0)

  ```
  x_h_prev = [x, h_prev]

  [r_bar u_bar] = x_h_prev * w_ru + b_ru

  r = sigmoid(r_bar)
  u = sigmoid(u_bar)

  h_prevr = h_prev \circ r

  x_h_prevr = [x h_prevr]

  c_bar = x_h_prevr * w_c + b_c
  c = tanh(c_bar)

  h = (1-u) \circ c + u \circ h_prev
  ```

  """

  @deprecated_args(None, "cell_size is deprecated, use num_units instead",
                   "cell_size")
  def __init__(self,
               num_units=None,
               cell_size=None,
               mask_size=None,
               reuse=None,
               name="gru_cell"):
    """Initialize the Block GRU cell.

    Args:
      num_units: int, The number of units in the GRU cell.
      cell_size: int, The old (deprecated) name for `num_units`.
      reuse: (optional) boolean describing whether to reuse variables in an
        existing scope.  If not `True`, and the existing scope already has the
        given variables, an error is raised.
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.  By default this is "lstm_cell", for variable-name compatibility
        with `tf.nn.rnn_cell.GRUCell`.

    Raises:
      ValueError: if both cell_size and num_units are not None;
        or both are None.
    """
    super(GRUBlockCellMask, self).__init__(_reuse=reuse, name=name)
    if (cell_size is None) == (num_units is None):
      raise ValueError(
          "Exactly one of num_units or cell_size must be provided.")
    if num_units is None:
      num_units = cell_size
    assert num_units % 2 == 0
    self.mask_size = mask_size
    self._cell_size = num_units
    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

  @property
  def state_size(self):
    return self._cell_size

  @property
  def output_size(self):
    return self._cell_size

  def build(self, input_shape):
    # Check if the input size exist.
    input_size = input_shape[1].value
    if input_size is None:
      raise ValueError("Expecting input_size to be set.")

    self._gate_kernel = self.add_variable(
        "w_ru", [input_size + self._cell_size, self._cell_size * 2])
    self._gate_bias = self.add_variable(
        "b_ru", [self._cell_size * 2],
        initializer=init_ops.constant_initializer(1.0))
    self._candidate_kernel = self.add_variable(
        "w_c", [input_size + self._cell_size, self._cell_size])
    self._candidate_bias = self.add_variable(
        "b_c", [self._cell_size],
        initializer=init_ops.constant_initializer(0.0))

    __h_mask = tf.zeros([self.mask_size, self._cell_size // 2])
    __h_pass = tf.ones(
      [input_size + self._cell_size - self.mask_size, self._cell_size // 2])
    __l_pass = tf.ones([input_size + self._cell_size, self._cell_size // 2])
    _h_mask = tf.concat([__h_mask, __h_pass], axis=0)
    _mask =  tf.concat([_h_mask, __l_pass], axis =1)
    mask = tf.tile(_mask, multiples=[1, 2])
    self.mask_w_ru = tf.stop_gradient(mask)
    self.mask_w_c = tf.stop_gradient(_mask)

    self.built = True

  def call(self, inputs, h_prev):
    """GRU cell."""
    # Check cell_size == state_size from h_prev.
    cell_size = h_prev.get_shape().with_rank(2)[1]
    if cell_size != self._cell_size:
      raise ValueError("Shape of h_prev[1] incorrect: cell_size %i vs %s" %
                       (self._cell_size, cell_size))

    _gru_block_cell = gen_gru_ops.gru_block_cell  # pylint: disable=invalid-name
    real_w_ru = self.mask_w_ru * self._gate_kernel
    real_w_c = self.mask_w_c * self._candidate_kernel

    _, _, _, new_h = _gru_block_cell(
        x=inputs,
        h_prev=h_prev,
        w_ru=real_w_ru,
        w_c=real_w_c,
        b_ru=self._gate_bias,
        b_c=self._candidate_bias)
    return new_h, new_h


class WaveGRU:
  def __init__(self, hparams, name=None):
    super(WaveGRU, self).__init__()
    self.name = 'Wave_GRU' if name is None else name
    self._num_units = hparams.num_units
    self._mask_size = hparams.mask_size
    self._quantize_channels = hparams.quantize_channels
    self._cell = GRUBlockCellMask(num_units=hparams.num_units, mask_size=hparams.mask_size)

    assert hparams.num_units % 2 == 0
    self._units_in = hparams.num_units
    self._units_out = hparams.num_channels


    self._affine_relu_h = Affine(units_in=self._units_in // 2, units_out=self._units_out, activation=tf.nn.relu,
                                 name="Relu-1_C")
    self._affine_h = Affine(units_in=self._units_out, units_out=self._units_out, activation=None, name="Output_C")
    self._affine_relu_l = Affine(units_in=self._units_in // 2, units_out=self._units_out, activation=tf.nn.relu,
                                 name="Relu-1_F")
    self._affine_l = Affine(units_in=self._units_out, units_out=self._units_out, activation=None, name="Output_F")

  def __call__(self, inputs, input_lengths):
    with tf.variable_scope(self.name):
      batch_size = tf.shape(inputs)[0]
      init_state = tf.zeros([batch_size, self._num_units], dtype=tf.float32)
      # B x T x C
      outputs, state = tf.nn.dynamic_rnn(
        self._cell,
        inputs,
        sequence_length=input_lengths,
        initial_state=init_state,
        dtype=tf.float32,
        swap_memory=True)
      state_h,  state_l = tf.split(outputs, num_or_size_splits=2, axis=-1)
      relu_outputs_h = self._affine_relu_h(state_h)
      outputs_h = self._affine_h(relu_outputs_h)
      relu_outputs_l = self._affine_relu_l(state_l)
      outputs_l = self._affine_l(relu_outputs_l)
      return outputs_h, outputs_l

  def incremental_8bits(self, c_encoder, time_length=100, initial_input=None):
    """
    need to be adjusted lynnn
    :param c_encoder:
    :param time_length:
    :param initial_input:
    :return:
    """
    with tf.variable_scope(self.name):
      # prepare weight mask matrix to synthesis the wave
      gate_kernel_r_h = self._cell._gate_kernel[:self._num_units // 2, self._mask_size * 2:]
      gate_kernel_u_h = self._cell._gate_kernel[(self._num_units // 2 + self._num_units):, self._mask_size * 2:]
      gate_bias_r_h = self._cell._gate_bias[:self._num_units // 2]
      gate_bias_u_h = self._cell._gate_bias[(self._num_units // 2 + self._num_units):]

      candiate_kernel_h = self._cell._candiate_kernel[:self._num_units // 2, self._mask_size * 2:]
      candiate_bias_h = self._cell._candiate_bias[(self._num_units // 2 + self._num_units):]

      gate_kernel_ru_h = tf.concat([gate_kernel_r_h, gate_kernel_u_h], axis=1)
      gate_bias_ru_h = tf.concat([gate_bias_r_h, gate_bias_u_h], axis=0)

      assert c_encoder is not None
      c_encoder_length = tf.shape(c_encoder)[1]
      if time_length is None:
        time_length = c_encoder_length
      init_time = tf.constant(0, dtype=tf.int32)
      if initial_input is None:
        init_input = tf.constant(128, dtype=tf.int32)

      init_state = tf.zeros([1, self.num_units], dtype=tf.float32)
      init_outputs_ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False)

      def condition(times, hidden_state, unused_current_input, outputs_ta):
        return tf.less(times, time_length)

      def body(times, state, current_input, outputs_ta):
        with tf.variable_scope('Scale-Input'):
          inputs = tf.cast(current_input, dtype=tf.float32)
          inputs = tf.truediv(inputs, 127.5) - 1.0
          inputs = tf.reshape(inputs, [1, 1])
        with tf.variable_scope(self.scope):
          ct = c_encoder[:, times, :]
          input_mel_h_prev = tf.concat([inputs, ct, state], axis=1)

          gate_ru_h = tf.matmul(input_mel_h_prev, gate_kernel_ru_h) + gate_bias_ru_h
          gate_ru_h_gate = tf.nn.sigmoid(gate_ru_h)
          r, u = tf.split(gate_ru_h_gate, 2, axis=0)

          state_r = r * state
          input_mel_h_prev_r = tf.concat([inputs, ct, state_r], axis=1)
          candiate_h = tf.matmul(input_mel_h_prev_r, candiate_kernel_h) + candiate_bias_h
          candidate = tf.tanh(candiate_h)

          state = state * u + candidate * (1 - u)

        relu_outputs = self._affine_relu_h(state)
        ouput_outputs = self._affine_h(relu_outputs)
        sample_int64 = tf.multinomial(ouput_outputs, 1, name='multinomial')
        sample_int32 = tf.cast(sample_int64[0, 0], tf.int32)
        sample = tf.Print(sample_int32, [times, sample_int32], message='Generated')
        outputs_ta = outputs_ta.write(times, sample)
        times = times + 1
        return times, state, sample, outputs_ta

      times, state, _, sample_array = tf.while_loop(
        condition,
        body,
        loop_vars=[init_time, init_state, init_input, init_outputs_ta],
        parallel_iterations=1,
        swap_memory=self._hparams.swap_with_cpu,
        name='while')

      sample_array = sample_array.stack()
      return sample_array


  def incremental_16bits(self, c_encoder, time_length=100, initial_input=None):
    """
    need to be adjusted lynnn
    :param c_encoder:
    :param time_length:
    :param initial_input:
    :return:
    """
    with tf.variable_scope(self.name):
      assert c_encoder is not None
      batch_size, c_encoder_length, input_size = tf.shape(c_encoder).aslist()

      # prepare weight mask matrix to synthesis the wave
      gate_kernel_ru = self._cell._gate_kernel[:, self._mask_size:]
      gate_bias_ru = self._cell._gate_bias[:, self._mask_size:]


      candiate_kernel_x = self._cell._candiate_kernel[:, self._mask_size:(self._mask_size * 2 + input_size)]
      candiate_kernel_r = self._cell._candiate_kernel[:, (self._mask_size * 2 + input_size):]

      candiate_kernel_x_supplement = tf.zeros([self._num_units, self._num_units])
      candiate_kernel_r_supplement = tf.zeros([self._num_units, (self._mask_size + input_size)])

      candiate_kernel_x_all = tf.concat([candiate_kernel_x, candiate_kernel_x_supplement], axis=0)
      candiate_kernel_r_all = tf.concat([candiate_kernel_r_supplement, candiate_kernel_r], axis=0)
      candiate_kernel_all = tf.concat([candiate_kernel_x_all, candiate_kernel_r_all], axis=1)

      kernel_all = tf.concat([gate_kernel_ru, candiate_kernel_all], axis=1)

      kernel_c_current_r = self._cell._gate_kernel[self._num_units // 2 : self._num_units, :self._mask_size]
      kernel_c_current_u = self._cell._gate_kernel[(self._num_units // 2 + self._num_units), :self._mask_size]
      kernel_c_current_c = self._cell._candiate_kernel[self._num_units // 2 : self._num_units, :self._mask_size]

      kernel_c_current_ruc = tf.concat([kernel_c_current_r, kernel_c_current_u, kernel_c_current_c], axis=0)

      candiate_bias_h = self._cell._candidate_bias[:self._num_units // 2]
      candiate_bias_l = self._cell._candidate_bias[self._num_units // 2:]

      if time_length is None:
        time_length = c_encoder_length
      init_time = tf.constant(0, dtype=tf.int32)
      if initial_input is None:
        init_input = tf.constant(128, dtype=tf.int32)

      init_state = tf.zeros([1, self.num_units], dtype=tf.float32)
      init_outputs_ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False)

      def condition(times, hidden_state, unused_current_input, outputs_ta):
        return tf.less(times, time_length)

      def body(times, state, current_input, outputs_ta):
        (state_h, state_l) = state
        with tf.variable_scope('Scale-Input'):
          inputs = tf.cast(current_input, dtype=tf.float32)
          inputs = tf.truediv(inputs, 127.5) - 1.0
          inputs = tf.reshape(inputs, [1, 2])
        with tf.variable_scope(self.scope):
          ct = c_encoder[:, times, :]
          input_mels = tf.concat([inputs, ct], axis=1)

          intermediate = tf.matmul(input_mels, kernel_all)
          ru_intermediate_, c_intermediate__ = tf.split(intermediate, 2, axis=1)
          ru_intermediate = ru_intermediate_ + gate_bias_ru
          r_h_, r_l__, u_h_, u_l__ = tf.split(ru_intermediate, 4, axis=1)
          c_intermediate__x_h, c_intermediate__x_l, c_intermediate__w_h, c_intermediate__w_l = \
            tf.split(c_intermediate__, 4, axis=1)

          r_h = tf.nn.sigmoid(r_h_)
          c_h = tf.tanh(r_h*c_intermediate__w_h + c_intermediate__x_h + candiate_bias_h)
          u_h = tf.nn.sigmoid(u_h_)
          state_h = state_h * u_h + c_h * (1 - u_h)

          relu_outputs_h = self._affine_relu_h(state_h)
          outputs_h = self._affine_h(relu_outputs_h)
          sample_int64_h = tf.multinomial(outputs_h, 1, name='multinomial')
          sample_int32_h = tf.cast(sample_int64_h[0, 0], tf.int32)
          sample_h = tf.Print(sample_int32_h, [times, sample_int32_h], message='Generated')
          outputs_current_h = tf.reshape(sample_h, [1, 1])

          ruc_current_h_ = kernel_c_current_ruc * outputs_current_h
          r_current_h_, u_current_h_, c_current_h_ = tf.split(ruc_current_h_, num_or_size_splits=3, axis=0)
          r_l_ = r_l__ + r_current_h_
          u_l_ = u_l__ + u_current_h_
          r_l = tf.nn.sigmoid(r_l_)
          c_l = tf.nn.tanh(r_l*(c_intermediate__w_l + c_current_h_) + c_intermediate__x_l + candiate_bias_l)
          u_l = tf.nn.sigmoid(u_l_)
          state_l = state_l * u_l + c_l * (1 - u_l)

          relu_outputs_l = self._affine_relu_h(state_l)
          outputs_l = self._affine_h(relu_outputs_l)
          sample_int64_l = tf.multinomial(outputs_l, 1, name='multinomial')
          sample_int32_l = tf.cast(sample_int64_l[0, 0], tf.int32)
          sample_l = tf.Print(sample_int32_h, [times, sample_int32_l], message='Generated')

        state = tf.nn.rnn_cell.LSTMStateTuple(state_h, state_l)
        sample = tf.concat([sample_int32_l, sample_int32_h], axis=0)
        times = times + 1
        return times, state, sample, outputs_ta

      times, state, _, sample_array = tf.while_loop(
        condition,
        body,
        loop_vars=[init_time, init_state, init_input, init_outputs_ta],
        parallel_iterations=1,
        swap_memory=self._hparams.swap_with_cpu,
        name='while')

      sample_array = sample_array.stack()
      return sample_array

