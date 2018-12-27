from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import base as base_layer
from tensorflow.contrib.rnn import LayerRNNCell, RNNCell
import tensorflow as tf
from wavernn.models.modules import Affine

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class GRUCellApple(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    """

    def __init__(self,
                 num_units_in,
                 num_units_out,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 state_is_tuple=True,
                 name=None,
                 dtype=None):
        """
        :param num_units_in:
        :param num_units_out:
        :param activation:
        :param reuse:
        :param kernel_initializer:
        :param bias_initializer:
        :param state_is_tuple:
        :param name:
        :param dtype:
        """
        super(GRUCellApple, self).__init__(_reuse=reuse, name=name, dtype=dtype)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units_in
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._state_is_tuple = state_is_tuple
        self._units_in = num_units_in
        self._units_out = num_units_out
        self._is_training = False
        self._is_synthesis = True
        self._teacher_force = False

    @property
    def state_size(self):
        return (self._num_units + 1)*2   # size of input_prev is 2, input_prev + state

    @property
    def output_size(self):
        if self._is_synthesis:
            return 2                 # size of output is 2(coarse + fine)
        else:
            return self._units_out * 2

    """
    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            with ops.device(self._device):
                return self._cell.zero_state(batch_size, dtype)
    """

    def set_gta(self, teacher_force):
        self._teacher_force = teacher_force

    def set_status(self, is_training, is_synthesis):
        self._is_synthesis = is_synthesis
        self._is_training = is_training

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)

        if self._teacher_force:
            input_depth = inputs_shape[1].value
        else:
            input_depth = inputs_shape[1].value + 2

        name = 'GRUCellApple'
        weight_shape_external = [input_depth, self._num_units * 3 * 2]
        weight_shape_internal = [self._num_units, self._num_units * 3]
        weight_shape_external_C_input = [1, self._num_units * 3]

        self._weight_external = tf.get_variable(
            name='kernel_ex_{}'.format(name),
            shape=weight_shape_external,
            initializer=tf.contrib.layers.xavier_initializer(),
            dtype=tf.float32
        )

        self._bias_external = tf.get_variable(
            name='bias_ex_{}'.format(name),
            shape=(self._num_units * 3 * 2,),
            initializer=tf.zeros_initializer(),
            dtype=tf.float32)

        self._weight_internal = tf.get_variable(
            name='kernel_in_{}'.format(name),
            shape=weight_shape_internal,
            initializer=tf.contrib.layers.xavier_initializer(),
            dtype=tf.float32
        )

        self._bias_internal = tf.get_variable(
            name='bias_in_{}'.format(name),
            shape=(self._num_units * 3,),
            initializer=tf.zeros_initializer(),
            dtype=tf.float32)

        self._weight_external_C_input = tf.get_variable(
            name='kernel_ex_{}_C_input'.format(name),
            shape=weight_shape_external_C_input,
            initializer=tf.contrib.layers.xavier_initializer(),
            dtype=tf.float32
        )

        self._bias_internal_C_input = tf.get_variable(
            name='bias_in_{}_C_input'.format(name),
            shape=(self._num_units * 3,),
            initializer=tf.zeros_initializer(),
            dtype=tf.float32)

        self._affine_relu_C = Affine(units_in=self._units_in, units_out=self._units_out,
                                     activation=tf.nn.relu, name="Relu-1_C")
        self._affine_C = Affine(units_in=self._units_out, units_out=self._units_out, activation=None, name="Output_C")
        self._affine_relu_F = Affine(units_in=self._units_in, units_out=self._units_out,
                                     activation=tf.nn.relu, name="Relu-1_F")
        self._affine_F = Affine(units_in=self._units_out, units_out=self._units_out, activation=None, name="Output_F")

        self.built = True

    def call(self, mel_input, state_mix):
        """
        Gated recurrent unit (GRU) with nunits cells.
        :param mel_input: coarse_input(cur) + fine_input(cur) + mel(cur) [teacher force] or mel(cur)
        :param state_mix: input(prev) + state
        :return: outputs(coarse and fine outputs), state
        """
        if self._state_is_tuple:
            (input_prev, state) = state_mix
            state_h, state_l = tf.split(state, 2, axis=1)
        else:
            input_prev = tf.slice(state_mix, [0, 0], [-1, 2])
            state = tf.slice(state_mix, [0, 2], [-1, -1])
            #state_h = tf.slice(state_mix, [0, 2], [-1, self._num_units])
            #state_l = tf.slice(state_mix, [0, (2+self._num_units)], [-1, self._num_units])

        if self._teacher_force:
            coarse_input_cur = tf.slice(mel_input, [0, 0], [-1, 1])
            fine_input_cur = tf.slice(mel_input, [0, 1], [-1, 1])
            mel_cur = tf.slice(mel_input, [0, 2], [-1, -1])
        else:
            mel_cur = mel_input

        input_prev_and_mel = tf.concat([input_prev, mel_cur], axis=1)          # 258
        X = tf.matmul(input_prev_and_mel, self._weight_external) + self._bias_external
        Xh, Xl = tf.split(X, 2, axis=1)

        state_h, state_l = tf.split(state, 2, axis=1)
        state = tf.reshape(state, [-1, self._num_units])
        H = tf.matmul(state, self._weight_internal) + self._bias_internal
        Hh, Hl = tf.split(H, 2, axis=0)
        # Hh = tf.matmul(state_h, self._weight_internal) + self._bias_internal   # 2 * 896; 896 * 2688; 2 * 2688
        # Hl = tf.matmul(state_l, self._weight_internal) + self._bias_internal
        Hhr, Hhu, Hhe_ = tf.split(Hh, 3, axis=1)
        Hlr, Hlu, Hle_ = tf.split(Hl, 3, axis=1)
#
        Xhr, Xhu, Xhe_ = tf.split(Xh, 3, axis=1)
        uh = tf.nn.sigmoid(Xhu + Hhu)
        rh = tf.nn.sigmoid(Xhr + Hhr)

        candidate_h = tf.tanh(rh * Hhe_ + Xhe_)
        state_h = state_h * uh + candidate_h * (1 - uh)

        relu_outputs_h = self._affine_relu_C(state_h)
        output_outputs_h = self._affine_C(relu_outputs_h)

        if not self._teacher_force:
            sample_h = tf.multinomial(output_outputs_h, 1, name='multinomial')
            coarse_input_cur = tf.cast(sample_h[0, 0], dtype=tf.int32)
            coarse_input_cur = tf.cast(coarse_input_cur, dtype=tf.float32)

        Xl_C = coarse_input_cur * self._weight_external_C_input + self._bias_internal_C_input
        Xl = Xl + Xl_C
        Xlr, Xlu, Xle_ = tf.split(Xl, 3, axis=1)

        ul = tf.nn.sigmoid(Xlu + Hlu)
        rl = tf.nn.sigmoid(Xlr + Hlr)
        candidate_l = tf.tanh(rl * Hle_ + Xle_)
        state_l = state_l * ul + candidate_l * (1 - ul)

        print("state l", state_l)
        print("state h", state_h)

        relu_outputs_l = self._affine_relu_F(state_l)
        output_outputs_l = self._affine_F(relu_outputs_l)

        if not self._teacher_force:
            sample_l = tf.multinomial(output_outputs_l, 1, name='multinomial')
            fine_input_cur = tf.cast(sample_l[0, 0], dtype=tf.int32)
            fine_input_cur = tf.cast(fine_input_cur, dtype=tf.float32)

        if not self._teacher_force:
            coarse_output_array = tf.reshape(coarse_input_cur, [1, 1])
            fine_output_array = tf.reshape(fine_input_cur, [1, 1])
            coarse_input_cur = tf.truediv(coarse_input_cur, 127.5) - 1.0
            coarse_input_cur = tf.reshape(coarse_input_cur, [1, 1])
            fine_input_cur = tf.truediv(fine_input_cur, 127.5) - 1.0
            fine_input_cur = tf.reshape(fine_input_cur, [1, 1])

        input_cur = tf.concat([coarse_input_cur, fine_input_cur], axis=1)

        if self._state_is_tuple:
            state = tf.concat([state_h, state_l], axis=1)
            state = tf.nn.rnn_cell.LSTMStateTuple(input_cur, state)
        else:
            state = tf.concat([state_h, state_l], axis=1)  # + state_l
            state = array_ops.concat([input_cur, state], axis=1)

        if self._is_synthesis:
            output_array = tf.concat([coarse_output_array, fine_output_array], axis=1)
            return output_array, state
        else:
            output_outputs = tf.concat([output_outputs_h, output_outputs_l], axis=1)
            return output_outputs, state

    def incremental(self, c_encoder, time_length=100, initial_input=None):
        """
        need to be adjusted lynnn
        :param c_encoder:
        :param time_length:
        :param initial_input:
        :return:
        """
        with tf.variable_scope("Model"):
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
                    mel_and_input = tf.concat([inputs, ct], axis=1)

                    H = tf.matmul(state, self._weight_internal) + self._bias_internal
                    X = tf.matmul(mel_and_input, self._weight_external) + self._bias_external

                    Hr, Hu, He_ = tf.split(H, 3, axis=1)
                    Xr, Xu, Xe_ = tf.split(X, 3, axis=1)

                    u = tf.nn.sigmoid(Xu + Hu)
                    r = tf.nn.sigmoid(Xr + Hr)
                    candidate = tf.tanh(r * He_ + Xe_)
                    state = state * u + candidate * (1 - u)

                relu_outputs = self.affine_relu(state)
                ouput_outputs = self.affine(relu_outputs)
                sample_int64 = tf.multinomial(ouput_outputs, 1, name='multinomial')
                sample_int32 = tf.cast(sample_int64[0, 0], tf.int32)
                sample = tf.Print(sample_int32, [times, output_2], message='Generated')
                outputs_ta = outputs_ta.write(times, sample)
                times = times + 1
                return times, state, sample, outputs_ta

            times, state, _, sample_array = tf.while_loop(
                condition,
                body,
                loop_vars=[init_time, init_state, init_input, init_outputs_ta],
                parallel_iterations=10,
                swap_memory=self._hparams.swap_with_cpu,
                name='while')

        sample_array = sample_array.stack()
        return sample_array, state


class EncoderGRU:
    """Wave Encoder Cell
    """
    def __init__(self, hparams, teacher_force=False, name=None):
        """
        :param hparams:
        :param apples:
        :param name:
        """
        super(EncoderGRU, self).__init__()
        self.name = 'enc_GRU' if name is None else name
        self.num_units = hparams.num_units
        self.quantize_channels = hparams.quantize_channels

        self._cell = GRUCellApple(num_units_in=hparams.num_units, num_units_out=hparams.num_channels,
                                  state_is_tuple=False, name="GRUCellApples")
        self._teacher_force = teacher_force
        self._is_synthesis = False

    def get_gta(self):
        return self._teacher_force

    def set_synthesis(self, is_synthesis):
        self._is_synthesis = is_synthesis

    def __call__(self, inputs, is_training=False, input_lengths=None):
        with tf.variable_scope(self.name):
            self._cell.set_gta(self._teacher_force)
            self._cell.set_status(is_training=is_training, is_synthesis=self._is_synthesis)
            #if self._is_synthesis:
            #    outputs, state = self._cell.incremental(c_encoder=inputs, time_length=input_lengths)
            #else:
            # to be better
            # actually we can also use the code above to generate the wave
            batch_size = tf.shape(inputs)[0]
            init_state = tf.zeros([batch_size, 2*(self.num_units+1)], dtype=tf.float32)

            outputs, state = tf.nn.dynamic_rnn(
                self._cell,
                inputs,
                sequence_length=input_lengths,
                initial_state=init_state,
                dtype=tf.float32,
                swap_memory=True)

            return outputs


class WaveRNNCell(RNNCell):
    def __init__(self, encoder_layers, gru_cell_layer):
        """
        :param encoder_layers:
        :param gru_cell_layer:
        """
        super(WaveRNNCell, self).__init__()
        # Initialize encoder layers
        self._encoder = encoder_layers
        self._cell = gru_cell_layer
        self._quantize_channels = gru_cell_layer.quantize_channels
        self._feature_channels = encoder_layers._hparams.num_channels
        self._is_synthesis = False

    def set_status(self, is_synthesis):
        self._is_synthesis = is_synthesis

    def __call__(self, inputs, mel_c, is_training=False, input_lengths=None):
        # Pass input sequence through a encoder layers
        enc_output = self._encoder(mel_c, is_training=is_training)
        batch_size = tf.shape(mel_c)[0]
        if inputs is None:
            self._is_synthesis = True
            self._cell.set_synthesis(is_synthesis=True)
        else:
            self._is_synthesis = False
            self._cell.set_synthesis(is_synthesis=False)
            if self._cell.get_gta():
                inputs_zero = tf.zeros([batch_size, 1, tf.shape(inputs)[-1]], dtype=tf.float32)
                inputs_pad_left_zero = array_ops.concat([inputs_zero, inputs], axis=1)
                inputs_right_shift = inputs_pad_left_zero[:, :-1, :]
                # [B T C] C = feature_channels + 2
                enc_output = array_ops.concat([inputs_right_shift, enc_output], axis=-1)
                #channels = self._feature_channels + 2
                #enc_output = array_ops.concat([inputs, enc_output], axis=-1)
                enc_output = tf.reshape(enc_output, [batch_size, -1, self._feature_channels+2])

        # Extract hidden representation from encoder gru cells
        gru_output = self._cell(enc_output, input_lengths=input_lengths, is_training=is_training)

        if self._is_synthesis:
            gru_output = tf.reshape(gru_output, [batch_size, -1, 2])
            sample_array = tf.cast(gru_output, dtype=tf.int32)
            return sample_array
        else:
            gru_output = tf.reshape(gru_output, [batch_size, -1, self._quantize_channels*2])
            return gru_output
