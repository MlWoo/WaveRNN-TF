import tensorflow as tf
import numpy as np
from .WaveRNNCell import WaveRNNCell, EncoderGRU
from .modules import Affine, Encoder, dequantize, mu_law_expanding, MaskedCrossEntropyLoss, DiscretizedMixtureLogisticLoss
from wavernn.util import sequence_mask
from wavernn.util import mulaw_quantize, mulaw, is_mulaw, is_mulaw_quantize, is_dual_channels_quantize, combine_signal
from infolog import log

def split_2_channels(input_channels):
    """
    split the inputs to coarse and fine parts
    :param input_channels:
    :return:
    """
    output_channels = []
    for i in range(input_channels.shape[-1]):
        input_channel = input_channels[:, :, i]
        output_channels.append(input_channel)
    return output_channels


def linear_scale(inputs):
    """
    linear scaling [0, 255] to [-1, 1]
    :param inputs: [B, T, C] and C=2
    :return:
    """
    scale = 2.0 / 255.0
    shift = -1.0

    return scale*inputs+shift


class WaveRNN:
    def __init__(self, hparams, teacher_force=False, apples=None, is_development=False,  name=None):
        super(WaveRNN, self).__init__()
        name = 'WaveRNN' if name is None else name
        with tf.variable_scope("Model"):

            with tf.variable_scope(name) as scope:
                self._hparams = hparams
                self.num_units = hparams.num_units

                if is_development:
                    # assert apples is not None
                    self.gru_cell = EncoderGRU(hparams, teacher_force=teacher_force, name="GRUCellApples")

                else:
                    weight_shape_external = [(hparams.num_channels + 1), self.num_units * 3]
                    weight_shape_internal = [self.num_units, self.num_units * 3]
                    self.affine_relu = Affine(units_in=hparams.num_units, units_out=hparams.num_channels, apples=apples,
                                              id='R_W', activation=tf.nn.relu, name="Relu-1")
                    self.affine = Affine(units_in=hparams.num_channels, units_out=hparams.num_channels, apples=apples,
                                         id='O_W', activation=None, name="Output")
                    if apples is None:
                        self.weight_external = tf.get_variable(
                            name='kernel_ex_{}'.format(name),
                            shape=weight_shape_external,
                            dtype=tf.float32
                        )

                        self.bias_external = tf.get_variable(
                            name='bias_ex_{}'.format(name),
                            shape=(self.num_units*3,),
                            initializer=tf.zeros_initializer(),
                            dtype=tf.float32)

                        self.weight_internal = tf.get_variable(
                            name='kernel_in_{}'.format(name),
                            shape=weight_shape_internal,
                            dtype=tf.float32
                        )

                        self.bias_internal = tf.get_variable(
                            name='bias_in_{}'.format(name),
                            shape=(self.num_units*3,),
                            initializer=tf.zeros_initializer(),
                            dtype=tf.float32)
                    else:
                        self.weight_external = tf.get_variable(
                            name='kernel_ex_{}'.format(name),
                            shape=weight_shape_external,
                            initializer=tf.constant_initializer(apples['weights_W']),
                            dtype=tf.float32
                        )

                        self.bias_external = tf.get_variable(
                            name='bias_ex_{}'.format(name),
                            shape=(self.num_units * 3,),
                            initializer=tf.constant_initializer(apples['biases_W']),
                            dtype=tf.float32)

                        self.weight_internal = tf.get_variable(
                            name='kernel_in_{}'.format(name),
                            shape=weight_shape_internal,
                            initializer=tf.constant_initializer(apples['weights_R']),
                            dtype=tf.float32
                        )

                        self.bias_internal = tf.get_variable(
                            name='bias_in_{}'.format(name),
                            shape=(self.num_units * 3,),
                            initializer=tf.constant_initializer(apples['biases_R']),
                            dtype=tf.float32)

                self.scope = scope

        if self.local_conditioning_enabled():
            assert hparams.num_mels == hparams.cin_channels
            assert hparams.padding == hparams.kernel_size // 2 * hparams.num_layers
            
            # encoder mel local condition
            if hparams.cell_type == "GRU_STD":
                self.encoder_local_cond = Encoder(hparams, GRUCell=False)
                # self.gru_cell = create_rnn_cell(num_units=hparams.num_units, dropout=hparams.dropout)

            elif hparams.cell_type == "GRU_FC":
                # gru_cf need to be investigate further
                self.encoder_local_cond = Encoder(hparams)

            if is_development:
                self.wavernn_cell = WaveRNNCell(self.encoder_local_cond, self.gru_cell)

        self._teacher_force = teacher_force
        self.is_development = is_development
        self.apples = apples

    def initialize(self, inputs, mels_c, input_lengths=None, is_synthesis=False, is_training=False):
        """

        :param inputs:
        :param mels_c:
        :param input_lengths:
        :param is_synthesis:
        :param is_training:
        :return:
        """
        assert inputs is not None and mels_c is not None
        self.targets_coarse, self.targets_fine = tf.split(inputs, num_or_size_splits=2, axis=-1)
        input_2_channels = inputs * 2.0 / 255.0 - 1.0

        self._is_training = is_training
        self._is_evaluating = not self._is_training and not is_synthesis

        self.mask = self.get_mask(input_lengths, maxlen=tf.shape(inputs)[1])  # To be used in loss computation
        output = self.wavernn_cell(inputs=input_2_channels, mel_c=mels_c, is_training=self._is_training,
                                   input_lengths=input_lengths)

        if self._is_evaluating or self._is_training:
            output_0 = tf.reshape(output[0], [1, tf.shape(output)[1], self._hparams.quantize_channels*2])
            output_0_h = output_0[0, :, :self._hparams.quantize_channels]
            output_0_l = output_0[0, :, self._hparams.quantize_channels:]
            sample_0_h = tf.multinomial(output_0_h, 1, name='multinomial')
            sample_0_l = tf.multinomial(output_0_l, 1, name='multinomial')

        else:
            output_0 = tf.reshape(output[0], [1, tf.shape(output)[1], 2])  #[-1, 1]
            sample_0_h = output_0[0, :, :self._hparams.quantize_channels]
            sample_0_l = output_0[0, :, self._hparams.quantize_channels:]

        self.sample_output_coarse = sample_0_h
        self.sample_output_fine = sample_0_l

        self.output_coarse, self.output_fine = tf.split(output, num_or_size_splits=2, axis=-1)

        self.inputs = inputs
        self.input_2_channels = input_2_channels
        self.input_lengths = input_lengths
        log('Initialized WaveRNN model. Dimensions (? = dynamic shape): ')
        log('  Train mode:               {}'.format(self._is_training))
        log('  Eval  mode:               {}'.format(self._is_evaluating))
        log('  GTA   mode:               {}'.format(self._teacher_force))

        self.variables = tf.trainable_variables()
        self.ema = tf.train.ExponentialMovingAverage(decay=self._hparams.wavernn_ema_decay)

    def draw_sample(self, prob, length):
        init_time = tf.constant(0, dtype=tf.int32)
        init_outputs_ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False)

        def condition(times, outputs_ta):
            return tf.less(times, length)

        def body(times, outputs_ta):
            sample_int64 = tf.multinomial(prob[:, times, :], 1, name='multinomial')
            sample_int32 = tf.cast(sample_int64[0, 0], tf.int32)
            outputs_ta = outputs_ta.write(times, sample_int32)
            times = times + 1
            return times, outputs_ta

        _, sample_array = tf.while_loop(
            condition,
            body,
            loop_vars=[init_time, init_outputs_ta],
            parallel_iterations=10,
            swap_memory=self._hparams.swap_with_cpu,
            name='while')
        sample_array = sample_array.stack()
        return sample_array


    def add_loss(self):
        """
        :return:
        """
        with tf.variable_scope('loss') as scope:
            print("targets_coarse", self.targets_coarse)
            print("output_coarse", self.output_coarse)
            assert is_dual_channels_quantize(self._hparams.input_type)
            coarse_loss = MaskedCrossEntropyLoss(self.output_coarse[:, :, :], self.targets_coarse[:, :, 0], mask=self.mask)
            fine_loss = MaskedCrossEntropyLoss(self.output_fine[:, :, :], self.targets_fine[:, :, 0], mask=self.mask)
            self.loss = coarse_loss + fine_loss


    def add_optimizer(self, global_step):
        '''Adds optimizer to the graph. Supposes that initialize function has already been called.
        '''
        with tf.variable_scope('optimizer'):
            hp = self._hparams

            # Adam with constant learning rate
            optimizer = tf.train.AdamOptimizer(hp.wavernn_learning_rate, hp.wavernn_adam_beta1,
                                               hp.wavernn_adam_beta2, hp.wavernn_adam_epsilon)

            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            self.gradients = gradients

            # Gradients clipping
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                adam_optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
                                                          global_step=global_step)

        # Add exponential moving average
        # https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
        # Use adam optimization process as a dependency
        with tf.control_dependencies([adam_optimize]):
            # Create the shadow variables and add ops to maintain moving averages
            # Also updates moving averages after each update step
            # This is the optimize call instead of traditional adam_optimize one.
            assert tuple(self.variables) == variables  # Verify all trainable variables are being averaged
            self.optimize = self.ema.apply(variables)



    def incremental(self, c_encoder, initial_input=None):
        with tf.variable_scope("Model"):
            batch_size = 1
            time_length = tf.shape(c_encoder)[1]
            num_channels = self._hparams.num_channels
            self.c_encoder = tf.reshape(c_encoder, [batch_size, time_length, num_channels])

            init_time = tf.constant(0, dtype=tf.int32)
            if initial_input is None:
                init_input = tf.constant(128, dtype=tf.int32)

            init_state = tf.zeros([1, self.num_units], dtype=tf.float32)
            init_outputs_ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False)

            def condition(times, hidden_state, unused_current_input, outputs_ta):
                return tf.less(times, time_length)

            if self.apples is None:
                def body(times, hidden_state, current_input, outputs_ta):
                    with tf.variable_scope('Scale-Input'):
                        input_0 = tf.cast(current_input, dtype=tf.float32)
                        input_0 = tf.truediv(input_0, 127.5) - 1.0
                        input_0 = tf.reshape(input_0, [1, 1])
                    with tf.variable_scope(self.scope):
                        ct = None if self.c_encoder is None else self.c_encoder[:, times, :]
                        mel_and_input = tf.concat([input_0, ct], axis=1)

                        H = tf.matmul(hidden_state, self.weight_internal) + self.bias_internal
                        X = tf.matmul(mel_and_input, self.weight_external) + self.bias_external
                        ''' std
                        '''
                        Hu, Hr, He_ = tf.split(H, 3, axis=1)
                        Xu, Xr, Xe_ = tf.split(X, 3, axis=1)
                        '''compare result
                        Hr, Hu, He_ = tf.split(H, 3, axis=1)
                        Xr, Xu, Xe_ = tf.split(X, 3, axis=1)
                        '''

                        u = tf.nn.sigmoid(Xu + Hu)
                        r = tf.nn.sigmoid(Xr + Hr)
                        candidate = tf.tanh(r * He_ + Xe_)
                        hidden_state = hidden_state * u + candidate * (1 - u)

                    ouput_0 = self.affine_relu(hidden_state)
                    ouput_1 = self.affine(ouput_0)
                    sample = tf.multinomial(ouput_1, 1, name='multinomial')
                    output_2 = tf.cast(sample[0, 0], tf.int32)
                    output = tf.Print(output_2, [times, output_2], message='Generated')
                    outputs_ta = outputs_ta.write(times, output)
                    times = times + 1

                    return (times, hidden_state, output, outputs_ta)

            else:
                def body(times, hidden_state, current_input, outputs_ta):
                    with tf.variable_scope('Scale-Input'):
                        input_0 = tf.cast(current_input, dtype=tf.float32)
                        input_0 = tf.truediv(input_0, 127.5) - 1.0
                        input_0 = tf.reshape(input_0, [1, 1])
                    with tf.variable_scope(self.scope):
                        ct = None if self.c_encoder is None else self.c_encoder[:, times, :]
                        mel_and_input = tf.concat([input_0, ct], axis=1)
                        H = tf.matmul(hidden_state, self.weight_internal) + self.bias_internal
                        X = tf.matmul(mel_and_input, self.weight_external) + self.bias_external

                        ''' std
                        Hu, Hr, He_ = tf.split(H, 3, axis=1)
                        Xu, Xr, Xe_ = tf.split(X, 3, axis=1)
                        '''
                        '''compare result
                        '''
                        Hr, Hu, He_ = tf.split(H, 3, axis=1)
                        Xr, Xu, Xe_ = tf.split(X, 3, axis=1)
                        u = tf.nn.sigmoid(Xu + Hu)
                        r = tf.nn.sigmoid(Xr + Hr)
                        candidate = tf.tanh(r * He_ + Xe_)
                        hidden_state = hidden_state * u + candidate * (1 - u)

                    ouput_0 = self.affine_relu(hidden_state)
                    ouput_1 = self.affine(ouput_0)
                    sample = tf.multinomial(ouput_1, 1, name='multinomial')
                    output_2 = tf.cast(sample[0, 0], tf.int32)
                    output = tf.Print(output_2, [times, output_2], message='Generated')
                    outputs_ta = outputs_ta.write(times, output)
                    times = times + 1

                    return(times, hidden_state, output, outputs_ta)

            _, _, _, sample_array = tf.while_loop(
                condition,
                body,
                loop_vars=[init_time, init_state, init_input, init_outputs_ta],
                parallel_iterations=10,
                swap_memory=self._hparams.swap_with_cpu,
                name='while')

            sample_array = sample_array.stack()
            sample_array = dequantize(sample_array)
            sample_point = mu_law_expanding(sample_array)
            self.output = sample_point


    def local_conditioning_enabled(self):
        return self._hparams.cin_channels > 0

    def get_mask(self, input_lengths, maxlen=None):
        expand = not is_dual_channels_quantize(self._hparams.input_type)
        mask = sequence_mask(input_lengths, max_len=maxlen, expand=expand)
        return mask[:, :]




