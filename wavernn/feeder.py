import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time
import threading
import os
from .util import is_scalar_input, is_mulaw_quantize, inv_mulaw_quantize, waveplot, is_dual_channels_quantize
from infolog import log
from datasets import audio
from keras.utils import np_utils

_batches_per_group = 32
_pad = 0
pow_2_15 = 32768


class Feeder:
  """
      Feeds batches of data into queue in a background thread.
  """
  def __init__(self, coordinator, metadata_filename, base_dir, hparams):
    super(Feeder, self).__init__()

    if hparams.gin_channels > 0:
      raise NotImplementedError('Global conditioning preprocessing has not been added yet, it will be out soon.\
        Thanks for your patience!')

    self._coord = coordinator
    self._hparams = hparams
    self._train_offset = 0
    self._test_offset = 0

    # Base directory of the project (to map files from different locations)
    self._base_dir = base_dir

    # Load metadata
    self._data_dir = os.path.dirname(metadata_filename)
    with open(metadata_filename, 'r') as f:
      self._metadata = [line.strip().split('|') for line in f]

    # Train test split
    if hparams.wavernn_test_size is None:
      assert hparams.wavernn_test_batches is not None

    test_size = (hparams.wavernn_test_size if hparams.wavernn_test_size is not None
                 else hparams.wavernn_test_batches * hparams.wavernn_batch_size)
    indices = np.arange(len(self._metadata))
    train_indices, test_indices = train_test_split(indices, test_size=test_size,
                                                   random_state=hparams.wavernn_data_random_state)

    # Make sure test size is a multiple of batch size else round up
    len_test_indices = _round_up(len(test_indices), hparams.wavernn_batch_size)
    extra_test = test_indices[len_test_indices:]
    test_indices = test_indices[:len_test_indices]
    train_indices = np.concatenate([train_indices, extra_test])

    self._train_meta = list(np.array(self._metadata)[train_indices])
    self._test_meta = list(np.array(self._metadata)[test_indices])

    self.test_steps = len(self._test_meta) // hparams.wavernn_batch_size

    if hparams.wavernn_test_size is None:
      assert hparams.wavernn_test_batches == self.test_steps

    with tf.device('/cpu:0'):
      # Create placeholders for inputs and targets. Don't specify batch size because we want
      # to be able to feed different batch sizes at eval time.
      assert is_dual_channels_quantize(hparams.input_type)
      input_placeholder = tf.placeholder(tf.float32, shape=(None, None, 2), name='audio_inputs')
      local_cond_placeholder = tf.placeholder(tf.float32, shape=(None, None, hparams.num_mels),
                                              name='local_condition_features')
      target_type = tf.float32

      self._placeholders = [
        input_placeholder,
        local_cond_placeholder,
        tf.placeholder(tf.int32, shape=(None, ), name='input_lengths'),
      ]

      queue_types = [tf.float32, target_type, tf.int32]

      # Create queue for buffering data
      queue = tf.FIFOQueue(8, queue_types, name='intput_queue')
      self._enqueue_op = queue.enqueue(self._placeholders)
      variables = queue.dequeue()

      self.inputs = variables[0]
      self.inputs.set_shape(self._placeholders[0].shape)
      self.local_condition_features = variables[1]
      self.local_condition_features.set_shape(self._placeholders[1].shape)
      self.input_lengths = variables[2]
      self.input_lengths.set_shape(self._placeholders[2].shape)

      # Create queue for buffering eval data
      eval_queue = tf.FIFOQueue(1, queue_types, name='eval_queue')
      self._eval_enqueue_op = eval_queue.enqueue(self._placeholders)
      eval_variables = eval_queue.dequeue()

      self.eval_inputs = eval_variables[0]
      self.eval_inputs.set_shape(self._placeholders[0].shape)
      self.eval_local_condition_features = eval_variables[1]
      self.eval_local_condition_features.set_shape(self._placeholders[1].shape)
      self.eval_input_lengths = eval_variables[2]
      self.eval_input_lengths.set_shape(self._placeholders[2].shape)

  def start_threads(self, session):
    self._session = session
    thread = threading.Thread(name='background', target=self._enqueue_next_train_group)
    thread.daemon = True # Thread will close when parent quits
    thread.start()

    thread = threading.Thread(name='background', target=self._enqueue_next_test_group)
    thread.daemon = True # Thread will close when parent quits
    thread.start()

  def _get_test_groups(self):
    meta = self._test_meta[self._test_offset]
    self._test_offset += 1

    audio_quant_file = meta[0]
    mel_file = meta[1]

    input_quant_data = np.load(os.path.join(self._base_dir, 'audio', audio_quant_file))

    local_condition_features = np.load(os.path.join(self._base_dir, 'mels', mel_file))

    return input_quant_data, local_condition_features.T, len(input_quant_data)

  def make_test_batches(self):
    start = time.time()

    # Read one example for evaluation
    n = 1

    # Test on entire test set (one sample at an evaluation step)
    examples = [self._get_test_groups() for i in range(len(self._test_meta))]
    batches = [examples[i: i+n] for i in range(0, len(examples), n)]
    np.random.shuffle(batches)

    log('\nGenerated {} test batches of size {} in {:.3f} sec'.format(len(batches), n, time.time() - start))
    return batches

  def _enqueue_next_train_group(self):
    while not self._coord.should_stop():
      start = time.time()

      # Read a group of examples
      n = self._hparams.wavernn_batch_size
      examples = [self._get_next_example() for i in range(n * _batches_per_group)]

      # Bucket examples base on similiar output length for efficiency
      examples.sort(key=lambda x: x[-1])
      batches = [examples[i: i+n] for i in range(0, len(examples), n)]
      np.random.shuffle(batches)

      log('\nGenerated {} train batches of size {} in {:.3f} sec'.format(len(batches), n, time.time() - start))
      for batch in batches:
        feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch)))
        self._session.run(self._enqueue_op, feed_dict=feed_dict)

  def _enqueue_next_test_group(self):
    test_batches = self.make_test_batches()
    while not self._coord.should_stop():
      for batch in test_batches:
        feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch)))
        self._session.run(self._eval_enqueue_op, feed_dict=feed_dict)

  def _get_next_example(self):
    '''Get a single example (input, output, len_output) from disk
    '''
    if self._train_offset >= len(self._train_meta):
      self._train_offset = 0
      np.random.shuffle(self._train_meta)
    meta = self._train_meta[self._train_offset]
    self._train_offset += 1

    audio_quant_file = meta[0]
    mel_file = meta[1]

    input_quant_data = np.load(os.path.join(self._base_dir, 'audio', audio_quant_file))
    local_condition_features = np.load(os.path.join(self._base_dir, 'mels', mel_file))

    return input_quant_data, local_condition_features.T, len(input_quant_data)

  def _prepare_batch(self, batch):
    np.random.shuffle(batch)

    # Limit time steps to save GPU Memory usage
    max_time_steps = self._limit_time()
    # Adjust time resolution for upsampling
    batch = self._adjust_time_resolution(batch, max_time_steps)

    # time lengths
    assert np.shape(batch[0][0])[1] == 2
    input_lengths = [len(x[0]) for x in batch]
    max_input_length = max(input_lengths)

    inputs = self._prepare_quant_inputs([x[0] for x in batch], max_input_length)
    local_condition_features = self._prepare_local_conditions([x[1] for x in batch])

    return inputs, local_condition_features, input_lengths

  def _prepare_quant_inputs(self, inputs, maxlen):
    x_batch = np.stack([_pad_quant_inputs(x.reshape(-1, 2), maxlen) for x in inputs]).astype(np.float32)
    assert len(x_batch.shape) == 3
    return x_batch

  def _prepare_local_conditions(self, c_features):
    maxlen = max([len(x) for x in c_features])
    c_batch = np.stack([_pad_local_cond(x, maxlen) for x in c_features]).astype(np.float32)
    assert len(c_batch.shape) == 3
    return c_batch

  def _limit_time(self):
    '''Limit time resolution to save GPU memory.
    '''
    if self._hparams.max_time_sec is not None:
      return int(self._hparams.max_time_sec * self._hparams.sample_rate)
    elif self._hparams.max_time_steps is not None:
      return self._hparams.max_time_steps
    else:
      return None

  def _adjust_time_resolution(self, batch, max_time_steps):
    '''
    Adjust time resolution between audio and local condition
    '''
    new_batch = []
    for b in batch:
      x, c, l = b
      self._assert_ready_for_upsample(x, c)
      if max_time_steps is not None:
        max_steps = _ensure_divisible(max_time_steps, audio.get_hop_size(self._hparams), True)
        if len(x) > max_time_steps:
          max_time_frames = max_steps // audio.get_hop_size(self._hparams)
          start = np.random.randint(0, len(c) - max_time_frames)
          time_start = start * audio.get_hop_size(self._hparams)
          x = x[time_start: time_start + max_time_frames * audio.get_hop_size(self._hparams)]
          c = c[start: start + max_time_frames, :]
          self._assert_ready_for_upsample(x, c)
        elif len(x) < max_time_steps:
          print("warning: the audio input will be padding")

      new_batch.append((x, c, l))
    return new_batch

  def _assert_ready_for_upsample(self, x, c):
    assert len(x) % len(c) == 0
    assert len(x) // len(c) == audio.get_hop_size(self._hparams)


def _pad_local_cond(x, maxlen):
  return np.pad(x, [(12, maxlen - len(x)+12), (0, 0)], mode='constant', constant_values=_pad)


def _pad_quant_inputs(x, maxlen):
  _pad_h = 128
  _pad_l = 0
  inputs_h = np.pad(x[:, 0], [(0, maxlen - len(x))], mode='constant', constant_values=_pad_h)
  inputs_l = np.pad(x[:, 1], [(0, maxlen - len(x))], mode='constant', constant_values=_pad_l)
  return np.stack([inputs_l, inputs_h], axis=1)


def _round_up(x, multiple):
  remainder = x % multiple
  return x if remainder == 0 else x + multiple - remainder


def _ensure_divisible(length, divisible_by=256, lower=True):
  if length % divisible_by == 0:
    return length
  if lower:
    return length - length % divisible_by
  else:
    return length + (divisible_by - length % divisible_by)
