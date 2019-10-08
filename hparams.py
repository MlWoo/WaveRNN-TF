import tensorflow as tf
import numpy as np


# Default hyperparameters
hparams = tf.contrib.training.HParams(
  wavernn_data_random_state = 12345,
  wavernn_random_seed = 5678,
  cell_type = "GRU_STD",  # or GRU_FC
  mask_size = 1,
  num_units = 896,
  num_layers = 3,
  kernel_size =9,
  #padding
  padding = 12, # kernel_size // 2 * num_layers

  input_type = "dual_channels_quantize",
  num_mels = 80, # actually 25
  num_channels = 256,
  multiples = 300, # =hop_length
  sample_rate = 24000,
  scaling = 0.185,
  hop_size = 300,
  win_size = 1200,
  n_fft = 2048, # not less than win_length
  fmin = 0,
  fmax = 8000,

  cin_channels = 80, #Set this to -1 to disable local conditioning, else it must be equal to mel_nums!!
  gin_channels = -1, #Set this to -1 to disable global conditioning, Only used for multi speaker dataset

  max_time_sec = None,
  max_time_steps = 1500, #Max time steps in audio used to train wavernn (decrease to save memory)
  swap_with_cpu = False,

  encoder_conditional_features = True,
  dropout = 0,
  quantize_channels = 256,
  mel_bias = 5.0,
  mel_scale = 10.0,

  training_batch_size = 48,
  testing_batch_size = 12,

  # audio data proprocessing
  rescale=True,  # Whether to rescale audio prior to preprocessing
  rescaling_max=0.8,  # Rescaling value
  trim_silence=True,  # Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
  clip_mels_length=True,  # For cases of OOM (Not really recommended, working on a workaround)
  max_mel_frames=900,  # Only relevant when clip_mels_length = True

  # M-AILABS (and other datasets) trim params
  trim_fft_size=512,
  trim_hop_size=128,
  trim_top_db=60,

  # Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
  # It's preferred to set True to use with https://github.com/r9y9/wavernn_vocoder
  # Does not work if n_ffit is not multiple of hop_size!!
  use_lws=False,
  silence_threshold=2,  # silence threshold used for sound trimming for wavernn preprocessing

  # Limits
  min_level_db=-100,
  ref_level_db=20,

  # Mel and Linear spectrograms normalization/scaling and clipping
  signal_normalization=True,
  allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
  symmetric_mels=True,  # Whether to scale the data to be symmetric around 0
  max_abs_value=4.,  # max absolute value of data. If symmetric, data will be [-max, max] else [0, max]
  loss_mask = False,

  wavernn_batch_size=32,  # batch size used to train wavernn.
  wavernn_test_size=0.0441,  # % of data to keep as test data, if None, wavernn_test_batches must be not None
  wavernn_test_batches=None,  # number of test batches.

  wavernn_learning_rate=1e-3,
  wavernn_adam_beta1=0.9,
  wavernn_adam_beta2=0.999,
  wavernn_adam_epsilon=1e-6,

  wavernn_ema_decay=0.9999,  # decay rate of exponential moving average

  wavernn_dropout=0.05,  # drop rate of wavernn layers
)


def hparams_debug_string():
  values = hparams.values()
  hp = ['  %s: %s' % (name, values[name]) for name in sorted(values) if name != 'sentences']
  return 'Hyperparameters:\n' + '\n'.join(hp)
