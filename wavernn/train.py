import argparse
import sys
import os
from datetime import datetime
import time
import librosa

from wavernn.models import create_model
from wavernn.feeder import Feeder
from wavernn.util import ValueWindow, combine_signal
import numpy as np 
from scipy.io import wavfile
import tensorflow as tf
from . import util

from hparams import hparams_debug_string
import infolog

log = infolog.log


def add_train_stats(model):
  with tf.variable_scope('stats') as scope:
    #tf.summary.histogram('wav_outputs', model.output_coarse)
    #tf.summary.histogram('wav_targets', model.targets_coarse)
    #tf.summary.histogram('wav_outputs', model.output_fine)
    #tf.summary.histogram('wav_targets', model.targets_fine)
    tf.summary.scalar('loss', model.loss)
    return tf.summary.merge_all()


def add_test_stats(summary_writer, step, eval_loss):
  values = [
  tf.Summary.Value(tag='model_1/loss/truediv:0'),
  ]
  test_summary = tf.Summary(value=values)
  summary_writer.add_summary(test_summary, step)


def create_shadow_saver(model, global_step=None):
  '''Load shadow variables of saved model.

  Inspired by: https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

  Can also use: shadow_dict = model.ema.variables_to_restore()
  '''
  # Add global step to saved variables to save checkpoints correctly
  shadow_variables = [model.ema.average_name(v) for v in model.variables]
  variables = model.variables

  if global_step is not None:
    shadow_variables += ['global_step']
    variables += [global_step]

  shadow_dict = dict(zip(shadow_variables, variables)) # dict(zip(keys, values)) -> {key1: value1, key2: value2, ...}
  return tf.train.Saver(shadow_dict, max_to_keep=5)


def load_averaged_model(sess, sh_saver, checkpoint_path):
  sh_saver.restore(sess, checkpoint_path)


def eval_step(sess, global_step, model, plot_dir, audio_dir, summary_writer, hparams):
  '''Evaluate model during training.
  Supposes that model variables are averaged.
  '''
  start_time = time.time()
  output_coarse, output_fine, target_combine, loss = sess.run([model.sample_output_coarse, model.sample_output_fine, model.inputs, model.loss])

  target_coarse, target_fine = np.split(target_combine, 2, axis=2)
  target = combine_signal(target_coarse, target_fine)
  #import pdb
  #pdb.set_trace()
  target = np.squeeze(target)
  output = combine_signal(output_coarse, output_fine)
  output = np.squeeze(output)

  duration = time.time() - start_time
  log('Time Evaluation: Generation of {} audio frames took {:.3f} sec ({:.3f} frames/sec)'.format(
      len(target), duration, len(target)/duration))

  pred_wav_path = os.path.join(audio_dir, 'step-{}-pred.wav'.format(global_step))
  target_wav_path = os.path.join(audio_dir, 'step-{}-real.wav'.format(global_step))
  plot_path = os.path.join(plot_dir, 'step-{}-waveplot.png'.format(global_step))

  # Save Audio
  wavfile.write(pred_wav_path, hparams.sample_rate, output)
  wavfile.write(target_wav_path, hparams.sample_rate, target)

  # Save figure
  util.waveplot(plot_path, output, target, model._hparams)
  log('Eval loss for global step {}: {:.3f}'.format(global_step, loss))

  log('Writing eval summary!')
  add_test_stats(summary_writer, global_step, loss)


def save_log(sess, global_step, model, plot_dir, audio_dir, hparams):
  log('\nSaving intermediate states at step {}'.format(global_step))
  idx = 0
  #output, target, length = sess.run([model.output[idx], model.target[idx], model.input_lengths[idx]])
  output_coarse, output_fine, target_combine, length = sess.run([model.sample_output_coarse, model.sample_output_fine,
                                                                 model.inputs[idx], model.input_lengths[idx]])

  target_coarse, target_fine = np.split(target_combine, 2, axis=1)
  target = combine_signal(target_coarse, target_fine)
  target = np.squeeze(target)
  output = combine_signal(output_coarse, output_fine)
  output = np.squeeze(output)

  # mask by length
  output[length:] = 0
  target[length:] = 0

  # Make audio and plot paths
  pred_wav_path = os.path.join(audio_dir, 'step-{}-pred.wav'.format(global_step))
  target_wav_path = os.path.join(audio_dir, 'step-{}-real.wav'.format(global_step))
  plot_path = os.path.join(plot_dir, 'step-{}-waveplot.png'.format(global_step))

  # Save audio
  librosa.output.write_wav(pred_wav_path, output, sr=hparams.sample_rate)
  librosa.output.write_wav(target_wav_path, target, sr=hparams.sample_rate)

  # Save figure

  util.waveplot(plot_path, output, target, hparams)


def save_checkpoint(sess, saver, checkpoint_path, global_step):
  saver.save(sess, checkpoint_path, global_step=global_step)


def model_train_mode(args, feeder, hparams, global_step):
  with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    model_name = None
    model = create_model(model_name or args.model, hparams, teacher_force=True, apples=None, is_development=True)
    # initialize model to train mode
    model.initialize(inputs=feeder.inputs, mels_c=feeder.local_condition_features,
                     input_lengths=feeder.input_lengths, is_synthesis=False, is_training=True)
    model.add_loss()
    model.add_optimizer(global_step)
    stats = add_train_stats(model)
    return model, stats


def model_test_mode(args, feeder, hparams):
  with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    model_name = None
    model = create_model(model_name or args.model, hparams, teacher_force=True, apples=None, is_development=True)
    # initialize model to test mode
    model.initialize(inputs=feeder.eval_inputs, mels_c=feeder.eval_local_condition_features,
                     input_lengths=feeder.eval_input_lengths, is_synthesis=False, is_training=False)
    model.add_loss()
    return model


def train(log_dir, args, hparams):
  save_dir = os.path.join(log_dir, 'wavernn_pretrained/')
  eval_dir = os.path.join(log_dir, 'eval-dir')
  audio_dir = os.path.join(log_dir, 'wavs')
  plot_dir = os.path.join(log_dir, 'plots')
  wav_dir = os.path.join(log_dir, 'wavs')
  eval_audio_dir = os.path.join(eval_dir, 'wavs')
  eval_plot_dir = os.path.join(eval_dir, 'plots')
  checkpoint_path = os.path.join(save_dir, 'wavernn_model.ckpt')
  input_path = os.path.join(args.base_output_dir, args.wavernn_input)
  os.makedirs(save_dir, exist_ok=True)
  os.makedirs(wav_dir, exist_ok=True)
  os.makedirs(audio_dir, exist_ok=True)
  os.makedirs(plot_dir, exist_ok=True)
  os.makedirs(eval_audio_dir, exist_ok=True)
  os.makedirs(eval_plot_dir, exist_ok=True)

  log('Checkpoint_path: {}'.format(checkpoint_path))
  log('Loading training data from: {}'.format(input_path))
  log('Using model: {}'.format(args.model))
  log(hparams_debug_string())

  # Start by setting a seed for repeatability
  tf.set_random_seed(hparams.wavernn_random_seed)

  # Set up data feeder
  coord = tf.train.Coordinator()
  with tf.variable_scope('datafeeder') as scope:
    feeder = Feeder(coord, input_path, os.path.join(args.base_output_dir, args.input_dir), hparams)

  # Set up model
  global_step = tf.Variable(0, name='global_step', trainable=False)
  model, stats = model_train_mode(args, feeder, hparams, global_step)
  eval_model = model_test_mode(args, feeder, hparams)

  # book keeping
  step = 0
  time_window = ValueWindow(100)
  loss_window = ValueWindow(100)
  sh_saver = create_shadow_saver(model, global_step)

  log('WaveRNN training set to a maximum of {} steps'.format(args.wavernn_train_steps))

  # Memory allocation on the memory
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  # Train
  with tf.Session(config=config) as sess:
    try:
      summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
      sess.run(tf.global_variables_initializer())

      # saved model restoring
      if args.restore:
        # Restore saved model if the user requested it, default = True
        try:
          checkpoint_state = tf.train.get_checkpoint_state(save_dir)
          if checkpoint_state and checkpoint_state.model_checkpoint_path:
            log('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path))
            load_averaged_model(sess, sh_saver, checkpoint_state.model_checkpoint_path)
        except tf.errors.OutOfRangeError as e:
          log('Cannot restore checkpoint: {}'.format(e))
      else:
        log('Starting new training!')

      # initializing feeder
      feeder.start_threads(sess)

      # Training loop
      while not coord.should_stop() and step < args.wavernn_train_steps:
        start_time = time.time()
        step, loss, opt = sess.run([global_step, model.loss, model.optimize])
        time_window.append(time.time() - start_time)
        loss_window.append(loss)

        message = 'Step {:7d} [{:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}]'.format(
          step, time_window.average, loss, loss_window.average)
        log(message, end='\r')

        #if loss > 100 or np.isnan(loss):
        if np.isnan(loss):
          log('Loss exploded to {:.5f} at step {:7d}'.format(loss, step))
          raise Exception('Loss exploded')

        if step % args.summary_interval == 0:
          log('\nWriting summary at step {:7d}'.format(step))
          summary_writer.add_summary(sess.run(stats), step)

        if step % args.checkpoint_interval == 0 or step == args.wavernn_train_steps:
          save_log(sess, step, model, plot_dir, audio_dir, hparams=hparams)
          save_checkpoint(sess, sh_saver, checkpoint_path, global_step)

        if step % args.eval_interval == 0:
          log('\nEvaluating at step {}'.format(step))
          eval_step(sess, step, eval_model, eval_plot_dir, eval_audio_dir, summary_writer=summary_writer,
            hparams=model._hparams)

      log('WaveRNN training complete after {} global steps'.format(args.wavernn_train_steps))
      return save_dir

    except Exception as e:
      log('Exiting due to Exception: {}'.format(e))


def wavernn_train(args, log_dir, hparams):
  return train(log_dir, args, hparams)

