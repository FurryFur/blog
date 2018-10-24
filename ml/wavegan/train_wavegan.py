from __future__ import print_function
import pickle
import os
import time
import math

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.training.summary_io import SummaryWriterCache
from six.moves import xrange

from history_buffer import HistoryBuffer
import loader
from wavegan import WaveGANGenerator, WaveGANDiscriminator, avg_downsample
from functools import reduce


"""
  Constants
"""
_FS = 16000
_WINDOW_LEN = 16384
_D_Z = 128


"""
  Trains a WaveGAN
"""
def train(fps, args):
  with tf.name_scope('loader'):
    x, cond_text, _ = loader.get_batch(fps, args.train_batch_size, _WINDOW_LEN, args.data_first_window, conditionals=True, name='batch')
    wrong_audio = loader.get_batch(fps, args.train_batch_size, _WINDOW_LEN, args.data_first_window, conditionals=False, name='wrong_batch')
   # wrong_cond_text, wrong_cond_text_embed = loader.get_batch(fps, args.train_batch_size, _WINDOW_LEN, args.data_first_window, wavs=False, conditionals=True, name='batch')
    
  # Make z vector
  z = tf.random_normal([args.train_batch_size, _D_Z])

  embed = hub.Module('https://tfhub.dev/google/elmo/2', trainable=False, name='embed')
  cond_text_embed = embed(cond_text)

  # Add conditioning input to the model
  args.wavegan_g_kwargs['context_embedding'] = cond_text_embed
  args.wavegan_d_kwargs['context_embedding'] = args.wavegan_g_kwargs['context_embedding']

  lod = tf.placeholder(tf.float32, shape=[])
  
  with tf.variable_scope('G'):
    # Make generator
    G_z, c_kl_loss = WaveGANGenerator(z, lod, train=True, **args.wavegan_g_kwargs)
    if args.wavegan_genr_pp:
      with tf.variable_scope('pp_filt'):
        G_z = tf.layers.conv1d(G_z, 1, args.wavegan_genr_pp_len, use_bias=False, padding='same')
  
  # Summarize
  G_z_rms = tf.sqrt(tf.reduce_mean(tf.square(G_z[:, :, 0]), axis=1))
  x_rms = tf.sqrt(tf.reduce_mean(tf.square(x[:, :, 0]), axis=1))
  x_rms_lod_4 = tf.sqrt(tf.reduce_mean(tf.square(avg_downsample(x)[:, :, 0]), axis=1))
  x_rms_lod_3 = tf.sqrt(tf.reduce_mean(tf.square(avg_downsample(avg_downsample(x))[:, :, 0]), axis=1))
  x_rms_lod_2 = tf.sqrt(tf.reduce_mean(tf.square(avg_downsample(avg_downsample(avg_downsample(x)))[:, :, 0]), axis=1))
  x_rms_lod_1 = tf.sqrt(tf.reduce_mean(tf.square(avg_downsample(avg_downsample(avg_downsample(avg_downsample(x))))[:, :, 0]), axis=1))
  x_rms_lod_0 = tf.sqrt(tf.reduce_mean(tf.square(avg_downsample(avg_downsample(avg_downsample(avg_downsample(avg_downsample(x)))))[:, :, 0]), axis=1))
  tf.summary.histogram('x_rms_batch', x_rms)
  tf.summary.histogram('G_z_rms_batch', G_z_rms)
  tf.summary.scalar('x_rms', tf.reduce_mean(x_rms))
  tf.summary.scalar('x_rms_lod_4', tf.reduce_mean(x_rms_lod_4))
  tf.summary.scalar('x_rms_lod_3', tf.reduce_mean(x_rms_lod_3))
  tf.summary.scalar('x_rms_lod_2', tf.reduce_mean(x_rms_lod_2))
  tf.summary.scalar('x_rms_lod_1', tf.reduce_mean(x_rms_lod_1))
  tf.summary.scalar('x_rms_lod_0', tf.reduce_mean(x_rms_lod_0))
  tf.summary.scalar('G_z_rms', tf.reduce_mean(G_z_rms))
  tf.summary.audio('x', x, _FS, max_outputs=10)
  tf.summary.audio('G_z', G_z, _FS, max_outputs=10)
  tf.summary.text('Conditioning Text', cond_text[:10])

  # with tf.variable_scope('G'):
  #   # Make history buffer
  #   history_buffer = HistoryBuffer(_WINDOW_LEN, args.train_batch_size * 100, args.train_batch_size)

  #   # Select half of batch from history buffer
  #   g_from_history, r_from_history, embeds_from_history = history_buffer.get_from_history_buffer()
  #   new_fake_batch = tf.concat([G_z[:tf.shape(G_z)[0] - tf.shape(g_from_history)[0]], g_from_history], 0) # Use tf.shape to handle case when g_from_history is empty
  #   new_cond_embeds = tf.concat([cond_text_embed[:tf.shape(cond_text_embed)[0] - tf.shape(embeds_from_history)[0]], embeds_from_history], 0)
  #   new_real_batch = tf.concat([x[:tf.shape(x)[0] - tf.shape(r_from_history)[0]], r_from_history], 0)
  #   with tf.control_dependencies([new_fake_batch, new_real_batch, new_cond_embeds]):
  #     with tf.control_dependencies([history_buffer.add_to_history_buffer(G_z, x, cond_text_embed)]):
  #       G_z = tf.identity(new_fake_batch)
  #       x = tf.identity(new_real_batch)
  #       args.wavegan_g_kwargs['context_embedding'] = tf.identity(new_cond_embeds)
  #       args.wavegan_d_kwargs['context_embedding'] = args.wavegan_g_kwargs['context_embedding']
  #   G_z.set_shape([args.train_batch_size, _WINDOW_LEN, 1])
  #   x.set_shape([args.train_batch_size, _WINDOW_LEN, 1])

  G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')

  # Print G summary
  print('-' * 80)
  print('Generator vars')
  nparams = 0
  for v in G_vars:
    v_shape = v.get_shape().as_list()
    v_n = reduce(lambda x, y: x * y, v_shape)
    nparams += v_n
    print('{} ({}): {}'.format(v.get_shape().as_list(), v_n, v.name))
  print('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))

  # Summarize
  # tf.summary.scalar('history_buffer_size', history_buffer.current_size)
  # tf.summary.scalar('g_from_history_size', tf.shape(g_from_history)[0])
  # tf.summary.scalar('r_from_history_size', tf.shape(r_from_history)[0])
  # tf.summary.scalar('embeds_from_history_size', tf.shape(embeds_from_history)[0])
  # tf.summary.audio('G_z_history', g_from_history, _FS, max_outputs=10)
  # tf.summary.audio('x_history', r_from_history, _FS, max_outputs=10)
  tf.summary.audio('wrong_audio', wrong_audio, _FS, max_outputs=10)
  tf.summary.scalar('Conditional Resample - KL-Loss', c_kl_loss)
  # tf.summary.scalar('embed_error_cosine', tf.reduce_sum(tf.multiply(cond_text_embed, expected_embed)) / (tf.norm(cond_text_embed) * tf.norm(expected_embed)))
  # tf.summary.scalar('embed_error_cosine_wrong', tf.reduce_sum(tf.multiply(wrong_cond_text_embed, expected_embed)) / (tf.norm(wrong_cond_text_embed) * tf.norm(expected_embed)))

  # Make real discriminator
  with tf.name_scope('D_x'), tf.variable_scope('D'):
    D_x = WaveGANDiscriminator(x, lod, **args.wavegan_d_kwargs)
  D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')

  # Print D summary
  print('-' * 80)
  print('Discriminator vars')
  nparams = 0
  for v in D_vars:
    v_shape = v.get_shape().as_list()
    v_n = reduce(lambda x, y: x * y, v_shape)
    nparams += v_n
    print('{} ({}): {}'.format(v.get_shape().as_list(), v_n, v.name))
  print('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))
  print('-' * 80)

  # Make fake / wrong discriminator
  with tf.name_scope('D_G_z'), tf.variable_scope('D', reuse=True):
    D_G_z = WaveGANDiscriminator(G_z, lod, **args.wavegan_d_kwargs)
  with tf.name_scope('D_w'), tf.variable_scope('D', reuse=True):
    D_w = WaveGANDiscriminator(wrong_audio, lod, **args.wavegan_d_kwargs)

  # Create loss
  D_clip_weights = None
  if args.wavegan_loss == 'dcgan':
    fake = tf.zeros([args.train_batch_size, 1], dtype=tf.float32)
    real = tf.ones([args.train_batch_size, 1], dtype=tf.float32)

    # Conditional G Loss
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=D_G_z[0],
      labels=real
    ))
    G_loss += c_kl_loss

    # Unconditional G Loss
    if args.use_extra_uncond_loss:
      G_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_G_z[1],
        labels=real
      ))
      G_loss /= 2

    # Conditional D Losses
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=D_G_z[0],
      labels=fake
    ))
    D_loss_wrong = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=D_w[0],
      labels=fake
    ))
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=D_x[0],
      labels=real
    ))

    # Unconditional D Losses
    if args.use_extra_uncond_loss:
      D_loss_fake_uncond = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_G_z[1],
        labels=fake
      ))
      D_loss_wrong_uncond = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_w[1],
        labels=real
      ))
      D_loss_real_uncond = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_x[1],
        labels=real
      ))

      D_loss = D_loss_real + 0.5 * (D_loss_wrong + D_loss_fake) \
             + 0.5 * (D_loss_real_uncond + D_loss_wrong_uncond) + D_loss_fake_uncond
      D_loss /= 2
    else:
      D_loss = D_loss_real + 0.5 * (D_loss_wrong + D_loss_fake)

    # Warmup Conditional Loss
    # D_warmup_loss = D_loss_real + D_loss_wrong
  elif args.wavegan_loss == 'lsgan':
    # Conditional G Loss
    G_loss = tf.reduce_mean((D_G_z[0] - 1.) ** 2)
    G_loss += c_kl_loss

    # Unconditional G Loss
    if args.use_extra_uncond_loss:
      G_loss += tf.reduce_mean((D_G_z[1] - 1.) ** 2)
      G_loss /= 2

    # Conditional D Loss
    D_loss_real = tf.reduce_mean((D_x[0] - 1.) ** 2)
    D_loss_wrong = tf.reduce_mean(D_w[0] ** 2)
    D_loss_fake = tf.reduce_mean(D_G_z[0] ** 2)

    # Unconditional D Loss
    if args.use_extra_uncond_loss:
      D_loss_real_uncond = tf.reduce_mean((D_x[1] - 1.) ** 2)
      D_loss_wrong_uncond = tf.reduce_mean((D_w[1] - 1.) ** 2)
      D_loss_fake_uncond = tf.reduce_mean(D_G_z[1] ** 2)

      D_loss = D_loss_real + 0.5 * (D_loss_wrong + D_loss_fake) \
             + 0.5 * (D_loss_real_uncond + D_loss_wrong_uncond) + D_loss_fake_uncond
      D_loss /= 2
    else:
      D_loss = D_loss_real + 0.5 * (D_loss_wrong + D_loss_fake)

    # Warmup Conditional Loss
    # D_warmup_loss = D_loss_real + D_loss_wrong
  elif args.wavegan_loss == 'wgan':
    # Conditional G Loss
    G_loss = -tf.reduce_mean(D_G_z[0])
    G_loss += c_kl_loss

    # Unconditional G Loss
    if args.use_extra_uncond_loss:
      G_loss += -tf.reduce_mean(D_G_z[1])
      G_loss /= 2

    # Conditional D Loss
    D_loss_real = -tf.reduce_mean(D_x[0])
    D_loss_wrong = tf.reduce_mean(D_w[0])
    D_loss_fake = tf.reduce_mean(D_G_z[0]) 

    # Unconditional D Loss
    if args.use_extra_uncond_loss:
      D_loss_real_uncond = -tf.reduce_mean(D_x[1])
      D_loss_wrong_uncond = -tf.reduce_mean(D_w[1])
      D_loss_fake_uncond = tf.reduce_mean(D_G_z[1])

      D_loss = D_loss_real + 0.5 * (D_loss_wrong + D_loss_fake) \
             + 0.5 * (D_loss_real_uncond + D_loss_wrong_uncond) + D_loss_fake_uncond
      D_loss /= 2
    else:
      D_loss = D_loss_real + 0.5 * (D_loss_wrong + D_loss_fake)

    # Warmup Conditional Loss
    # D_warmup_loss = D_loss_real + D_loss_wrong

    with tf.name_scope('D_clip_weights'):
      clip_ops = []
      for var in D_vars:
        clip_bounds = [-.01, .01]
        clip_ops.append(
          tf.assign(
            var,
            tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
          )
        )
      D_clip_weights = tf.group(*clip_ops)
  elif args.wavegan_loss == 'wgan-gp':
    # Conditional G Loss
    G_loss = -tf.reduce_mean(D_G_z[0])
    G_loss += c_kl_loss

    # Unconditional G Loss
    if args.use_extra_uncond_loss:
      G_loss += -tf.reduce_mean(D_G_z[1])
      G_loss /= 2

    # Conditional D Loss
    D_loss_real = -tf.reduce_mean(D_x[0])
    D_loss_wrong = tf.reduce_mean(D_w[0])
    D_loss_fake = tf.reduce_mean(D_G_z[0]) 

    # Unconditional D Loss
    if args.use_extra_uncond_loss:
      D_loss_real_uncond = -tf.reduce_mean(D_x[1])
      D_loss_wrong_uncond = -tf.reduce_mean(D_w[1])
      D_loss_fake_uncond = tf.reduce_mean(D_G_z[1])

      D_loss = D_loss_real + 0.5 * (D_loss_wrong + D_loss_fake) \
             + 0.5 * (D_loss_real_uncond + D_loss_wrong_uncond) + D_loss_fake_uncond
      D_loss /= 2
    else:
      D_loss = D_loss_real + 0.5 * (D_loss_wrong + D_loss_fake)

    # Warmup Conditional Loss
    # D_warmup_loss = D_loss_real + D_loss_wrong

    # Conditional Gradient Penalty
    alpha = tf.random_uniform(shape=[args.train_batch_size, 1, 1], minval=0., maxval=1.)
    real = x
    fake = tf.concat([G_z[:args.train_batch_size // 2], wrong_audio[:args.train_batch_size // 2]], 0)
    differences = fake - real
    interpolates = real + (alpha * differences)
    with tf.name_scope('D_interp'), tf.variable_scope('D', reuse=True):
      D_interp = WaveGANDiscriminator(interpolates, lod, **args.wavegan_d_kwargs)[0] # Only want conditional output
    gradients = tf.gradients(D_interp, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
    cond_gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)

    # Unconditional Gradient Penalty
    alpha = tf.random_uniform(shape=[args.train_batch_size, 1, 1], minval=0., maxval=1.)
    real = tf.concat([x[:args.train_batch_size // 2], wrong_audio[:args.train_batch_size // 2]], 0)
    fake = G_z
    differences = fake - real
    interpolates = real + (alpha * differences)
    with tf.name_scope('D_interp'), tf.variable_scope('D', reuse=True):
      D_interp = WaveGANDiscriminator(interpolates, lod, **args.wavegan_d_kwargs)[1] # Only want unconditional output
    gradients = tf.gradients(D_interp, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
    uncond_gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)

    # Warmup Gradient Penalty
    # alpha = tf.random_uniform(shape=[args.train_batch_size, 1, 1], minval=0., maxval=1.)
    # real = x
    # fake = wrong_audio
    # differences = fake - real
    # interpolates = real + (alpha * differences)
    # with tf.name_scope('D_interp'), tf.variable_scope('D', reuse=True):
    #   D_interp = WaveGANDiscriminator(interpolates, lod, **args.wavegan_d_kwargs)[0] # Only want conditional output
    # gradients = tf.gradients(D_interp, [interpolates])[0]
    # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
    # warmup_gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)

    gradient_penalty = (cond_gradient_penalty + uncond_gradient_penalty) / 2

    LAMBDA = 10
    D_loss += LAMBDA * gradient_penalty
    # D_warmup_loss += LAMBDA * warmup_gradient_penalty
  else:
    raise NotImplementedError()

  tf.summary.scalar('G_loss', G_loss)
  if (args.wavegan_loss == 'wgan-gp'):
    tf.summary.scalar('Gradient Penalty', LAMBDA * gradient_penalty)
  if (args.wavegan_loss == 'wgan' or args.wavegan_loss == 'wgan-gp'):
    if args.use_extra_uncond_loss:
      tf.summary.scalar('Critic Score - Real Data - Condition Match', -D_loss_real)
      tf.summary.scalar('Critic Score - Fake Data - Condition Match', D_loss_fake)
      tf.summary.scalar('Critic Score - Wrong Data - Condition Match', D_loss_wrong)
      tf.summary.scalar('Critic Score - Real Data', -D_loss_real_uncond)
      tf.summary.scalar('Critic Score - Wrong Data', -D_loss_wrong_uncond)
      tf.summary.scalar('Critic Score - Fake Data', D_loss_fake_uncond)
      tf.summary.scalar('Wasserstein Distance - No Regularization Term',
                        -((D_loss_real + 0.5 * (D_loss_wrong + D_loss_fake) \
                         + 0.5 * (D_loss_real_uncond + D_loss_wrong_uncond) + D_loss_fake_uncond) / 2))
      tf.summary.scalar('Wasserstein Distance - Real-Wrong Only', -(D_loss_real + D_loss_wrong))
      tf.summary.scalar('Wasserstein Distance - Real-Fake Only',
                        -((D_loss_real + D_loss_fake \
                         + D_loss_real_uncond + D_loss_fake_uncond) / 2))
    else:
      tf.summary.scalar('Critic Score - Real Data', -D_loss_real)
      tf.summary.scalar('Critic Score - Wrong Data', D_loss_wrong)
      tf.summary.scalar('Critic Score - Fake Data', D_loss_fake)
      tf.summary.scalar('Wasserstein Distance - No Regularization Term', -(D_loss_real + 0.5 * (D_loss_wrong + D_loss_fake)))
    tf.summary.scalar('Wasserstein Distance - With Regularization Term', -D_loss)
  else:
    if args.use_extra_uncond_loss:
      tf.summary.scalar('D_acc_uncond', 0.5 * ((0.5 * (tf.reduce_mean(tf.sigmoid(D_x[1])) + tf.reduce_mean(tf.sigmoid(D_w[1])))) \
                                             + tf.reduce_mean(1 - tf.sigmoid(D_G_z[1]))))
      tf.summary.scalar('D_acc', 0.5 * (tf.reduce_mean(tf.sigmoid(D_x[0])) \
                                      + 0.5 * (tf.reduce_mean(1 - tf.sigmoid(D_w[0])) + tf.reduce_mean(1 - tf.sigmoid(D_G_z[0])))))
      tf.summary.scalar('D_acc_real_wrong_only', 0.5 * (tf.reduce_mean(tf.sigmoid(D_x[0])) \
                                                      + tf.reduce_mean(1 - tf.sigmoid(D_w[0]))))
      tf.summary.scalar('D_loss_cond_real', D_loss_real)
      tf.summary.scalar('D_loss_uncond_real', D_loss_real_uncond)
      tf.summary.scalar('D_loss_cond_wrong', D_loss_wrong)
      tf.summary.scalar('D_loss_uncond_wrong', D_loss_wrong_uncond)
      tf.summary.scalar('D_loss_cond_fake', D_loss_fake)
      tf.summary.scalar('D_loss_uncond_fake', D_loss_fake_uncond)
      tf.summary.scalar('D_loss_unregularized', 
                         (D_loss_real + 0.5 * (D_loss_wrong + D_loss_fake) \
                        + 0.5 * (D_loss_real_uncond + D_loss_wrong_uncond) + D_loss_fake_uncond) / 2)
    else:
      tf.summary.scalar('D_acc', 0.5 * (tf.reduce_mean(tf.sigmoid(D_x[0])) \
                                      + 0.5 * (tf.reduce_mean(1 - tf.sigmoid(D_w[0])) + tf.reduce_mean(1 - tf.sigmoid(D_G_z[0])))))
      tf.summary.scalar('D_loss_real', D_loss_real)
      tf.summary.scalar('D_loss_wrong', D_loss_wrong)
      tf.summary.scalar('D_loss_fake', D_loss_fake)
      tf.summary.scalar('D_loss_unregularized', D_loss_real + 0.5 * (D_loss_wrong + D_loss_fake))
    tf.summary.scalar('D_loss', D_loss)

  # Create (recommended) optimizer
  if args.wavegan_loss == 'dcgan':
    G_opt = tf.train.AdamOptimizer(
        learning_rate=0.0002,
        beta1=0.5)
    D_opt = tf.train.AdamOptimizer(
        learning_rate=0.000002,
        beta1=0.5)
  elif args.wavegan_loss == 'lsgan':
    G_opt = tf.train.RMSPropOptimizer(
        learning_rate=1e-4)
    D_opt = tf.train.RMSPropOptimizer(
        learning_rate=1e-4)
  elif args.wavegan_loss == 'wgan':
    G_opt = tf.train.RMSPropOptimizer(
        learning_rate=5e-5)
    D_opt = tf.train.RMSPropOptimizer(
        learning_rate=5e-5)
  elif args.wavegan_loss == 'wgan-gp':
    G_opt = tf.train.AdamOptimizer(
        learning_rate=2e-4,
        beta1=0.0,
        beta2=0.9)
    D_opt = tf.train.AdamOptimizer(
        learning_rate=2e-4,
        beta1=0.0,
        beta2=0.9)
  else:
    raise NotImplementedError()

  # Create training ops
  G_train_op = G_opt.minimize(G_loss, var_list=G_vars,
      global_step=tf.train.get_or_create_global_step())
  D_train_op = D_opt.minimize(D_loss, var_list=D_vars)

  # Variables for smoothly interpolating between LOD levels
  steps_at_cur_lod_var = tf.get_variable('steps_at_cur_lod', shape=[], dtype=tf.int32, trainable=False)
  steps_at_cur_lod_incr_op = steps_at_cur_lod_var.assign(steps_at_cur_lod_var + 1)

  def smoothstep(x, mi, mx):
    return mi + (mx-mi)*(lambda t: np.where(t < 0 , 0, np.where( t <= 1 , 3*t**2-2*t**3, 1 ) ) )( x )

  # Run training
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=args.train_dir,
      save_checkpoint_secs=args.train_save_secs,
      save_summaries_secs=args.train_summary_secs) as sess:
    # Get the summary writer for writing extra summary statistics
    summary_writer = SummaryWriterCache.get(args.train_dir)

    _lod = 0
    while True:
      # Calculate Maximum LOD to train
      step, steps_at_cur_lod = sess.run([tf.train.get_or_create_global_step(), steps_at_cur_lod_var], feed_dict={lod: _lod})
      cur_lod = np.piecewise(float(steps_at_cur_lod),
                            [         steps_at_cur_lod < 4000 , 4000  <= steps_at_cur_lod < 8000,
                             8000  <= steps_at_cur_lod < 12000, 12000 <= steps_at_cur_lod < 16000,
                             16000 <= steps_at_cur_lod < 20000, 20000 <= steps_at_cur_lod < 24000,
                             24000 <= steps_at_cur_lod < 28000, 28000 <= steps_at_cur_lod < 32000,
                             32000 <= steps_at_cur_lod < 36000, 36000 <= steps_at_cur_lod < 40000,
                             40000 <= steps_at_cur_lod < 44000, 44000 <= steps_at_cur_lod < 48000,
                             48000 <= steps_at_cur_lod < 52000, 52000 <= steps_at_cur_lod < 56000,
                             56000 <= steps_at_cur_lod < 60000, 60000 <= steps_at_cur_lod < 64000,
                             64000 <= steps_at_cur_lod < 68000, 68000 <= steps_at_cur_lod < 72000,
                             72000 <= steps_at_cur_lod < 76000, 76000 <= steps_at_cur_lod < 80000,
                             80000 <= steps_at_cur_lod < 84000, 84000 <= steps_at_cur_lod < 88000,
                             88000 <= steps_at_cur_lod < 92000, 92000 <= steps_at_cur_lod < 96000],
                            [0,  lambda x: smoothstep((x - 4000 ) / 4000, 0, 1),
                             1,  lambda x: smoothstep((x - 12000) / 4000, 1, 2),
                             2,  lambda x: smoothstep((x - 20000) / 4000, 2, 3),
                             3,  lambda x: smoothstep((x - 28000) / 4000, 3, 4),
                             4,  lambda x: smoothstep((x - 36000) / 4000, 4, 5),
                             5,  lambda x: smoothstep((x - 44000) / 4000, 5, 6),
                             6,  lambda x: smoothstep((x - 52000) / 4000, 6, 7),
                             7,  lambda x: smoothstep((x - 60000) / 4000, 7, 8),
                             8,  lambda x: smoothstep((x - 68000) / 4000, 8, 9),
                             9,  lambda x: smoothstep((x - 76000) / 4000, 9, 10),
                             10, lambda x: smoothstep((x - 84000) / 4000, 10, 11),
                             11, lambda x: smoothstep((x - 92000) / 4000, 11, 12),
                             12])
      
      if cur_lod > 0:
        # Randomly train on either, a previous LOD, or the current one.
        random_prev_lod = np.random.randint(math.ceil(cur_lod))
        _lod = np.random.choice([cur_lod, random_prev_lod]) # 50% probability of training on the current LOD
      else:
        _lod = cur_lod
      
      # Increment step counter when training on current LOD to smoothly interpolate between LOD levels.
      if _lod == cur_lod:
          sess.run(steps_at_cur_lod_incr_op, feed_dict={lod: _lod})


      # Output current LOD and 'steps at currrent LOD' to tensorboard
      step = float(sess.run(tf.train.get_or_create_global_step(), feed_dict={lod: _lod}))
      lod_summary = tf.Summary(value=[
        tf.Summary.Value(tag="current_lod", simple_value=float(cur_lod)),
      ])
      steps_at_cur_lod_summary = tf.Summary(value=[
        tf.Summary.Value(tag="steps_at_cur_lod", simple_value=float(steps_at_cur_lod))
      ])
      summary_writer.add_summary(lod_summary, step)
      summary_writer.add_summary(steps_at_cur_lod_summary, step)

      # Train discriminator
      for i in xrange(args.wavegan_disc_nupdates):
        sess.run(D_train_op, feed_dict={lod: _lod})

        # Enforce Lipschitz constraint for WGAN
        if D_clip_weights is not None:
          sess.run(D_clip_weights, feed_dict={lod: _lod})

      # Train generator
      sess.run(G_train_op, feed_dict={lod: _lod})


"""
  Creates and saves a MetaGraphDef for simple inference
  Tensors:
    'samp_z_n' int32 []: Sample this many latent vectors
    'samp_z' float32 [samp_z_n, 100]: Resultant latent vectors
    'z:0' float32 [None, 100]: Input latent vectors
    'c:0' float32 [None, 1024]: Input context embedding vector
    'flat_pad:0' int32 []: Number of padding samples to use when flattening batch to a single audio file
    'G_z:0' float32 [None, 16384, 1]: Generated outputs
    'G_z_int16:0' int16 [None, 16384, 1]: Same as above but quantizied to 16-bit PCM samples
    'G_z_flat:0' float32 [None, 1]: Outputs flattened into single audio file
    'G_z_flat_int16:0' int16 [None, 1]: Same as above but quantized to 16-bit PCM samples
  Example usage:
    import tensorflow as tf
    tf.reset_default_graph()

    saver = tf.train.import_meta_graph('infer.meta')
    graph = tf.get_default_graph()
    sess = tf.InteractiveSession()
    saver.restore(sess, 'model.ckpt-10000')

    z_n = graph.get_tensor_by_name('samp_z_n:0')
    _z = sess.run(graph.get_tensor_by_name('samp_z:0'), {z_n: 10})

    z = graph.get_tensor_by_name('G_z:0')
    _G_z = sess.run(graph.get_tensor_by_name('G_z:0'), {z: _z})
"""
def infer(args):
  infer_dir = os.path.join(args.train_dir, 'infer')
  if not os.path.isdir(infer_dir):
    os.makedirs(infer_dir)

  # Subgraph that generates latent vectors
  samp_z_n = tf.placeholder(tf.int32, [], name='samp_z_n')
  samp_z = tf.random_normal([samp_z_n, _D_Z], name='samp_z')

  # Input zo
  z = tf.placeholder(tf.float32, [None, _D_Z], name='z')
  flat_pad = tf.placeholder(tf.int32, [], name='flat_pad')

  # Conditioning input
  c = tf.placeholder(tf.float32, [None, 1024], name='c')

  # Execute generator
  with tf.variable_scope('G'):
    G_z, _ = WaveGANGenerator(z, tf.constant(5, dtype=tf.float32), train=False, context_embedding=c, **args.wavegan_g_kwargs)
    if args.wavegan_genr_pp:
      with tf.variable_scope('pp_filt'):
        G_z = tf.layers.conv1d(G_z, 1, args.wavegan_genr_pp_len, use_bias=False, padding='same')
  G_z = tf.identity(G_z, name='G_z')

  # Flatten batch
  nch = int(G_z.get_shape()[-1])
  G_z_padded = tf.pad(G_z, [[0, 0], [0, flat_pad], [0, 0]])
  G_z_flat = tf.reshape(G_z_padded, [-1, nch], name='G_z_flat')

  # Encode to int16
  def float_to_int16(x, name=None):
    x_int16 = x * 32767.
    x_int16 = tf.clip_by_value(x_int16, -32767., 32767.)
    x_int16 = tf.cast(x_int16, tf.int16, name=name)
    return x_int16
  G_z_int16 = float_to_int16(G_z, name='G_z_int16')
  G_z_flat_int16 = float_to_int16(G_z_flat, name='G_z_flat_int16')

  # Create saver
  G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G')
  global_step = tf.train.get_or_create_global_step()
  saver = tf.train.Saver(G_vars + [global_step])

  # Export graph
  tf.train.write_graph(tf.get_default_graph(), infer_dir, 'infer.pbtxt')

  # Export MetaGraph
  infer_metagraph_fp = os.path.join(infer_dir, 'infer.meta')
  tf.train.export_meta_graph(
      filename=infer_metagraph_fp,
      clear_devices=True,
      saver_def=saver.as_saver_def())

  # Reset graph (in case training afterwards)
  tf.reset_default_graph()


"""
  Generates a preview audio file every time a checkpoint is saved
"""
def preview(args):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from scipy.io.wavfile import write as wavwrite
  from scipy.signal import freqz

  preview_dir = os.path.join(args.train_dir, 'preview')
  if not os.path.isdir(preview_dir):
    os.makedirs(preview_dir)

  # Load graph
  infer_metagraph_fp = os.path.join(args.train_dir, 'infer', 'infer.meta')
  graph = tf.get_default_graph()
  saver = tf.train.import_meta_graph(infer_metagraph_fp)

  # Generate or restore z_i and z_o
  z_fp = os.path.join(preview_dir, 'z.pkl')
  if os.path.exists(z_fp):
    with open(z_fp, 'rb') as f:
      _zs = pickle.load(f)
  else:
    # Sample z
    samp_feeds = {}
    samp_feeds[graph.get_tensor_by_name('samp_z_n:0')] = args.preview_n
    samp_fetches = {}
    samp_fetches['zs'] = graph.get_tensor_by_name('samp_z:0')
    with tf.Session() as sess:
      _samp_fetches = sess.run(samp_fetches, samp_feeds)
    _zs = _samp_fetches['zs']

    # Save z
    with open(z_fp, 'wb') as f:
      pickle.dump(_zs, f)

  # Set up graph for generating preview images
  feeds = {}
  feeds[graph.get_tensor_by_name('z:0')] = _zs
  feeds[graph.get_tensor_by_name('flat_pad:0')] = _WINDOW_LEN // 2
  fetches = {}
  fetches['step'] = tf.train.get_or_create_global_step()
  fetches['G_z'] = graph.get_tensor_by_name('G_z:0')
  fetches['G_z_flat_int16'] = graph.get_tensor_by_name('G_z_flat_int16:0')
  if args.wavegan_genr_pp:
    fetches['pp_filter'] = graph.get_tensor_by_name('G/pp_filt/conv1d/kernel:0')[:, 0, 0]

  # Summarize
  G_z = graph.get_tensor_by_name('G_z_flat:0')
  summaries = [
      tf.summary.audio('preview', tf.expand_dims(G_z, axis=0), _FS, max_outputs=1)
  ]
  fetches['summaries'] = tf.summary.merge(summaries)
  summary_writer = tf.summary.FileWriter(preview_dir)

  # PP Summarize
  if args.wavegan_genr_pp:
    pp_fp = tf.placeholder(tf.string, [])
    pp_bin = tf.read_file(pp_fp)
    pp_png = tf.image.decode_png(pp_bin)
    pp_summary = tf.summary.image('pp_filt', tf.expand_dims(pp_png, axis=0))

  # Loop, waiting for checkpoints
  ckpt_fp = None
  while True:
    latest_ckpt_fp = tf.train.latest_checkpoint(args.train_dir)
    if latest_ckpt_fp != ckpt_fp:
      print('Preview: {}'.format(latest_ckpt_fp))

      with tf.Session() as sess:
        saver.restore(sess, latest_ckpt_fp)

        _fetches = sess.run(fetches, feeds)

        _step = _fetches['step']

      preview_fp = os.path.join(preview_dir, '{}.wav'.format(str(_step).zfill(8)))
      wavwrite(preview_fp, _FS, _fetches['G_z_flat_int16'])

      summary_writer.add_summary(_fetches['summaries'], _step)

      if args.wavegan_genr_pp:
        w, h = freqz(_fetches['pp_filter'])

        fig = plt.figure()
        plt.title('Digital filter frequncy response')
        ax1 = fig.add_subplot(111)

        plt.plot(w, 20 * np.log10(abs(h)), 'b')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [rad/sample]')

        ax2 = ax1.twinx()
        angles = np.unwrap(np.angle(h))
        plt.plot(w, angles, 'g')
        plt.ylabel('Angle (radians)', color='g')
        plt.grid()
        plt.axis('tight')

        _pp_fp = os.path.join(preview_dir, '{}_ppfilt.png'.format(str(_step).zfill(8)))
        plt.savefig(_pp_fp)

        with tf.Session() as sess:
          _summary = sess.run(pp_summary, {pp_fp: _pp_fp})
          summary_writer.add_summary(_summary, _step)

      print('Done')

      ckpt_fp = latest_ckpt_fp

    time.sleep(1)


"""
  Computes inception score every time a checkpoint is saved
"""
def incept(args):
  incept_dir = os.path.join(args.train_dir, 'incept')
  if not os.path.isdir(incept_dir):
    os.makedirs(incept_dir)

  # Load GAN graph
  gan_graph = tf.Graph()
  with gan_graph.as_default():
    infer_metagraph_fp = os.path.join(args.train_dir, 'infer', 'infer.meta')
    gan_saver = tf.train.import_meta_graph(infer_metagraph_fp)
    score_saver = tf.train.Saver(max_to_keep=1)
  gan_z = gan_graph.get_tensor_by_name('z:0')
  gan_G_z = gan_graph.get_tensor_by_name('G_z:0')[:, :, 0]
  gan_step = gan_graph.get_tensor_by_name('global_step:0')

  # Load or generate latents
  z_fp = os.path.join(incept_dir, 'z.pkl')
  if os.path.exists(z_fp):
    with open(z_fp, 'rb') as f:
      _zs = pickle.load(f)
  else:
    gan_samp_z_n = gan_graph.get_tensor_by_name('samp_z_n:0')
    gan_samp_z = gan_graph.get_tensor_by_name('samp_z:0')
    with tf.Session(graph=gan_graph) as sess:
      _zs = sess.run(gan_samp_z, {gan_samp_z_n: args.incept_n})
    with open(z_fp, 'wb') as f:
      pickle.dump(_zs, f)

  # Load classifier graph
  incept_graph = tf.Graph()
  with incept_graph.as_default():
    incept_saver = tf.train.import_meta_graph(args.incept_metagraph_fp)
  incept_x = incept_graph.get_tensor_by_name('x:0')
  incept_preds = incept_graph.get_tensor_by_name('scores:0')
  incept_sess = tf.Session(graph=incept_graph)
  incept_saver.restore(incept_sess, args.incept_ckpt_fp)

  # Create summaries
  summary_graph = tf.Graph()
  with summary_graph.as_default():
    incept_mean = tf.placeholder(tf.float32, [])
    incept_std = tf.placeholder(tf.float32, [])
    summaries = [
        tf.summary.scalar('incept_mean', incept_mean),
        tf.summary.scalar('incept_std', incept_std)
    ]
    summaries = tf.summary.merge(summaries)
  summary_writer = tf.summary.FileWriter(incept_dir)

  # Loop, waiting for checkpoints
  ckpt_fp = None
  _best_score = 0.
  while True:
    latest_ckpt_fp = tf.train.latest_checkpoint(args.train_dir)
    if latest_ckpt_fp != ckpt_fp:
      print('Incept: {}'.format(latest_ckpt_fp))

      sess = tf.Session(graph=gan_graph)

      gan_saver.restore(sess, latest_ckpt_fp)

      _step = sess.run(gan_step)

      _G_zs = []
      for i in xrange(0, args.incept_n, 100):
        _G_zs.append(sess.run(gan_G_z, {gan_z: _zs[i:i+100]}))
      _G_zs = np.concatenate(_G_zs, axis=0)

      _preds = []
      for i in xrange(0, args.incept_n, 100):
        _preds.append(incept_sess.run(incept_preds, {incept_x: _G_zs[i:i+100]}))
      _preds = np.concatenate(_preds, axis=0)

      # Split into k groups
      _incept_scores = []
      split_size = args.incept_n // args.incept_k
      for i in xrange(args.incept_k):
        _split = _preds[i * split_size:(i + 1) * split_size]
        _kl = _split * (np.log(_split) - np.log(np.expand_dims(np.mean(_split, 0), 0)))
        _kl = np.mean(np.sum(_kl, 1))
        _incept_scores.append(np.exp(_kl))

      _incept_mean, _incept_std = np.mean(_incept_scores), np.std(_incept_scores)

      # Summarize
      with tf.Session(graph=summary_graph) as summary_sess:
        _summaries = summary_sess.run(summaries, {incept_mean: _incept_mean, incept_std: _incept_std})
      summary_writer.add_summary(_summaries, _step)

      # Save
      if _incept_mean > _best_score:
        score_saver.save(sess, os.path.join(incept_dir, 'best_score'), _step)
        _best_score = _incept_mean

      sess.close()

      print('Done')

      ckpt_fp = latest_ckpt_fp

    time.sleep(1)

  incept_sess.close()


if __name__ == '__main__':
  import argparse
  import glob
  import sys

  parser = argparse.ArgumentParser()

  parser.add_argument('mode', type=str, choices=['train', 'preview', 'incept', 'infer'])
  parser.add_argument('train_dir', type=str,
      help='Training directory')

  data_args = parser.add_argument_group('Data')
  data_args.add_argument('--data_dir', type=str,
      help='Data directory')
  data_args.add_argument('--data_first_window', action='store_true', dest='data_first_window',
      help='If set, only use the first window from each audio example')

  wavegan_args = parser.add_argument_group('WaveGAN')
  wavegan_args.add_argument('--wavegan_kernel_len', type=int,
      help='Length of 1D filter kernels')
  wavegan_args.add_argument('--wavegan_dim', type=int,
      help='Dimensionality multiplier for model of G and D')
  wavegan_args.add_argument('--wavegan_batchnorm', action='store_true', dest='wavegan_batchnorm',
      help='Enable batchnorm')
  wavegan_args.add_argument('--wavegan_disc_nupdates', type=int,
      help='Number of discriminator updates per generator update')
  wavegan_args.add_argument('--wavegan_loss', type=str, choices=['dcgan', 'lsgan', 'wgan', 'wgan-gp'],
      help='Which GAN loss to use')
  wavegan_args.add_argument('--wavegan_genr_upsample', type=str, choices=['zeros', 'nn', 'lin', 'cub'],
      help='Generator upsample strategy')
  wavegan_args.add_argument('--wavegan_genr_pp', action='store_true', dest='wavegan_genr_pp',
      help='If set, use post-processing filter')
  wavegan_args.add_argument('--wavegan_genr_pp_len', type=int,
      help='Length of post-processing filter for DCGAN')
  wavegan_args.add_argument('--wavegan_disc_phaseshuffle', type=int,
      help='Radius of phase shuffle operation')
  wavegan_args.add_argument('--use_extra_uncond_loss', action='store_true', dest='use_extra_uncond_loss',
      help='If set, use post-processing filter')

  train_args = parser.add_argument_group('Train')
  train_args.add_argument('--train_batch_size', type=int,
      help='Batch size')
  train_args.add_argument('--train_save_secs', type=int,
      help='How often to save model')
  train_args.add_argument('--train_summary_secs', type=int,
      help='How often to report summaries')

  preview_args = parser.add_argument_group('Preview')
  preview_args.add_argument('--preview_n', type=int,
      help='Number of samples to preview')

  incept_args = parser.add_argument_group('Incept')
  incept_args.add_argument('--incept_metagraph_fp', type=str,
      help='Inference model for inception score')
  incept_args.add_argument('--incept_ckpt_fp', type=str,
      help='Checkpoint for inference model')
  incept_args.add_argument('--incept_n', type=int,
      help='Number of generated examples to test')
  incept_args.add_argument('--incept_k', type=int,
      help='Number of groups to test')

  parser.set_defaults(
    data_dir=None,
    data_first_window=False,
    wavegan_kernel_len=24,
    wavegan_dim=4,
    wavegan_batchnorm=False,
    wavegan_disc_nupdates=5,
    wavegan_loss='wgan-gp',
    wavegan_genr_upsample='zeros',
    wavegan_genr_pp=False,
    wavegan_genr_pp_len=512,
    wavegan_disc_phaseshuffle=2,
    use_extra_uncond_loss=False,
    train_batch_size=64,
    train_save_secs=300,
    train_summary_secs=120,
    preview_n=32,
    incept_metagraph_fp='./eval/inception/infer.meta',
    incept_ckpt_fp='./eval/inception/best_acc-103005',
    incept_n=5000,
    incept_k=10)

  args = parser.parse_args()

  # Make train dir
  if not os.path.isdir(args.train_dir):
    os.makedirs(args.train_dir)

  # Save args
  with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

  # Make model kwarg dicts
  setattr(args, 'wavegan_g_kwargs', {
      'kernel_len': args.wavegan_kernel_len,
      'dim': args.wavegan_dim,
      'use_batchnorm': args.wavegan_batchnorm,
      'upsample': args.wavegan_genr_upsample
  })
  setattr(args, 'wavegan_d_kwargs', {
      'kernel_len': args.wavegan_kernel_len,
      'dim': args.wavegan_dim,
      'use_batchnorm': args.wavegan_batchnorm,
      'phaseshuffle_rad': args.wavegan_disc_phaseshuffle,
      'use_extra_uncond_output': args.use_extra_uncond_loss
  })

  # Assign appropriate split for mode
  if args.mode == 'train':
    split = 'train'
  else:
    split = None

  # Find fps for split
  if split is not None:
    fps = glob.glob(os.path.join(args.data_dir, split) + '*.tfrecord')

  if args.mode == 'train':
    infer(args)
    train(fps, args)
  elif args.mode == 'preview':
    preview(args)
  elif args.mode == 'incept':
    incept(args)
  elif args.mode == 'infer':
    infer(args)
  else:
    raise NotImplementedError()
