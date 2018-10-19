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
_D_Z = 256


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
  args.wavegan_d_dcgan_kwargs['context_embedding'] = cond_text_embed
  args.wavegan_d_wgan_gp_kwargs['context_embedding'] = cond_text_embed

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

  # Make real discriminators
  with tf.name_scope('D_x'), tf.variable_scope('D_dcgan'):
    D_x_dcgan = WaveGANDiscriminator(x, lod, **args.wavegan_d_dcgan_kwargs)
  with tf.name_scope('D_x'), tf.variable_scope('D_wgan_gp'):
    D_x_wgan_gp = WaveGANDiscriminator(x, lod, **args.wavegan_d_wgan_gp_kwargs)
  D_vars_dcgan   = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_dcgan')
  D_vars_wgan_gp = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_wgan_gp')

  # Print D summary
  print('-' * 80)
  print('Discriminator vars')
  nparams = 0
  for v in D_vars_dcgan + D_vars_wgan_gp:
    v_shape = v.get_shape().as_list()
    v_n = reduce(lambda x, y: x * y, v_shape)
    nparams += v_n
    print('{} ({}): {}'.format(v.get_shape().as_list(), v_n, v.name))
  print('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))
  print('-' * 80)

  # Make fake / wrong discriminator
  with tf.name_scope('D_G_z'), tf.variable_scope('D_dcgan', reuse=True):
    D_G_z_dcgan = WaveGANDiscriminator(G_z, lod, **args.wavegan_d_dcgan_kwargs)
  with tf.name_scope('D_w'), tf.variable_scope('D_dcgan', reuse=True):
    D_w_dcgan = WaveGANDiscriminator(wrong_audio, lod, **args.wavegan_d_dcgan_kwargs)
  with tf.name_scope('D_G_z'), tf.variable_scope('D_wgan_gp', reuse=True):
    D_G_z_wgan_gp = WaveGANDiscriminator(G_z, lod, **args.wavegan_d_wgan_gp_kwargs)
  with tf.name_scope('D_w'), tf.variable_scope('D_wgan_gp', reuse=True):
    D_w_wgan_gp = WaveGANDiscriminator(wrong_audio, lod, **args.wavegan_d_wgan_gp_kwargs)

  #####################################################################
  # Create loss (DCGAN)                                               #
  #####################################################################

  fake = tf.zeros([args.train_batch_size, 1], dtype=tf.float32)
  real = tf.ones([args.train_batch_size, 1], dtype=tf.float32)

  # Conditional G Loss
  G_loss_dcgan = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_G_z_dcgan[0],
    labels=real
  ))
  G_loss_dcgan += c_kl_loss

  # Unconditional G Loss
  if args.use_extra_uncond_loss:
    G_loss_dcgan += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=D_G_z_dcgan[1],
      labels=real
    ))
    G_loss_dcgan /= 2

  # Conditional D Losses
  D_loss_dcgan_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_G_z_dcgan[0],
    labels=fake
  ))
  D_loss_dcgan_wrong = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_w_dcgan[0],
    labels=fake
  ))
  D_loss_dcgan_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_x_dcgan[0],
    labels=real
  ))

  # Unconditional D Losses
  if args.use_extra_uncond_loss:
    D_loss_dcgan_fake_uncond = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=D_G_z_dcgan[1],
      labels=fake
    ))
    D_loss_dcgan_wrong_uncond = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=D_w_dcgan[1],
      labels=real
    ))
    D_loss_dcgan_real_uncond = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=D_x_dcgan[1],
      labels=real
    ))

    D_loss_dcgan = D_loss_dcgan_real + 0.5 * (D_loss_dcgan_wrong + D_loss_dcgan_fake) \
          + 0.5 * (D_loss_dcgan_real_uncond + D_loss_dcgan_wrong_uncond) + D_loss_dcgan_fake_uncond
    D_loss_dcgan /= 2
  else:
    D_loss_dcgan = D_loss_dcgan_real + 0.5 * (D_loss_dcgan_wrong + D_loss_dcgan_fake)

  ####################################################################################

  ####################################################################################
  # Create loss (WGAN-GP)                                                            #
  ####################################################################################

  # Conditional G Loss
  G_loss_wgan_gp = -tf.reduce_mean(D_G_z_wgan_gp[0])
  G_loss_wgan_gp += c_kl_loss

  # Unconditional G Loss
  if args.use_extra_uncond_loss:
    G_loss_wgan_gp += -tf.reduce_mean(D_G_z_wgan_gp[1])
    G_loss_wgan_gp /= 2

  # Conditional D Loss
  D_loss_wgan_gp_real = -tf.reduce_mean(D_x_wgan_gp[0])
  D_loss_wgan_gp_wrong = tf.reduce_mean(D_w_wgan_gp[0])
  D_loss_wgan_gp_fake = tf.reduce_mean(D_G_z_wgan_gp[0]) 

  # Unconditional D Loss
  if args.use_extra_uncond_loss:
    D_loss_wgan_gp_real_uncond = -tf.reduce_mean(D_x_wgan_gp[1])
    D_loss_wgan_gp_wrong_uncond = -tf.reduce_mean(D_w_wgan_gp[1])
    D_loss_wgan_gp_fake_uncond = tf.reduce_mean(D_G_z_wgan_gp[1])

    D_loss_wgan_gp = D_loss_wgan_gp_real + 0.5 * (D_loss_wgan_gp_wrong + D_loss_wgan_gp_fake) \
            + 0.5 * (D_loss_wgan_gp_real_uncond + D_loss_wgan_gp_wrong_uncond) + D_loss_wgan_gp_fake_uncond
    D_loss_wgan_gp /= 2
  else:
    D_loss_wgan_gp = D_loss_wgan_gp_real + 0.5 * (D_loss_wgan_gp_wrong + D_loss_wgan_gp_fake)


  # Stack duplicate context embeddings for extra interps on wrong audio
  interp_args = args.wavegan_d_wgan_gp_kwargs.copy()
  interp_args['context_embedding'] = tf.concat([interp_args['context_embedding'], interp_args['context_embedding']], 0)

  # Conditional Gradient Penalty
  alpha = tf.random_uniform(shape=[args.train_batch_size * 2, 1, 1], minval=0., maxval=1.)
  real = tf.concat([x, x], 0)
  fake = tf.concat([G_z, wrong_audio], 0)
  differences = fake - real
  interpolates = real + (alpha * differences)
  with tf.name_scope('D_interp'), tf.variable_scope('D_wgan_gp', reuse=True):
    D_interp = WaveGANDiscriminator(interpolates, lod, **interp_args)[0] # Only want conditional output
  gradients = tf.gradients(D_interp, [interpolates])[0]
  slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
  cond_gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)

  # Unconditional Gradient Penalty
  alpha = tf.random_uniform(shape=[args.train_batch_size * 2, 1, 1], minval=0., maxval=1.)
  real = tf.concat([x, wrong_audio], 0)
  fake = tf.concat([G_z, G_z], 0)
  differences = fake - real
  interpolates = real + (alpha * differences)
  with tf.name_scope('D_interp'), tf.variable_scope('D_wgan_gp', reuse=True):
    D_interp = WaveGANDiscriminator(interpolates, lod, **interp_args)[1] # Only want unconditional output
  gradients = tf.gradients(D_interp, [interpolates])[0]
  slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
  uncond_gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)

  gradient_penalty = (cond_gradient_penalty + uncond_gradient_penalty) / 2

  LAMBDA = 10
  D_loss_wgan_gp += LAMBDA * gradient_penalty

  #################################################################################

  tf.summary.scalar('G_loss_dcgan', G_loss_dcgan)
  tf.summary.scalar('G_loss_wgan_gp', G_loss_wgan_gp)

  # WGAN-GP
  tf.summary.scalar('Gradient Penalty', LAMBDA * gradient_penalty)
  if args.use_extra_uncond_loss:
    tf.summary.scalar('Critic Score - Real Data - Condition Match', -D_loss_wgan_gp_real)
    tf.summary.scalar('Critic Score - Fake Data - Condition Match', D_loss_wgan_gp_fake)
    tf.summary.scalar('Critic Score - Wrong Data - Condition Match', D_loss_wgan_gp_wrong)
    tf.summary.scalar('Critic Score - Real Data', -D_loss_wgan_gp_real_uncond)
    tf.summary.scalar('Critic Score - Wrong Data', -D_loss_wgan_gp_wrong_uncond)
    tf.summary.scalar('Critic Score - Fake Data', D_loss_wgan_gp_fake_uncond)
    tf.summary.scalar('Wasserstein Distance - No Regularization Term',
                      -((D_loss_wgan_gp_real + 0.5 * (D_loss_wgan_gp_wrong + D_loss_wgan_gp_fake) \
                        + 0.5 * (D_loss_wgan_gp_real_uncond + D_loss_wgan_gp_wrong_uncond) + D_loss_wgan_gp_fake_uncond) / 2))
    tf.summary.scalar('Wasserstein Distance - Real-Wrong Only', -(D_loss_wgan_gp_real + D_loss_wgan_gp_wrong))
    tf.summary.scalar('Wasserstein Distance - Real-Fake Only',
                      -((D_loss_wgan_gp_real + D_loss_wgan_gp_fake \
                        + D_loss_wgan_gp_real_uncond + D_loss_wgan_gp_fake_uncond) / 2))
  else:
    tf.summary.scalar('Critic Score - Real Data', -D_loss_wgan_gp_real)
    tf.summary.scalar('Critic Score - Wrong Data', D_loss_wgan_gp_wrong)
    tf.summary.scalar('Critic Score - Fake Data', D_loss_wgan_gp_fake)
    tf.summary.scalar('Wasserstein Distance - No Regularization Term', -(D_loss_wgan_gp_real + 0.5 * (D_loss_wgan_gp_wrong + D_loss_wgan_gp_fake)))
  tf.summary.scalar('Wasserstein Distance - With Regularization Term', -D_loss_wgan_gp)

  # DCGAN
  if args.use_extra_uncond_loss:
    tf.summary.scalar('D_acc_uncond', 0.5 * ((0.5 * (tf.reduce_mean(tf.sigmoid(D_x_dcgan[1])) + tf.reduce_mean(tf.sigmoid(D_w_dcgan[1])))) \
                                            + tf.reduce_mean(1 - tf.sigmoid(D_G_z_dcgan[1]))))
    tf.summary.scalar('D_acc', 0.5 * (tf.reduce_mean(tf.sigmoid(D_x_dcgan[0])) \
                                    + 0.5 * (tf.reduce_mean(1 - tf.sigmoid(D_w_dcgan[0])) + tf.reduce_mean(1 - tf.sigmoid(D_G_z_dcgan[0])))))
    tf.summary.scalar('D_acc_real_wrong_only', 0.5 * (tf.reduce_mean(tf.sigmoid(D_x_dcgan[0])) \
                                                    + tf.reduce_mean(1 - tf.sigmoid(D_w_dcgan[0]))))
    tf.summary.scalar('D_loss_cond_real', D_loss_dcgan_real)
    tf.summary.scalar('D_loss_dcgan_uncond_real', D_loss_dcgan_real_uncond)
    tf.summary.scalar('D_loss_dcgan_cond_wrong', D_loss_dcgan_wrong)
    tf.summary.scalar('D_loss_dcgan_uncond_wrong', D_loss_dcgan_wrong_uncond)
    tf.summary.scalar('D_loss_dcgan_cond_fake', D_loss_dcgan_fake)
    tf.summary.scalar('D_loss_dcgan_uncond_fake', D_loss_dcgan_fake_uncond)
    tf.summary.scalar('D_loss_dcgan_unregularized', 
                        (D_loss_dcgan_real + 0.5 * (D_loss_dcgan_wrong + D_loss_dcgan_fake) \
                      + 0.5 * (D_loss_dcgan_real_uncond + D_loss_dcgan_wrong_uncond) + D_loss_dcgan_fake_uncond) / 2)
  else:
    tf.summary.scalar('D_acc', 0.5 * (tf.reduce_mean(tf.sigmoid(D_x_dcgan[0])) \
                                    + 0.5 * (tf.reduce_mean(1 - tf.sigmoid(D_w_dcgan[0])) + tf.reduce_mean(1 - tf.sigmoid(D_G_z_dcgan[0])))))
    tf.summary.scalar('D_loss_dcgan_real', D_loss_dcgan_real)
    tf.summary.scalar('D_loss_dcgan_wrong', D_loss_dcgan_wrong)
    tf.summary.scalar('D_loss_dcgan_fake', D_loss_dcgan_fake)
    tf.summary.scalar('D_loss_dcgan_unregularized', D_loss_dcgan_real + 0.5 * (D_loss_dcgan_wrong + D_loss_dcgan_fake))
    tf.summary.scalar('D_loss_dcgan', D_loss_dcgan)

  # DCGAN
  G_dcgan_opt = tf.train.AdamOptimizer(
      learning_rate=2e-4,
      beta1=0.5)
  D_dcgan_opt = tf.train.AdamOptimizer(
      learning_rate=2e-4,
      beta1=0.5)
  
  # WGAN-GP
  G_wgan_gp_opt = tf.train.AdamOptimizer(
      learning_rate=1e-4,
      beta1=0.0,
      beta2=0.9)
  D_wgan_gp_opt = tf.train.AdamOptimizer(
      learning_rate=1e-4,
      beta1=0.0,
      beta2=0.9)

  # Create training ops
  # DCGAN
  G_train_op_dcgan = G_dcgan_opt.minimize(G_loss_dcgan, var_list=G_vars,
      global_step=tf.train.get_or_create_global_step())
  D_train_op_dcgan = D_dcgan_opt.minimize(D_loss_dcgan, var_list=D_vars_dcgan)
  # WGAN-GP
  G_train_op_wgan_gp = G_wgan_gp_opt.minimize(G_loss_wgan_gp, var_list=G_vars,
      global_step=tf.train.get_or_create_global_step())
  D_train_op_wgan_gp = D_wgan_gp_opt.minimize(D_loss_wgan_gp, var_list=D_vars_wgan_gp)

  # Variables for smoothly interpolating between LOD levels
  # steps_at_cur_lod_var = tf.get_variable('steps_at_cur_lod', shape=[], dtype=tf.int32, trainable=False)
  # steps_at_cur_lod_incr_op = steps_at_cur_lod_var.assign(steps_at_cur_lod_var + 1)

  # def smoothstep(x, mi, mx):
  #   return mi + (mx-mi)*(lambda t: np.where(t < 0 , 0, np.where( t <= 1 , 3*t**2-2*t**3, 1 ) ) )( x )

  # Run training
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=args.train_dir,
      save_checkpoint_secs=args.train_save_secs,
      save_summaries_secs=args.train_summary_secs) as sess:

    # Always use unconditional loss for switching as we are trying to improve quality, not conditioning
    if args.use_extra_uncond_loss:
      D_dcgan_acc = 0.5 * ((0.5 * (tf.reduce_mean(tf.sigmoid(D_x_dcgan[1])) + tf.reduce_mean(tf.sigmoid(D_w_dcgan[1])))) \
                         + tf.reduce_mean(1 - tf.sigmoid(D_G_z_dcgan[1])))
    else:
      D_dcgan_acc = 0.5 * (tf.reduce_mean(tf.sigmoid(D_x_dcgan[0])) + tf.reduce_mean(1 - tf.sigmoid(D_G_z_dcgan[0])))
    
    _lod = 0
    while True:

      if D_dcgan_acc > 0.60:
        # Use WGAN-GP if DCGAN discriminator is performing too well (generated samples are too distant from real)
        D_train_op = D_train_op_wgan_gp
        G_train_op = D_train_op_wgan_gp
        min_train_iterations = 100
      else:
        # Otherwise use DCGAN (generated samples are converging to real)
        D_train_op = D_train_op_dcgan
        G_train_op = D_train_op_dcgan
        min_train_iterations = 100

      for _ in range(min_train_iterations):
        # Select a random LOD to train
        _lod = np.random.randint(6)

        # Train discriminator
        for _ in range(args.wavegan_disc_nupdates):
          sess.run(D_train_op, feed_dict={lod: _lod})

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
    wavegan_dim=16,
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
  setattr(args, 'wavegan_d_dcgan_kwargs', {
      'kernel_len': args.wavegan_kernel_len,
      'dim': args.wavegan_dim,
      'use_batchnorm': args.wavegan_batchnorm,
      'phaseshuffle_rad': args.wavegan_disc_phaseshuffle,
      'use_extra_uncond_output': args.use_extra_uncond_loss
  })
  setattr(args, 'wavegan_d_wgan_gp_kwargs', {
      'kernel_len': args.wavegan_kernel_len,
      'dim': args.wavegan_dim,
      'use_batchnorm': False,
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
