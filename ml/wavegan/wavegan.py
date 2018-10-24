import tensorflow as tf


def lerp_clip(a, b, t): 
  return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)


def nn_upsample(inputs, stride=2):
  '''
  Upsamples an audio clip using nearest neighbor upsampling.
  Output is of size 'audio clip length' x 'stride'
  '''
  with tf.variable_scope('nn_upsample'):
    w = tf.shape(inputs)[1]
    output = tf.expand_dims(inputs, axis=1)
    output = tf.image.resize_nearest_neighbor(output, [1, w * stride])
    output = output[:, 0]
    return output


def tconv_upsample(inputs, filters, kernel_size=8, stride=2):
  '''
  Upsamples an audio clip using transpose convolution upsampling.
  Output is of size 'audio clip length' x 'stride'
  '''
  with tf.variable_scope('tconv_upsample'):
    return tf.layers.conv2d_transpose(
      tf.expand_dims(inputs, axis=1),
      filters,
      kernel_size=(1, kernel_size),
      strides=(1, stride),
      padding='same')[:, 0]


def avg_downsample(inputs, stride=2):
  with tf.variable_scope('downsample'):
    return tf.layers.average_pooling1d(inputs, pool_size=stride, strides=stride, padding='same')


def lrelu(inputs, alpha=0.2):
  with tf.variable_scope('lrelu'):
    return tf.maximum(alpha * inputs, inputs)
  

def to_audio(in_code, pre_activation=lrelu, post_activation=tf.tanh, normalization=lambda x: x):
  '''
  Converts 2d feature map into an audio clip.
  Usage Note: :param in_code: is expected to be non-activated (linear).
  :param pre_activation: Will be applied to in_code before downsampling feature dimensions.
  :param post_activation: Will be applied after downsampling to get final audio output.
  '''
  with tf.variable_scope('to_audio'):
    output = normalization(in_code)
    output = pre_activation(output)
    output = tf.layers.conv1d(output, filters=1, kernel_size=1, strides=1, padding='same')
    return post_activation(output)

def from_audio(inputs, out_feature_maps):
  '''
  Converts an input audio clips into a 2d feature maps.
  Usage Note: Output is linear transform (no non-linear activation function applied).
              Intended to be used before a residual block, that does pre-activation as its first step.
  :param out_feature_maps: The number of feature maps to output.
  '''
  with tf.variable_scope('from_audio'):
    return tf.layers.conv1d(inputs, filters=out_feature_maps, kernel_size=1, strides=1, padding='same')


def add_conditioning(in_code, cond_embed):
  '''
  Adds conditioning input to a hidden state by tiling and appending to feature maps
  '''
  with tf.variable_scope('add_conditioning'):
    state_size = in_code.get_shape().as_list()[1]
    c_code = tf.expand_dims(cond_embed, 1)
    c_code = tf.tile(c_code, [1, state_size, 1])
    h_c_code = tf.concat([in_code, c_code], 2)
    return h_c_code


def up_block(inputs, audio_lod, filters, on_amount, kernel_size=9, stride=2, activation=lrelu, normalization=lambda x: x, upsample_method='zeros'):
  '''
  Up Block
  '''
  with tf.variable_scope('up_block'):
    def skip():
      with tf.variable_scope('skip'):
        skip_connection_audio = nn_upsample(audio_lod, stride)
        skip_connection_code = tf.zeros([tf.shape(inputs)[0], tf.shape(inputs)[1] * stride, filters], dtype=tf.float32)
        return skip_connection_code, skip_connection_audio

    def transition():
      with tf.variable_scope('transition'):
        skip_connection_audio = nn_upsample(audio_lod, stride)

        # Shortcut
        with tf.variable_scope('shortcut'):
          shortcut = nn_upsample(inputs, stride)
          if shortcut.get_shape().as_list()[2] != filters:
            shortcut = tf.layers.conv1d(shortcut, filters, kernel_size=1, strides=1, padding='same')

        code = inputs

        # Convolution layers
        with tf.variable_scope('conv_0'):
          code = normalization(code)
          code = activation(code) # Pre-Activation
          if upsample_method == 'zeros':
            code = tconv_upsample(code, filters, kernel_size, stride=stride) # Upsample - Transposed Convolution
          elif upsample_method == 'nn':
            code = nn_upsample(code, stride) # Upsample - Nearest Neighbor
            code = tf.layers.conv1d(code, filters, kernel_size, strides=1, padding='same')
          else:
            raise NotImplementedError
        with tf.variable_scope('conv_1'):
          code = normalization(code)
          code = activation(code)  # Pre-Activation
          code = tf.layers.conv1d(code, filters, kernel_size, strides=1, padding='same')
        
        # Add shortcut connection
        code = shortcut + code
      
        # Blend this LOD block in over time
        audio_lod_ = to_audio(code, normalization=normalization)
        audio_lod_ = lerp_clip(skip_connection_audio, audio_lod_, on_amount)
        return code, audio_lod_

    code, audio_lod = tf.cond(on_amount <= 0.0, skip, transition)
    code.set_shape([inputs.shape[0], inputs.shape[1] * stride, filters])
    audio_lod.set_shape([inputs.shape[0], inputs.shape[1] * stride, 1])

    return code, audio_lod


def down_block(inputs, audio_lod, filters, on_amount, kernel_size=9, stride=2, activation=lrelu, normalization=lambda x: x):
  '''
  Down Block
  '''
  with tf.variable_scope('down_block'):
    audio_lod = avg_downsample(audio_lod, stride)
    def skip():
      with tf.variable_scope('skip'):
        skip_connection_code = from_audio(audio_lod, filters)
        return skip_connection_code

    def transition():
      with tf.variable_scope('transition'):
        skip_connection_code = from_audio(audio_lod, filters)

        # Shortcut
        with tf.variable_scope('shortcut'):
          shortcut = avg_downsample(inputs, stride)
          if shortcut.get_shape().as_list()[2] != filters:
            shortcut = tf.layers.conv1d(shortcut, filters, kernel_size=1, strides=1, padding='same')

        code = inputs

         # Convolution layers
        with tf.variable_scope('conv_0'):
          code = normalization(code)
          code = activation(code)  # Pre-Activation
          code = tf.layers.conv1d(code, inputs.get_shape().as_list()[2], kernel_size, strides=1, padding='same')
        with tf.variable_scope('conv_1'):
          code = normalization(code)
          code = activation(code)  # Pre-Activation
          code = tf.layers.conv1d(code, filters, kernel_size, strides=stride, padding='same')

        # Add shortcut connection
        code = shortcut + code

        # Blend this LOD block in over time
        return lerp_clip(skip_connection_code, code, on_amount)
      
    code = tf.cond(on_amount <= 0.0, skip, transition)
    code.set_shape([inputs.shape[0], inputs.shape[1] // stride, filters])

    return code, audio_lod


def residual_block(inputs, filters, kernel_size=9, stride=1, padding='same', activation=lrelu, normalization=lambda x: x):
  with tf.variable_scope('residual_block'):
    shortcut = inputs
    if shortcut.get_shape().as_list()[2] != filters:
      shortcut = tf.layers.conv1d(shortcut, filters, kernel_size=1, strides=1, padding='same')

    code = inputs

    # Convolution layers
    with tf.variable_scope('conv_0'):
      code = normalization(code)
      code = activation(code)  # Pre-Activation
      code = tf.layers.conv1d(code, inputs.get_shape().as_list()[2], kernel_size, strides=1, padding='same')
    with tf.variable_scope('conv_1'):
      code = normalization(code)
      code = activation(code)  # Pre-Activation
      code = tf.layers.conv1d(code, filters, kernel_size, strides=stride, padding='same')

    # Add shortcut connection
    code = shortcut + code

    return code


def compress_embedding(embedding, embed_size):
  """
  Return compressed embedding for discriminator.
  Note that this is a linear transform and no non-linear activation is applied here.
  c: Embedded context to reduce, this should be a [batch_size x N] tensor
  embed_size: The size of the new embedding
  returns: [batch_size x embed_size] tensor
  """
  with tf.variable_scope('reduce_embed'):
    embedding = tf.layers.dense(embedding, embed_size)
    return embedding


def generate_context_dist_params(embedding, embed_size, train=False):
  """
  Generates the parameters for a gaussian distribution derived from a
  supplied context embedding.
    embedding - The input context from which to derive the sampling distribution parameters
    embed_size - The size of the embedding vector we are using in this program
    train - Flag to tell the generator whether to use dropout or not
    Returns - [batch_size, 2 * embed_size] sized tensor containing distribution parameters
              (mean, log(sigma)) where sigma is the diagonal entries for the covariance matrix
  """
  with tf.variable_scope('gen_context_dist'):
      params = lrelu(tf.layers.dense(embedding, 2 * embed_size))
  mean = params[:, :embed_size]
  log_sigma = params[:, embed_size:]
  return mean, log_sigma


def KL_loss(mu, log_sigma):
  with tf.name_scope("KL_divergence"):
    loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
    loss = tf.reduce_mean(loss)
    return loss


def sample_context_embeddings(embedding, embed_size, train=False):
  """
  Resamples the context embedding from a normal distribution derived 
  from the supplied context embedding.
    embedding - The input context from which to derive the sampling distribution
    embed_size - The size of output embedding vector
    train - Whether to do resample or just reduce the embedding
  """
  mean, log_sigma = generate_context_dist_params(embedding, embed_size, train)
  if train:
    epsilon = tf.truncated_normal(tf.shape(mean))
    stddev = tf.exp(log_sigma)
    c = mean + stddev * epsilon

    kl_loss = KL_loss(mean, log_sigma)
  else:
    c = mean # This is just the unmodified compressed embedding.
    kl_loss = 0

  TRAIN_COEFF_KL = 2.0
  return c, TRAIN_COEFF_KL * kl_loss


def minibatch_stddev_layer(x, group_size=4):
  with tf.variable_scope('minibatch_stddev'):
    group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
    s = x.shape                                             # [NWC]  Input shape.
    y = tf.reshape(x, [group_size, -1, s[1], s[2]])         # [GMWC] Split minibatch into G groups of size M.
    y = tf.cast(y, tf.float32)                              # [GMWC] Cast to FP32.
    y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMWC] Subtract mean over group.
    y = tf.reduce_mean(tf.square(y), axis=0)                # [MWC]  Calc variance over group.
    y = tf.sqrt(y + 1e-8)                                   # [MWC]  Calc stddev over group.
    y = tf.reduce_mean(y, axis=[1,2], keepdims=True)        # [M11]  Take average over fmaps and samples.
    y = tf.cast(y, x.dtype)                                 # [M11]  Cast back to original data type.
    y = tf.tile(y, [group_size, s[1], 1])                   # [NW1]  Replicate over group and samples.
    return tf.concat([x, y], axis=2)                        # [NWC]  Append as new fmap.


"""
  Input: [None, 100]
  Output: [None, 16384, 1], kl_loss for regularizing context embedding sample distribution
"""
def WaveGANGenerator(
    z,
    lod,
    kernel_len=24,
    dim=4,
    use_batchnorm=False,
    upsample='zeros',
    train=False,
    context_embedding=None,
    embedding_dim=128):
  batch_size = tf.shape(z)[0]
  lod_levels = 13

  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
    _batchnorm = lambda x: tf.contrib.layers.batch_norm(x, is_training=train, updates_collections=None) # Hacky fix for weird tensorflow bug that only happens when using batchnorm in an up_block
  else:
    _batchnorm = batchnorm = lambda x: x

  if (context_embedding is not None):
    # Reduce or expand context embedding to be size [embedding_dim]
    c_code = compress_embedding(context_embedding, embedding_dim)
    kl_loss = 0
    h_c_code = lrelu(batchnorm(c_code)) # Apply normalization and activation to c_code before passing it to fully connected layer
    h_code = tf.concat([z, h_c_code], 1) 
  else:
    h_code = z
    kl_loss = 0

  # FC and reshape for convolution
  # [256] -> [4, 256]
  with tf.variable_scope('z_project'):
    h_code = tf.layers.dense(h_code, 4 * dim * 64)
    h_code = tf.reshape(h_code, [batch_size, 4, dim * 64])

  # [4, 256] -> [4, 256]
  with tf.variable_scope('layer_0'):
    h_code = residual_block(h_code, filters=dim * 64, kernel_size=kernel_len, normalization=batchnorm)
    if (context_embedding is not None):
      h_code = add_conditioning(h_code, c_code)
    audio_lod = to_audio(h_code, normalization=batchnorm)
    # Slightly hacky fix for tf.summary.audio not working with low sample rates
    summary_audio = audio_lod
    for _ in range(lod_levels - 1): # 12 upsamples 13 lod levels
      summary_audio = nn_upsample(summary_audio)
    tf.summary.audio('G_audio', summary_audio, 16000, max_outputs=10, family='G_audio_lod_0')

  # [4, 256] -> [16384, 4] (h_code)
  # [16384, 4] -> [16384, 1] (audio_lod)
  for i in range(1, lod_levels):
    with tf.variable_scope('upconv_{}'.format(i - 1)):
      on_amount = lod - i + 1
      filters = min(dim * 64, dim * (2 ** (lod_levels - 1)) // (2 ** i))
      if (i != 0 and context_embedding is not None): # Re-apply conditioning on all later lods
        h_code = add_conditioning(h_code, c_code)
      h_code, audio_lod = up_block(h_code, audio_lod=audio_lod, filters=filters, kernel_size=kernel_len, normalization=_batchnorm, on_amount=on_amount, upsample_method=upsample)
      
      # Summary info
      tf.summary.scalar('on_amount', on_amount)
      # Slightly hacky fix for tf.summary.audio not working with low sample rates
      summary_audio = audio_lod
      for _ in range(lod_levels - 1 - i):
        summary_audio = nn_upsample(summary_audio)
      tf.summary.audio('G_audio', summary_audio, 16000, max_outputs=10, family='G_audio_lod_{}'.format(i))

  # Automatically update batchnorm moving averages every time G is used during training
  if train and use_batchnorm:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # if len(update_ops) != 10:
    #   raise Exception('Other update ops found in graph')
    with tf.control_dependencies(update_ops):
      audio_lod = tf.identity(audio_lod)

  return audio_lod, kl_loss


def apply_phaseshuffle(x, rad, pad_type='reflect'):
  b, x_len, nch = x.get_shape().as_list()

  phase = tf.random_uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
  pad_l = tf.maximum(phase, 0)
  pad_r = tf.maximum(-phase, 0)
  phase_start = pad_r
  x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

  x = x[:, phase_start:phase_start+x_len]
  x.set_shape([b, x_len, nch])

  return x


def encode_audio(x,
    lod,
    kernel_len=24,
    dim=4,
    use_batchnorm=False,
    phaseshuffle_rad=0,
    embedding_dim=128):
  lod_levels = 13
  
  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
  else:
    batchnorm = lambda x: x

  if phaseshuffle_rad > 0:
    phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
  else:
    phaseshuffle = lambda x: x

  with tf.variable_scope('audio_encode'):
    if 'D_x/' in tf.get_default_graph().get_name_scope():
      tf.summary.audio('input_audio', x, 16000, max_outputs=10, family='D_audio_lod_0')

    # [16384, 1] -> [16384, 4] (audio_lod)
    # [16384, 4] -> [4, 256] (h_code)
    audio_lod = x
    h_code = from_audio(x, dim)
    for i in range(lod_levels - 1, 0, -1):
      with tf.variable_scope('downconv_{}'.format(lod_levels - 1 - i)):
        on_amount = lod - i + 1
        filters = min(dim * 64, dim * (2 ** (lod_levels - 1)) // (2 ** i))
        h_code, audio_lod = down_block(h_code, audio_lod=audio_lod, filters=filters, kernel_size=kernel_len, normalization=batchnorm, on_amount=on_amount)

        # Summary info
        if 'D_x/' in tf.get_default_graph().get_name_scope(): # Only output for a single discriminator (discriminator on real audio clips)
          tf.summary.scalar('on_amount', on_amount)
          # Slightly hacky fix for tf.summary.audio not working with low sample rates
          summary_audio = audio_lod
          for _ in range(lod_levels - 1 - i):
            summary_audio = nn_upsample(summary_audio)
          tf.summary.audio('D_audio', summary_audio, 16000, max_outputs=10, family='D_audio_lod_{}'.format(i))

    return h_code, audio_lod


"""
  Input: [None, 16384, 1]
  Output: [None] (linear output)
"""
def WaveGANDiscriminator(
    x,
    lod,
    kernel_len=24,
    dim=4,
    use_batchnorm=False,
    phaseshuffle_rad=0,
    context_embedding=None,
    embedding_dim=128,
    use_extra_uncond_output=False):
  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
  else:
    batchnorm = lambda x: x

  x_code, _ = encode_audio(x, lod, kernel_len, dim, use_batchnorm, phaseshuffle_rad, embedding_dim)
  
  if (context_embedding is not None):
    with tf.variable_scope('conditional'):
      # Add conditioning to audio encoding
      c_code = compress_embedding(context_embedding, embedding_dim)
      cond_out = add_conditioning(x_code, c_code)
      output = cond_out
  else:
    output = x_code

  # <-- TODO: Minibatch std deviation layer goes here

  # Final residual block
  # [4, 256] -> [4, 256]
  with tf.variable_scope('final_convs'):
    output = residual_block(output, filters=cond_out.get_shape().as_list()[2], kernel_size=kernel_len, normalization=batchnorm, stride=1, padding='same')
  if (use_extra_uncond_output) and (context_embedding is not None):
    with tf.variable_scope('final_convs_uncond'):
      uncond_out = residual_block(x_code, filters=x_code.get_shape().as_list()[2], kernel_size=kernel_len, normalization=batchnorm, stride=1, padding='same')

  # FC 1
  # [4, 256] -> [256]
  batch_size = tf.shape(x)[0]
  with tf.variable_scope('fully_connected_1'):
    output = tf.reshape(output, [batch_size, -1]) # Flatten
    output = batchnorm(output)
    output = lrelu(output)
    output = tf.layers.dense(output, dim * 64)

    if (use_extra_uncond_output) and (context_embedding is not None):
      uncond_out = tf.reshape(uncond_out, [batch_size, -1]) # Flatten
      uncond_out = batchnorm(uncond_out)
      uncond_out = lrelu(uncond_out)
      uncond_out = tf.layers.dense(uncond_out, dim * 64)

  # FC 2
  # [256] -> [1]
  with tf.variable_scope('fully_connected_2'):
    output = batchnorm(output)
    output = lrelu(output)
    output = tf.layers.dense(output, 1)

    if (use_extra_uncond_output) and (context_embedding is not None):
      uncond_out = batchnorm(uncond_out)
      uncond_out = lrelu(uncond_out)
      uncond_out = tf.layers.dense(uncond_out, 1)
      return [output, uncond_out]
    else:
      return [output]
  
  # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training
