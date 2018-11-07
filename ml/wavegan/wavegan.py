import tensorflow as tf


def lerp_clip(a, b, t): 
  return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)


def pixel_norm(x, epsilon=1e-8, axis=2):
  with tf.variable_scope('PN'):
    return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=axis, keepdims=True) + epsilon)


def nn_upsample(inputs, stride=4):
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


def tconv_upsample(inputs, filters, kernel_size=8, stride=4):
  '''
  Upsamples an audio clip using transpose convolution upsampling.
  Output is of size 'audio clip length' x 'stride'
  '''
  with tf.variable_scope('tcup'):
    return tf.layers.conv2d_transpose(
      tf.expand_dims(inputs, axis=1),
      filters,
      kernel_size=(1, kernel_size),
      strides=(1, stride),
      padding='same')[:, 0]


def avg_downsample(inputs, stride=4):
  with tf.variable_scope('downsample'):
    return tf.layers.average_pooling1d(inputs, pool_size=stride, strides=stride, padding='same')


def lrelu(inputs, alpha=0.2):
  with tf.variable_scope('lrelu'):
    return tf.maximum(alpha * inputs, inputs)
  

def to_audio(in_code, pre_activation=lrelu, post_activation=tf.tanh, normalization=lambda x: x, use_pixel_norm=True):
  '''
  Converts 2d feature map into an audio clip.
  Usage Note: :param in_code: is expected to be non-activated (linear).
  :param pre_activation: Will be applied to in_code before downsampling feature dimensions.
  :param post_activation: Will be applied after downsampling to get final audio output.
  '''
  with tf.variable_scope('to_audio'):
    output = in_code
    output = normalization(output)
    output = pre_activation(output)
    if use_pixel_norm:
      output = pixel_norm(output)
    output = tf.layers.conv1d(output, filters=1, kernel_size=1, strides=1, padding='same')
    # output = post_activation(output)
    return output

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


def up_block(inputs, audio_lod, filters, on_amount, kernel_size=9, stride=4, activation=lrelu, normalization=lambda x: x, use_pixel_norm=True, upsample_method='zeros'):
  '''
  Up Block
  '''
  with tf.variable_scope('up'):
    skip_connection_audio = nn_upsample(audio_lod, stride)

    def skip():
      with tf.variable_scope('sk'):
        skip_connection_code = tf.zeros([tf.shape(inputs)[0], tf.shape(inputs)[1] * stride, filters], dtype=tf.float32)
        return skip_connection_code, skip_connection_audio

    def transition():
      with tf.variable_scope('tr'):
        # Shortcut
        with tf.variable_scope('sh'):
          shortcut = nn_upsample(inputs, stride)
          if shortcut.get_shape().as_list()[2] != filters:
            shortcut = tf.layers.conv1d(shortcut, filters, kernel_size=1, strides=1, padding='same')

        # TODO: Only compute code when fully on (this is when the code will actually be used by the next layer)
        code = inputs

        # Convolution layers
        with tf.variable_scope('co0'):
          code = normalization(code)
          code = activation(code) # Pre-Activation
          if use_pixel_norm:
            code = pixel_norm(code)
          if upsample_method == 'zeros':
            code = tconv_upsample(code, filters, kernel_size, stride=stride) # Upsample - Transposed Convolution
          elif upsample_method == 'nn':
            code = nn_upsample(code, stride) # Upsample - Nearest Neighbor
            code = tf.layers.conv1d(code, filters, kernel_size, strides=1, padding='same')
          else:
            raise NotImplementedError
        with tf.variable_scope('co1'):
          code = normalization(code)
          code = activation(code)  # Pre-Activation
          if use_pixel_norm:
            code = pixel_norm(code)
          code = tf.layers.conv1d(code, filters, kernel_size, strides=1, padding='same')
        
        # Add shortcut connection
        code = shortcut + code
      
        # Blend this LOD block in over time
        audio_lod_ = to_audio(code, normalization=normalization, use_pixel_norm=use_pixel_norm)
        audio_lod_ = lerp_clip(skip_connection_audio, audio_lod_, on_amount)
        return code, audio_lod_

    code, audio_lod = tf.cond(on_amount <= 0.0, skip, transition)
    code.set_shape([inputs.shape[0], inputs.shape[1] * stride, filters])
    audio_lod.set_shape([inputs.shape[0], inputs.shape[1] * stride, 1])

    return code, audio_lod


def down_block(inputs, audio_lod, filters, on_amount, kernel_size=9, stride=4, activation=lrelu, normalization=lambda x: x, use_pixel_norm=True):
  '''
  Down Block
  '''
  with tf.variable_scope('db'):
    audio_lod = avg_downsample(audio_lod, stride)
    skip_connection_code = from_audio(audio_lod, filters)

    def skip():
      with tf.variable_scope('dk'):
        return skip_connection_code

    def transition():
      with tf.variable_scope('tr'):
        # Shortcut
        with tf.variable_scope('sh'):
          shortcut = avg_downsample(inputs, stride)
          if shortcut.get_shape().as_list()[2] != filters:
            shortcut = tf.layers.conv1d(shortcut, filters, kernel_size=1, strides=1, padding='same')

        code = inputs

         # Convolution layers
        with tf.variable_scope('co0'):
          code = normalization(code)
          code = activation(code)  # Pre-Activation
          if use_pixel_norm:
            code = pixel_norm(code)
          code = tf.layers.conv1d(code, inputs.get_shape().as_list()[2], kernel_size, strides=1, padding='same')
        with tf.variable_scope('co1'):
          code = normalization(code)
          code = activation(code)  # Pre-Activation
          if use_pixel_norm:
            code = pixel_norm(code)
          code = tf.layers.conv1d(code, filters, kernel_size, strides=stride, padding='same')

        # Add shortcut connection
        code = shortcut + code

        # Blend this LOD block in over time
        return lerp_clip(skip_connection_code, code, on_amount)
      
    code = tf.cond(on_amount <= 0.0, skip, transition)
    code.set_shape([inputs.shape[0], inputs.shape[1] // stride, filters])

    return code, audio_lod


def residual_block(inputs, filters, kernel_size=9, stride=1, padding='same', activation=lrelu, normalization=lambda x: x, use_pixel_norm=True):
  with tf.variable_scope('rb'):
    shortcut = inputs
    if shortcut.get_shape().as_list()[2] != filters:
      shortcut = tf.layers.conv1d(shortcut, filters, kernel_size=1, strides=1, padding='same')

    code = inputs

    # Convolution layers
    with tf.variable_scope('co0'):
      code = normalization(code)
      code = activation(code)  # Pre-Activation
      if use_pixel_norm:
        code = pixel_norm(code)
      code = tf.layers.conv1d(code, inputs.get_shape().as_list()[2], kernel_size, strides=1, padding='same')
    with tf.variable_scope('co1'):
      code = normalization(code)
      code = activation(code)  # Pre-Activation
      if use_pixel_norm:
        code = pixel_norm(code)
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
    dim=16,
    use_batchnorm=False,
    upsample='zeros',
    train=False,
    context_embedding=None,
    embedding_dim=128,
    use_pixel_norm=True):
  batch_size = tf.shape(z)[0]

  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
    _batchnorm = lambda x: tf.contrib.layers.batch_norm(x, is_training=train, updates_collections=None) # Hacky fix for weird tensorflow bug that only happens when using batchnorm in an up_block
  else:
    _batchnorm = batchnorm = lambda x: x

  h_code = z
  if use_pixel_norm:
    h_code = pixel_norm(h_code, axis=1)

  if (context_embedding is not None):
    # Reduce or expand context embedding to be size [embedding_dim]
    c_code = compress_embedding(context_embedding, embedding_dim)
    kl_loss = 0
    h_c_code = lrelu(batchnorm(c_code)) # Apply normalization and activation to c_code before passing it to fully connected layer
    if use_pixel_norm:
      h_c_code = pixel_norm(h_c_code, axis=1)
    h_code = tf.concat([h_code, h_c_code], 1)
    if use_pixel_norm:
      h_code /= 2 # z and c may be drawn from distributions with different scales, after PN and concat divide by 2 to combine
  else:
    kl_loss = 0

  # Pixelwise normalize latent vector
  if use_pixel_norm:
    h_code = pixel_norm(h_code, axis=1)

  # FC and reshape for convolution
  # [512] -> [16, 512]
  with tf.variable_scope('z'):
    h_code = tf.layers.dense(h_code, 16 * dim * 32)
    h_code = tf.reshape(h_code, [batch_size, 16, dim * 32])

  # First residual block
  # [16, 512] -> [16, 512]
  with tf.variable_scope('l0'):
    h_code = residual_block(h_code, filters=dim * 32, kernel_size=kernel_len, normalization=batchnorm, use_pixel_norm=use_pixel_norm)
    audio_lod = to_audio(h_code, normalization=batchnorm, use_pixel_norm=use_pixel_norm)
    tf.summary.audio('G_audio', nn_upsample(nn_upsample(nn_upsample(nn_upsample(nn_upsample(audio_lod))))), 16000, max_outputs=10, family='G_audio_lod_0')

  # Layer 1
  # [16, 512] -> [64, 256]
  with tf.variable_scope('up1'):
    on_amount = lod-0
    h_code, audio_lod = up_block(h_code, audio_lod=audio_lod, filters=dim * 16, kernel_size=kernel_len, normalization=_batchnorm, on_amount=on_amount, upsample_method=upsample, use_pixel_norm=use_pixel_norm)
    tf.summary.scalar('on_amount', on_amount)
    tf.summary.audio('G_audio', nn_upsample(nn_upsample(nn_upsample(nn_upsample(audio_lod)))), 16000, max_outputs=10, family='G_audio_lod_1')

  # Layer 2
  # [64, 256] -> [256, 128]
  with tf.variable_scope('up2'):
    on_amount = lod-1
    h_code, audio_lod = up_block(h_code, audio_lod=audio_lod, filters=dim * 8, kernel_size=kernel_len, normalization=_batchnorm, on_amount=on_amount, upsample_method=upsample, use_pixel_norm=use_pixel_norm)
    tf.summary.scalar('on_amount', on_amount)
    tf.summary.audio('G_audio', nn_upsample(nn_upsample(nn_upsample(audio_lod))), 16000, max_outputs=10, family='G_audio_lod_2')

  # Layer 3
  # [256, 128] -> [1024, 64]
  with tf.variable_scope('up3'):
    on_amount = lod-2
    h_code, audio_lod = up_block(h_code, audio_lod=audio_lod, filters=dim * 4, kernel_size=kernel_len, normalization=_batchnorm, on_amount=on_amount, upsample_method=upsample, use_pixel_norm=use_pixel_norm)
    tf.summary.scalar('on_amount', on_amount)
    tf.summary.audio('G_audio', nn_upsample(nn_upsample(audio_lod)), 16000, max_outputs=10, family='G_audio_lod_3')

  # Layer 3
  # [1024, 64] -> [4096, 32]
  with tf.variable_scope('up4'):
    on_amount = lod-3
    h_code, audio_lod = up_block(h_code, audio_lod=audio_lod, filters=dim * 2, kernel_size=kernel_len, normalization=_batchnorm, on_amount=on_amount, upsample_method=upsample, use_pixel_norm=use_pixel_norm)
    tf.summary.scalar('on_amount', on_amount)
    tf.summary.audio('G_audio', nn_upsample(audio_lod), 16000, max_outputs=10, family='G_audio_lod_4')

  # Layer 4
  # [4096, 32] -> [16384, 16] (h_code)
  # [16384, 16] -> [16384, 1] (audio_lod)
  with tf.variable_scope('up5'):
    on_amount = lod-4
    h_code, audio_lod = up_block(h_code, audio_lod=audio_lod, filters=dim, kernel_size=kernel_len, normalization=_batchnorm, on_amount=on_amount, upsample_method=upsample, use_pixel_norm=use_pixel_norm)
    tf.summary.scalar('on_amount', on_amount)
    tf.summary.audio('G_audio', audio_lod, 16000, max_outputs=10, family='G_audio_lod_5')

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
    dim=16,
    use_batchnorm=False,
    use_pixel_norm=True,
    phaseshuffle_rad=0,
    embedding_dim=128):
  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
  else:
    batchnorm = lambda x: x

  if phaseshuffle_rad > 0:
    phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
  else:
    phaseshuffle = lambda x: x

  with tf.variable_scope('ae'):
    if 'D_x/' in tf.get_default_graph().get_name_scope():
      tf.summary.audio('input_audio', x, 16000, max_outputs=10, family='D_audio_lod_5')

    # Layer 0
    # [16384, 1] -> [16384, 16] (audio_lod)
    # [16384, 1] -> [4096, 32] (h_code)
    with tf.variable_scope('do0'):
      on_amount = lod-4
      h_code, audio_lod = down_block(from_audio(x, dim), audio_lod=x, filters=dim * 2, kernel_size=kernel_len, normalization=batchnorm, on_amount=on_amount, use_pixel_norm=use_pixel_norm)
      if 'D_x/' in tf.get_default_graph().get_name_scope():
        tf.summary.audio('audio_downsample', nn_upsample(audio_lod), 16000, max_outputs=10, family='D_audio_lod_4')
        tf.summary.scalar('on_amount', on_amount)
      

    # Layer 1
    # [4096, 32] -> [1024, 64]
    with tf.variable_scope('do1'):
      on_amount = lod-3
      h_code, audio_lod = down_block(h_code, audio_lod=audio_lod, filters=dim * 4, kernel_size=kernel_len, normalization=batchnorm, on_amount=on_amount, use_pixel_norm=use_pixel_norm)
      if 'D_x/' in tf.get_default_graph().get_name_scope():
        tf.summary.audio('audio_downsample', nn_upsample(nn_upsample(audio_lod)), 16000, max_outputs=10, family='D_audio_lod_3')
        tf.summary.scalar('on_amount', on_amount)
    
    # Layer 2
    # [1024, 64] -> [256, 128]
    with tf.variable_scope('do2'):
      on_amount = lod-2
      h_code, audio_lod = down_block(h_code, audio_lod=audio_lod, filters=dim * 8, kernel_size=kernel_len, normalization=batchnorm, on_amount=on_amount, use_pixel_norm=use_pixel_norm)
      if 'D_x/' in tf.get_default_graph().get_name_scope():
        tf.summary.audio('audio_downsample', nn_upsample(nn_upsample(nn_upsample(audio_lod))), 16000, max_outputs=10, family='D_audio_lod_2')
        tf.summary.scalar('on_amount', on_amount)

    # Layer 3
    # [256, 128] -> [64, 256]
    with tf.variable_scope('do3'):
      on_amount = lod-1
      h_code, audio_lod = down_block(h_code, audio_lod=audio_lod, filters=dim * 16, kernel_size=kernel_len, normalization=batchnorm, on_amount=on_amount, use_pixel_norm=use_pixel_norm)
      if 'D_x/' in tf.get_default_graph().get_name_scope():
        tf.summary.audio('audio_downsample', nn_upsample(nn_upsample(nn_upsample(nn_upsample(audio_lod)))), 16000, max_outputs=10, family='D_audio_lod_1')
        tf.summary.scalar('on_amount', on_amount)

    # Layer 4
    # [64, 256] -> [16, 512]
    with tf.variable_scope('do4'):
      on_amount = lod-0
      h_code, audio_lod = down_block(h_code, audio_lod=audio_lod, filters=dim * 32, kernel_size=kernel_len, normalization=batchnorm, on_amount=on_amount, use_pixel_norm=use_pixel_norm)
      if 'D_x/' in tf.get_default_graph().get_name_scope():
        tf.summary.audio('audio_downsample', nn_upsample(nn_upsample(nn_upsample(nn_upsample(nn_upsample(audio_lod))))), 16000, max_outputs=10, family='D_audio_lod_0')
        tf.summary.scalar('on_amount', on_amount)

    return h_code, audio_lod


"""
  Input: [None, 16384, 1]
  Output: [None] (linear output)
"""
def WaveGANDiscriminator(
    x,
    lod,
    kernel_len=24,
    dim=16,
    use_batchnorm=False,
    use_pixel_norm=True,
    phaseshuffle_rad=0,
    context_embedding=None,
    embedding_dim=128,
    use_extra_uncond_output=False):
  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
  else:
    batchnorm = lambda x: x

  x_code, _ = encode_audio(x, lod, kernel_len, dim, use_batchnorm, phaseshuffle_rad, embedding_dim)
  
  # if (context_embedding is not None):
  #   with tf.variable_scope('cond'):
  #     # Add conditioning to audio encoding
  #     c_code = compress_embedding(context_embedding, embedding_dim)
  #     cond_out = add_conditioning(x_code, c_code)
  #     output = cond_out
  # else:
    # output = x_code
  output = x_code

  # <-- TODO: Minibatch std deviation layer goes here

  # Final residual block
  # [16, 512] -> [16, 512]
  with tf.variable_scope('frb'):
    output = residual_block(output, filters=output.get_shape().as_list()[2], kernel_size=kernel_len, normalization=batchnorm, stride=1, padding='same', use_pixel_norm=use_pixel_norm)
  if (use_extra_uncond_output) and (context_embedding is not None):
    with tf.variable_scope('frb_u'):
      uncond_out = residual_block(x_code, filters=x_code.get_shape().as_list()[2], kernel_size=kernel_len, normalization=batchnorm, stride=1, padding='same', use_pixel_norm=use_pixel_norm)

  # FC 1
  # [16, 512] -> [512]
  batch_size = tf.shape(x)[0]
  with tf.variable_scope('fc1'):
    output = tf.reshape(output, [batch_size, -1]) # Flatten
    output = batchnorm(output)
    output = lrelu(output)
    if use_pixel_norm:
      output = pixel_norm(output, axis=1)

    # Add conditioning
    if (context_embedding is not None):
      c_code = compress_embedding(context_embedding, embedding_dim)
      c_code = batchnorm(c_code)
      c_code = lrelu(c_code)
      if use_pixel_norm:
        c_code = pixel_norm(c_code, axis=1)
      output = tf.concat([output, c_code], 1)
      if use_pixel_norm:
        output /= 2 # z and c may be drawn from distributions with different scales, after PN and concat divide by 2 to combine

    output = tf.layers.dense(output, dim * 32)

    if (use_extra_uncond_output) and (context_embedding is not None):
      uncond_out = tf.reshape(uncond_out, [batch_size, -1]) # Flatten
      uncond_out = batchnorm(uncond_out)
      uncond_out = lrelu(uncond_out)
      if use_pixel_norm:
        uncond_out = pixel_norm(uncond_out, axis=1)
      uncond_out = tf.layers.dense(uncond_out, dim * 32)

  # FC 2
  # [512] -> [1]
  with tf.variable_scope('fc2'):
    output = batchnorm(output)
    output = lrelu(output)
    if use_pixel_norm:
      output = pixel_norm(output, axis=1)
    output = tf.layers.dense(output, 1)

    if (use_extra_uncond_output) and (context_embedding is not None):
      with tf.variable_scope('fc2_u'):
        uncond_out = batchnorm(uncond_out)
        uncond_out = lrelu(uncond_out)
        if use_pixel_norm:
          uncond_out = pixel_norm(uncond_out, axis=1)
        uncond_out = tf.layers.dense(uncond_out, 1)
      return [output, uncond_out]
    else:
      return [output]
  
  # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training
