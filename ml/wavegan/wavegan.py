import tensorflow as tf


def custom_conv1d(
    inputs,
    filters,
    kernel_width,
    stride=4,
    padding='same',
    upsample=None):
  if upsample == 'zeros':
    return tf.layers.conv2d_transpose(
        tf.expand_dims(inputs, axis=1),
        filters,
        (1, kernel_width),
        strides=(1, stride),
        padding='same'
        )[:, 0]
  elif upsample == 'nn':
    batch_size = tf.shape(inputs)[0]
    _, w, nch = inputs.get_shape().as_list()

    x = inputs

    x = tf.expand_dims(x, axis=1)
    x = tf.image.resize_nearest_neighbor(x, [1, w * stride])
    x = x[:, 0]

    return tf.layers.conv1d(
        x,
        filters,
        kernel_width,
        1,
        padding='same')
  else:
    return tf.layers.conv1d(
        inputs, 
        filters, 
        kernel_width, 
        strides=stride, 
        padding='same')


def lrelu(inputs, alpha=0.2):
  return tf.maximum(alpha * inputs, inputs)


def compress_embedding(embedding, embed_size):
  """
  Return compressed embedding for discriminator
  c: Embedded context to reduce, this should be a [batch_size x N] tensor
  embed_size: The size of the new embedding
  returns: [batch_size x embed_size] tensor
  """
  with tf.variable_scope('reduce_embed'):
    embedding = lrelu(tf.layers.dense(embedding, embed_size))
    # embedding = tf.layers.dropout(embedding)
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
      # params = tf.layers.dropout(params, 0.5 if train else 0)
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
  with tf.variable_scope('MinibatchStddev'):
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
    kernel_len=24,
    dim=64,
    use_batchnorm=False,
    upsample='zeros',
    train=False,
    context_embedding=None,
    embedding_dim=128):
  batch_size = tf.shape(z)[0]

  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
  else:
    batchnorm = lambda x: x

  if (context_embedding is not None):
    # Reduce or expand context embedding to be size [embedding_dim]
    c = compress_embedding(context_embedding, embedding_dim)
    kl_loss = 0
    output = tf.concat([z, c], 1)
  else:
    output = z
    kl_loss = 0

  # FC and reshape for convolution
  # [100] -> [16, 1024]
  with tf.variable_scope('z_project'):
    output = tf.layers.dense(output, 4 * 4 * dim * 16)
    output = tf.reshape(output, [batch_size, 16, dim * 16])
    output = batchnorm(output)
  output = tf.nn.relu(output)

  # Layer 0
  # [16, 1024] -> [64, 512]
  with tf.variable_scope('upconv_0'):
    output = custom_conv1d(output, dim * 8, kernel_len, 4, upsample=upsample)
    output = batchnorm(output)
  output = tf.nn.relu(output)

  # Layer 1
  # [64, 512] -> [256, 256]
  with tf.variable_scope('upconv_1'):
    output = custom_conv1d(output, dim * 4, kernel_len, 4, upsample=upsample)
    output = batchnorm(output)
  output = tf.nn.relu(output)

  # Layer 2
  # [256, 256] -> [1024, 128]
  with tf.variable_scope('upconv_2'):
    output = custom_conv1d(output, dim * 2, kernel_len, 4, upsample=upsample)
    output = batchnorm(output)
  output = tf.nn.relu(output)

  # Layer 3
  # [1024, 128] -> [4096, 64]
  with tf.variable_scope('upconv_3'):
    output = custom_conv1d(output, dim, kernel_len, 4, upsample=upsample)
    output = batchnorm(output)
  output = tf.nn.relu(output)

  # Layer 4
  # [4096, 64] -> [16384, 1]
  with tf.variable_scope('upconv_4'):
    output = custom_conv1d(output, 1, kernel_len, 4, upsample=upsample)
  output = tf.nn.tanh(output)

  # Automatically update batchnorm moving averages every time G is used during training
  if train and use_batchnorm:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if len(update_ops) != 10:
      raise Exception('Other update ops found in graph')
    with tf.control_dependencies(update_ops):
      output = tf.identity(output)

  return output, kl_loss


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
    kernel_len=24,
    dim=64,
    use_batchnorm=False,
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

  with tf.variable_scope('encode_audio'):
    # Layer 0
    # [16384, 1] -> [4096, 64]
    output = x
    with tf.variable_scope('downconv_0'):
      output = tf.layers.conv1d(output, dim, kernel_len, 4, padding='SAME')
      output = batchnorm(output)
      output = lrelu(output)
      output = phaseshuffle(output)

    # Layer 1
    # [4096, 64] -> [1024, 128]
    with tf.variable_scope('downconv_1'):
      output = tf.layers.conv1d(output, dim * 2, kernel_len, 4, padding='SAME')
      output = batchnorm(output)
      output = lrelu(output)
      output = phaseshuffle(output)

    # Layer 2
    # [1024, 128] -> [256, 256]
    with tf.variable_scope('downconv_2'):
      output = tf.layers.conv1d(output, dim * 4, kernel_len, 4, padding='SAME')
      output = batchnorm(output)
      output = lrelu(output)
      output = phaseshuffle(output)

    # Layer 3
    # [256, 256] -> [64, 512]
    with tf.variable_scope('downconv_3'):
      output = tf.layers.conv1d(output, dim * 8, kernel_len, 4, padding='SAME')
      output = batchnorm(output)
      output = lrelu(output)
      output = phaseshuffle(output)

    # Layer 4
    # [64, 512] -> [16, 1024]
    with tf.variable_scope('downconv_4'):
      output = tf.layers.conv1d(output, dim * 16, kernel_len, 4, padding='SAME')
      output = batchnorm(output)
      output = lrelu(output)

      # Flatten
    # [16, 1024] -> [16384]
    batch_size = tf.shape(x)[0]
    output = tf.reshape(output, [batch_size, -1])

    return output


"""
  Input: [None, 16384, 1]
  Output: [None] (linear output)
"""
def WaveGANDiscriminator(
    x,
    kernel_len=24,
    dim=64,
    use_batchnorm=False,
    phaseshuffle_rad=0,
    context_embedding=None,
    embedding_dim=128,
    use_extra_uncond_output=False):

  x_code = encode_audio(x, kernel_len, dim, use_batchnorm, phaseshuffle_rad, embedding_dim)
  
  if (context_embedding is not None):
    with tf.variable_scope('conditional'):
      cond_out = x_code

      # Concat context embeddings
      # [16384] -> [16384 + embedding_dim]
      c = compress_embedding(context_embedding, embedding_dim)
      cond_out = tf.concat([cond_out, c], 1)

      # FC
      # [16384 + embedding_dim] -> [1024]
      with tf.variable_scope('FC'):
        cond_out = tf.layers.dense(cond_out, dim * 16)
        cond_out = lrelu(cond_out)
        output = cond_out
  else:
    output = x_code

  # Connect to single logit
  # [16384] -> [1]
  with tf.variable_scope('output'):
    output = tf.layers.dense(output, 1)
    if (use_extra_uncond_output) and (context_embedding is not None):
      uncond_out = tf.layers.dense(x_code, 1)
      return [output, uncond_out]
    else:
      return [output]

  # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training
