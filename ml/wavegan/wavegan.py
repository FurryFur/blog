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


def gmau(inputs):
  """
  Gated Multi-Activation Unit
  """
  with tf.variable_scope('gmau'):
    #regularizer = tf.contrib.layers.l1_regularizer(scale=0.00)
    #linear_gate = tf.get_variable('linear_gate', [inputs.shape[-1]], regularizer=regularizer)
    #tanh_gate = tf.get_variable('tanh_gate', [inputs.shape[-1]], regularizer=regularizer, initializer=tf.truncated_normal_initializer(0.33, 0.1))
    #relu_gate = tf.get_variable('relu_gate', [inputs.shape[-1]], regularizer=regularizer)
    #lrelu_gate = tf.get_variable('lrelu_gate', [inputs.shape[-1]], regularizer=regularizer, initializer=tf.truncated_normal_initializer(0.33, 0.1))
    #elu_gate = tf.get_variable('elu_gate', [inputs.shape[-1]], regularizer=regularizer)
    #sin_gate = tf.get_variable('sin_gate', [inputs.shape[-1]], regularizer=regularizer, initializer=tf.truncated_normal_initializer(0.33, 0.1))

    # Gate linear ops
    if inputs.get_shape().ndims == 3:
      tanh_gate = tf.layers.conv1d(inputs, 32, 1, padding="SAME")
      tanh_gate = tf.layers.conv1d(tanh_gate, 32, 6, padding="SAME")
      tanh_gate = tf.layers.conv1d(tanh_gate, inputs.shape[-1], 1, padding="SAME")

      lrelu_gate = tf.layers.conv1d(inputs, 32, 1, padding="SAME")
      lrelu_gate = tf.layers.conv1d(lrelu_gate, 32, 6, padding="SAME")
      lrelu_gate = tf.layers.conv1d(lrelu_gate, inputs.shape[-1], 1, padding="SAME")

      sin_gate = tf.layers.conv1d(inputs, 32, 1, padding="SAME")
      sin_gate = tf.layers.conv1d(sin_gate, 32, 6, padding="SAME")
      sin_gate = tf.layers.conv1d(sin_gate, inputs.shape[-1], 1, padding="SAME")
    else:
      tanh_gate = tf.layers.dense(inputs, inputs.shape[-1])
      lrelu_gate = tf.layers.dense(inputs, inputs.shape[-1])
      sin_gate = tf.layers.dense(inputs, inputs.shape[-1])

    # Gate non-linear ops
    tanh_gate = tf.sigmoid(tanh_gate)
    lrelu_gate = tf.sigmoid(lrelu_gate)
    sin_gate = tf.sigmoid(sin_gate)

    # Apply gated activation functions
    #linear_out = linear_gate * inputs
    tanh_out = tanh_gate * tf.nn.tanh(inputs)
    #relu_out = relu_gate * tf.nn.relu(inputs)
    lrelu_out = lrelu_gate * lrelu(inputs)
    #elu_out = elu_gate * tf.nn.elu(inputs)
    sin_out = sin_gate * tf.sin(inputs)

    return tanh_out + lrelu_out + sin_out


def residual_unit(    
    inputs,
    filters,
    kernel_width=24,
    stride=1,
    padding='same',
    upsample=None,
    activation=gmau,
    batchnorm_fn=lambda x : x):
  # Shortcut connection
  if (upsample is not None) or (inputs.shape[-1] != filters) or (stride != 1):
    shortcut = custom_conv1d(inputs, filters, 1, stride, padding, upsample)
  else:
    shortcut = inputs

  # Conv + Activation
  output = custom_conv1d(inputs, filters, kernel_width, stride, padding, upsample)
  output = batchnorm_fn(output)
  output = activation(output)

  return output + shortcut


def dense_block(
    inputs,
    num_units,
    filters_per_unit=32,
    kernel_width=24,
    out_dim=None,
    activation=gmau,
    batchnorm_fn=lambda x: x):
  """
  input: Input tensor
  num_units: Number of internal convolution units in the dense block
  batchnorm_fn: A function for normalizing each layer
  filters_per_unit: The number of filters produced by each unit, these are stacked together
  so the final output filters will be num_units * filters_per_unit + input filters
  out_dim: Settings this will override the output dimension using 1 by 1 convolution at end of block
  kernel_width: The size of the kernel used by each convolutional unit
  """
  output = inputs
  for i in range(num_units):
    with tf.variable_scope("unit_{}".format(i)):
      unit_out = tf.layers.conv1d(output, filters_per_unit, kernel_width, padding="SAME")
      unit_out = batchnorm_fn(unit_out)
      unit_out = activation(unit_out)
      output = tf.concat([output, unit_out], 2)

  # Resize out dimensions on request
  if out_dim is not None:
    with tf.variable_scope("1_by_1"):
      output = tf.layers.conv1d(inputs, out_dim, 1, padding="SAME")
      output = batchnorm_fn(output)
      return activation(output)
  else:
    return output


def inception_block(inputs, filters_internal=64, kernel_width=24):
  shortcut = inputs

  filter1 = tf.layers.conv1d(inputs, filters_internal, 1, padding="SAME")
  filter1 = tf.layers.conv1d(filter1, filters_internal, kernel_width // 4, padding="SAME")

  filter2 = tf.layers.conv1d(inputs, filters_internal, 1, padding="SAME")
  filter2 = tf.layers.conv1d(filter2, filters_internal, kernel_width // 2, padding="SAME")

  filter3 = tf.layers.conv1d(inputs, filters_internal, 1, padding="SAME")
  filter3 = tf.layers.conv1d(filter3, filters_internal, kernel_width, padding="SAME")

  concat = tf.concat([filter1, filter2, filter3], 2)
  output = tf.layers.conv1d(concat, inputs.shape[-1], 1, padding="SAME")

  return shortcut + output


def compress_embedding(embedding, embed_size):
  """
  Return compressed embedding for discriminator
  c: Embedded context to reduce, this should be a [batch_size x N] tensor
  embed_size: The size of the new embedding
  returns: [batch_size x embed_size] tensor
  """
  with tf.variable_scope('reduce_embed'):
    embedding = gmau(tf.layers.dense(embedding, embed_size))
    return tf.layers.dropout(embedding)


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
      params = gmau(tf.layers.dense(embedding, 2 * embed_size))
      params = tf.layers.dropout(params, 0.5 if train else 0)
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
    train - A flag 
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


"""
  Input: [None, 100]
  Output: [None, 16384, 1], kl_loss for regularizing context embedding sample distribution
"""
def WaveGANGenerator(
    z,
    kernel_len=25,
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
    c, kl_loss = sample_context_embeddings(context_embedding, embedding_dim, train=train)
    output = tf.concat([z, c], 1)
  else:
    output = z

  # FC and reshape for convolution
  # [100 + context_embedding size] -> [16, 128]
  with tf.variable_scope('z_project'):
    output = tf.layers.dense(output, 4 * 4 * dim * 2)
    output = tf.reshape(output, [batch_size, 16, dim * 2])
    output = batchnorm(output)
    output = gmau(output)

  # Dense 0
  # [16, 128] -> [16, 1024]
  with tf.variable_scope('dense_0'):
    output = dense_block(output, 7, dim * 2, kernel_len, batchnorm_fn=batchnorm)

  # Layer 0
  # [16, 1024] -> [64, 256]
  with tf.variable_scope('upconv_0'):
    output = residual_unit(output, dim * 4, kernel_len, 4, batchnorm_fn=batchnorm, upsample=upsample)
    #output = batchnorm(output)
  #output = tf.nn.relu(output)

  # Dense 1
  # [64, 256] -> [64, 512]
  with tf.variable_scope('dense_1'):
    output = dense_block(output, 4, dim, kernel_len, batchnorm_fn=batchnorm)

  # Layer 1
  # [64, 512] -> [256, 64]
  with tf.variable_scope('upconv_1'):
    output = residual_unit(output, dim, kernel_len, 4, batchnorm_fn=batchnorm, upsample=upsample)
    #output = batchnorm(output)
  #output = tf.nn.relu(output)

  # Dense 2
  # [256, 64] -> [256, 256]
  with tf.variable_scope('dense_2'):
    output = dense_block(output, 3, dim, kernel_len, batchnorm_fn=batchnorm)

  # Layer 2
  # [256, 256] -> [1024, 64]
  with tf.variable_scope('upconv_2'):
    output = residual_unit(output, dim, kernel_len, 4, batchnorm_fn=batchnorm, upsample=upsample)
    #output = batchnorm(output)
  #output = tf.nn.relu(output)

  # Dense 3
  # [1024, 64] -> [1024, 128]
  with tf.variable_scope('dense_3'):
    output = dense_block(output, 1, dim, kernel_len, batchnorm_fn=batchnorm)

  # Layer 3
  # [1024, 128] -> [4096, 64]
  with tf.variable_scope('upconv_3'):
    output = residual_unit(output, dim, kernel_len, 4, batchnorm_fn=batchnorm, upsample=upsample)
    #output = batchnorm(output)
  #output = tf.nn.relu(output)

  # Layer 4
  # [4096, 64] -> [16384, 1]
  with tf.variable_scope('upconv_4'):
    output = residual_unit(output, 1, kernel_len, 4, batchnorm_fn=batchnorm, upsample=upsample)
  #output = tf.nn.tanh(output)

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


"""
  Input: [None, 16384, 1]
  Output: [None] (linear output)
"""
def WaveGANDiscriminator(
    x,
    kernel_len=25,
    dim=64,
    use_batchnorm=False,
    phaseshuffle_rad=0,
    context_embedding=None,
    embedding_dim=128):
  batch_size = tf.shape(x)[0]

  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
  else:
    batchnorm = lambda x: x

  if phaseshuffle_rad > 0:
    phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
  else:
    phaseshuffle = lambda x: x

  # Layer 0
  # [16384, 1] -> [4096, 64]
  output = x
  with tf.variable_scope('downconv_0'):
    output = tf.layers.conv1d(output, dim, kernel_len, 4, padding='SAME')
    output = gmau(output)
  output = phaseshuffle(output)

  # Layer 1
  # [4096, 64] -> [1024, 128]
  with tf.variable_scope('downconv_1'):
    output = tf.layers.conv1d(output, dim * 2, kernel_len, 4, padding='SAME')
    output = batchnorm(output)
    output = gmau(output)
  output = phaseshuffle(output)

  # Layer 2
  # [1024, 128] -> [256, 256]
  with tf.variable_scope('downconv_2'):
    output = tf.layers.conv1d(output, dim * 4, kernel_len, 4, padding='SAME')
    output = batchnorm(output)
    output = gmau(output)
  output = phaseshuffle(output)

  # Layer 3
  # [256, 256] -> [64, 512]
  with tf.variable_scope('downconv_3'):
    output = tf.layers.conv1d(output, dim * 8, kernel_len, 4, padding='SAME')
    output = batchnorm(output)
    output = gmau(output)
  output = phaseshuffle(output)

  # Layer 4
  # [64, 512] -> [16, 1024]
  with tf.variable_scope('downconv_4'):
    output = tf.layers.conv1d(output, dim * 16, kernel_len, 4, padding='SAME')
    output = batchnorm(output)
    output = gmau(output)

  # Flatten
  # [16, 1024] -> [16384]
  output = tf.reshape(output, [batch_size, -1])

  if (context_embedding is not None):
    # Concat context embeddings
    # [16384] -> [16384 + embedding_dim]
    c = compress_embedding(context_embedding, embedding_dim)
    output = tf.concat([output, c], 1)

    # FC
    # [16384 + embedding_dim] -> [1024]
    with tf.variable_scope('FC'):
      output = tf.layers.dense(output, dim * 16)
      output = gmau(output)
    output = tf.layers.dropout(output)

  # Connect to single logit
  # [16384] -> [1]
  with tf.variable_scope('output'):
    output = tf.layers.dense(output, 1)[:, 0]

  # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training

  return output
