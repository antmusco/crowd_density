"""
Author: Anthony G. Musco
Email:  amusco@cs.stonybrook.edu
Date:   05/01/2017

This file contains the operations to construct a counting CNN. The architecture
of the net is defined by the `inference()` function.

Based off the Hyda CNN model by Daniel Onoro-Rubio and Roberto J. Lopenz
Sastre: https://github.com/gramuah/ccnn
"""

import os
import time
from   datetime import datetime
import numpy as np
import tensorflow as tf

#==============================================================================
# Public
#==============================================================================

def inference(patches, is_train, cfg):
    """ 
    Passes the patches through the CCNN to produce the learned density maps.

    Note that this function defines the architecture of the CCNN as a series of
    convolutional and pooling layers, with different initialization parameters.

    Constructs the net if 'is_train' is True, otherwise re-uses the weights
    learned during training to apply to validation or test set in the same
    graph.

    Args:
        patches: Image patches returned from inputs(). 4D tensor of size
            [BATCH_SIZE, CNN_INPUT_WIDTH, CNN_INPUT_WIDTH, 3].
        is_train: True if we should create new variables to train, False if we
            are using pre-existing variables.

    Returns:
        Predicted density maps of size [BATCH_SIZE, CNN_OUTPUT_WIDTH,
            CNN_OUTPUT_WIDTH]
    """
    # Conv 1 with batch normalization
    conv1 = _conv_layer(patches, shape=[7, 7, 3, 32], stddev=5e-2, decay=0.0,
        bias=0.0, name='conv1', is_train=is_train)
    norm1 =_batch_norm_layer(conv1, depth=32, name='norm1', is_train=is_train)

    # Pooling 1
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME', name='pool1')

    # Conv 2 with batch normalization
    conv2 = _conv_layer(pool1, shape=[7, 7, 32, 32], stddev=5e-2, decay=0.0,
        bias=0.1, name='conv2', is_train=is_train)
    norm2 = _batch_norm_layer(conv2, depth=32, name='norm2', is_train=is_train)

    # Pooling 2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME', name='pool2')

    # Conv 3 with batch normalization
    conv3 = _conv_layer(pool2, shape=[5, 5, 32, 64], stddev=5e-2, decay=0.0,
        bias=0.1, name='conv3', is_train=is_train)
    norm3 = _batch_norm_layer(conv3, depth=64, name='norm3', is_train=is_train)

    # Conv 4 with batch normalization
    conv4 = _conv_layer(norm3, shape=[1, 1, 64, 1000], stddev=5e-2, decay=0.0,
        bias=0.1, name='conv4', is_train=is_train)
    norm4 = _batch_norm_layer(conv4, depth=1000, name='norm4', is_train=is_train)

    # Conv 5 with batch normalization
    conv5 = _conv_layer(norm4, shape=[1, 1, 1000, 400], stddev=5e-2, decay=0.0,
        bias=0.1, name='conv5', is_train=is_train)
    norm5 = _batch_norm_layer(conv5, depth=400, name='norm5', is_train=is_train)

    # Conv 6 output
    return  _conv_layer(norm5, shape=[1, 1, 400, 1], stddev=5e-2, decay=0.0,
        bias=0.1, name='conv6', is_train=is_train)


def loss(predict, truth):
    """
    Add L2Loss to all the trainable variables.

    Args:
        predict: Predicted density maps for each patch. 3D Tensor of size
            [BATCH_SIZE, CNN_OUTPUT_SIZE, CNN_OUTPUT_SIZE]
        truth: Actual density maps for each patch. 3D Tensor of size
            [BATCH_SIZE, CNN_OUTPUT_SIZE, CNN_OUTPUT_SIZE]

    Returns:
        Loss tensor of type float.
    """
    # L2 loss between images.
    diff = tf.subtract(predict, truth)
    l2_loss = tf.nn.l2_loss(t=diff, name='l2_loss')
    tf.add_to_collection("losses", l2_loss)
    # Combine the l2_loss with the weight losses.
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def miscount(predict, truth, totals=False):
    """ 
    Calculate the miscounts for the validation set. The miscount is defined to
    be sum(predict) - sum(truth). 

    Args:
        predict: Predicted density maps for each patch. 3D Tensor of size
            [BATCH_SIZE, CNN_OUTPUT_SIZE, CNN_OUTPUT_SIZE]
        truth: Actual density maps for each patch. 3D Tensor of size
            [BATCH_SIZE, CNN_OUTPUT_SIZE, CNN_OUTPUT_SIZE]
        totals: If True, return the list of miscounts for all images. If False,
            return the averages.
    Returns:
        Miscount value between predicted and ground truth density maps.
    """
    # Miscounts over batches.
    predict_cnt   = tf.reduce_sum(predict, [1,2])
    truth_cnt     = tf.reduce_sum(truth, [1,2])
    miscounts     = tf.subtract(predict_cnt, truth_cnt)
    if totals:
        return truth_cnt, predict_cnt, miscounts
    else:
        # Average miscount for the batch.
        miscounts_avg   = tf.reduce_mean(miscounts)
        truth_cnt_avg   = tf.reduce_mean(truth_cnt)
        predict_cnt_avg = tf.reduce_mean(predict_cnt)
        return truth_cnt_avg, predict_cnt_avg, miscounts_avg


def optimize(total_loss, global_step, epoch_size, cfg):
    """Optimize the CCNN model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
        total_loss:  Total loss from loss().
        global_step: Integer Variable counting the number of training steps
            processed.
        epoch_size: Number of examples per epoch.
        cfg: Configuration settings.
    Returns:
        train_op: op for training.
    """
    # Start decaying after CNN_DECAY_START epochs of examples.
    decay_steps = int(epoch_size * cfg.CNN_DECAY_START)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(cfg.CNN_INIT_LEARN_RATE, global_step,
            decay_steps, cfg.CNN_DECAY_FACTOR, staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_summaries(total_loss, tf.get_collection('losses'),
          'loss_avg')

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
      optimizer = tf.train.GradientDescentOptimizer(lr)
      grads = optimizer.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        cfg.CNN_MOV_AVG_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Create a dummy no_op node to return representing the training operation.
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      train_op = tf.no_op(name='train')
    return train_op


def inputs(example_set, scale, cfg):
    """
    Construct input for CCNN evaluation using the Reader ops.

    Args:
        example_set: Either 'train', 'valid', or 'test'
        scale: The scale to train at.
        cfg: The configuration parameters.

    Returns:
        patches: Image patches. 4D tensor of [BATCH_SIZE, CNN_INPUT_WIDTH,
            CNN_INPUT_WIDTH, 3] size.
        densities: Density maps. 3D tensor of [BATCH_SIZE, CNN_OUTPUT_WIDTH,
            CNN_OUTPUT_WIDTH] size.
    """

    # List, count, and verify the files.
    list_file = cfg.EXL_SET_FMT % (example_set, scale)
    list_path = os.path.join(cfg.EXL_SET_DIR, list_file)
    filenames = np.loadtxt(list_path, dtype='str')
    count = 0
    for f in filenames:
        count += 1
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    fn_tensor = tf.constant(filenames)
    filename_queue = tf.train.string_input_producer(fn_tensor, name=example_set
        + '_input')

    # Parse examples from files in the filename queue.
    patch, density = _parseExamples(filename_queue, cfg.CNN_INPUT_WIDTH,
        cfg.CNN_OUTPUT_WIDTH)

    # Pre-process the patches (0 mean, 1 std).
    with tf.name_scope('pre-process'):
        adj_patch = tf.image.per_image_standardization(patch)

    # Ensure that the random shuffling has good mixing properties.
    total_examples = count * cfg.BATCH_SIZE * cfg.BATCH_PER_CHUNK
    min_queue_pct  = 0.4
    min_queue_examples = int(total_examples * min_queue_pct)

    # Generate batch of patches/densities.
    return _genBatch(adj_patch, density, min_queue_examples, cfg.BATCH_SIZE,
        shuffle=False)


def serializeExample(patch, density):
    """
    Serializes a single example to be written to a file. An example consists of
    a patch/density image pair.

    Args:
        patch: Image patch.
        density: Density map of the image patch.

    Return:
        Serialized tf.train.Example instance ready to be written to a file.
    """
    return _serializeExample(patch, density)


def deserializeExample(serialized_example):
    """
    De-serializes a single example read from to a file. An example consists of
    a patch/density image pair.

    Args:
        example: Serialized example read from a file.

    Return:
        patch: Image patch.
        density: Density map of the image patch.
    """
    # Parse out the serialized example.
    example = tf.parse_single_example(
        serialized_example,
        features={
            'patch':   tf.FixedLenFeature([], tf.string),
            'density': tf.FixedLenFeature([], tf.string),
        },
    )
    # Decode the string features as list of floats.
    patch   = tf.decode_raw(example['patch'],   tf.float32)
    density = tf.decode_raw(example['density'], tf.float32)
    return (patch, density)


class LoggerHook(tf.train.SessionRunHook):
  """ Logs loss and runtime for a training or test session. """

  def __init__(self, name, loss, cfg):
    self._name = name
    self._loss = loss 
    self._cfg  = cfg

  def begin(self):
    self._step = -1
    self._time = time.time()

  def before_run(self, run_context):
    self._step += 1
    return tf.train.SessionRunArgs(self._loss)  # Asks for latest loss value.

  def after_run(self, run_context, run_values):
    if self._step % self._cfg.LOG_FREQUENCY == 0:
      now        = time.time()
      duration   = now - self._time
      self._time = now

      loss_value = run_values.results
      ex_per_sec = self._cfg.LOG_FREQUENCY * self._cfg.BATCH_SIZE / duration

      print '%s: step %d, %s = %.2f (%.1f examples/sec' % \
            (datetime.now(), self._step, self._name, loss_value, ex_per_sec)

#==============================================================================
# Private
#==============================================================================

def _bytesFeature(value):
    """ Shorthand request for a TF BytesList feature. """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _serializeExample(patch, density):
    """
    Serializes a single example to be written to a file. An example consists of
    a patch/density image pair.

    Args:
        patch: Image patch.
        density: Density map of the image patch.

    Return:
        Serialized tf.train.Example instance ready to be written to a file.
    """
    features = tf.train.Features(feature = {
        'patch':   _bytesFeature(patch.tostring()),
        'density': _bytesFeature(density.tostring()),
    })
    return tf.train.Example(features=features)


def _parseExamples(filename_queue, input_width, output_width):
    """
    Reads and parses examples from the TFRecord Examples in the files.

    Args:
        filename_queue: A queue of strings with the filenames to read from.

    Returns:
        An object representing a single example, with the following fields:
            patch: a [CNN_INPUT_WIDTH, CNN_INPUT_WIDTH, 3] size tensor.
            density: a [CNN_OUTPUT_WIDTH, CNN_OUTPUT_WIDTH] size tensor.
    """
    # Scoping to make TensorBoard graph readable.
    with tf.name_scope('parse'):

        # Read a single serialized example and deserialize it.
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        patch, density = deserializeExample(serialized_example)
        
        # Reshape patch and density.
        patch_shape   = tf.constant([input_width, input_width, 3])
        density_shape = tf.constant([output_width, output_width, 1])
        patch         = tf.reshape(patch, patch_shape)
        density       = tf.reshape(density, density_shape)
        
        # Return example.
        return patch, density


def _genBatch(patch, density, min_queue_examples, batch_size, shuffle):
    """
    Construct a queued batch of patches and densities.

    Args:
        patch: 3D Tensor of [CNN_INPUT_WIDTH, CNN_INPUT_WIDTH, 3] of type.float32.
        density: 2D Tensor of [CNN_OUTPUT_WIDTH, CNN_OUTPUT_WIDTH] type.float32
        min_queue_examples: int32, minimum number of samples to retain in the queue
            that provides the batches of examples.
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
        patches: Image patches. 4D tensor of [BATCH_SIZE, CNN_INPUT_WIDTH,
            CNN_INPUT_WIDTH, 3] size.
        densities: Density maps. 3D tensor of [BATCH_SIZE, CNN_OUTPUT_WIDTH,
            CNN_OUTPUT_WIDTH] size.
    """
    # Create a queue that shuffles the examples, and then read 'batch_size'
    # patches and densities from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        patches, densities = tf.train.shuffle_batch(
            [patch, density],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        patches, densities = tf.train.batch(
            [patch, density],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)
    return patches, densities


def _variable_with_weight_decay(name, shape, stddev, decay):
    """
    Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name:   Name of the variable
      shape:  List of ints
      stddev: Standard deviation of a truncated Gaussian
      decay:  Add L2Loss weight decay multiplied by this float. If None, weight
                  decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    var = tf.get_variable(name, shape=shape,
          initializer=tf.truncated_normal_initializer(stddev=stddev,
          dtype=tf.float32))
    if decay is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), decay, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _batch_norm_layer(inputs, depth, name, is_train):
    """
    Batch normalization layer.
    Args:
        inputs:  Tensor of size [batch_size, INPUT_HEIGHT, INPUT_CHANNELS,
                    OUTPUT_CHANNELS]
        depth:   Integer, depth of input maps
        name:    Name of this batch normalization layer.
        is_train: True if we are creating trainable variables.
    Return:
        norm: Batch-normalized maps
    """
    # If we are training, create variables.
    if is_train:
        with tf.variable_scope(name) as scope:
            # Create trainable scale and shift parameters.
            beta  = tf.Variable(tf.constant(0.0, shape=[depth]), name='beta',
                        trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[depth]), name='gamma',
                        trainable=True)
            # Extract the mean and variance from the batch.
            mean, \
            var   = tf.nn.moments(inputs, [0,1,2], name='moments')
            # Apply moving average to the moments.
            ema    = tf.train.ExponentialMovingAverage(decay=0.5)
            ema_op = ema.apply([mean, var])
            # Wait for op to complete, extract new moments.
            with tf.control_dependencies([ema_op]):
                mean = tf.identity(mean)
                var  = tf.identity(var)
            # Normalize the batch.
            norm   = tf.nn.batch_normalization(inputs, mean, var, beta, gamma,
                 1e-3)
            return norm
    # Otherwise, re-use pre-trained variables.
    else:
        with tf.variable_scope(name, reuse=True) as scope:
            # Grab variables from pre-existing scope.
            beta  = tf.get_variable('beta')
            gamma = tf.get_variable('gamma')
            # Extract the mean and variance from the batch.
            mean, \
            var   = tf.nn.moments(inputs, [0,1,2], name='moments')
            # Apply moving average to the moments.
            ema   = tf.train.ExponentialMovingAverage(decay=0.5)
            mean  = ema.average(mean)
            var   = ema.average(var)
            norm  = tf.nn.batch_normalization(inputs, mean, var, beta, gamma,
                1e-3)
            return norm


def _conv_layer(inputs, shape, stddev, decay, bias, name, is_train):
    """ 
    Applies a convolutional layer to the inputs.

    Args:
        inputs: Tensor of size [batch_size, INPUT_HEIGHT, INPUT_CHANNELS,
            OUTPUT_CHANNELS]
        shape:  A list of [KERN_HEIGHT, KERN_WIDTH, INPUT_CHANNELS,
            OUTPUT_CHANNELS]
        stddev: Standard deviation of the truncated normal to initialize
            kernel.
        decay:  Initial weight decay value.
        bias:   Initial value for the bias
        name:   Name of this convolutional layer.
        is_train:  True if we are training the variables.

    Returns:
        actv: The activations after applying the convolution to inputs.
    """
    # If we are training, create variables.
    if is_train:
        with tf.variable_scope(name) as scope:
            kern = _variable_with_weight_decay('weights', shape, stddev, decay)
            conv = tf.nn.conv2d(inputs, kern, strides=[1,1,1,1], padding='SAME')
            bias = tf.get_variable('biases', shape=[shape[-1]],
                      initializer=tf.constant_initializer(bias))
            return tf.nn.relu(tf.nn.bias_add(conv, bias), name=scope.name)
    # Otherwise, re-use pre-trained variables.
    else:
        with tf.variable_scope(name, reuse=True) as scope:
            kern = tf.get_variable('weights')
            conv = tf.nn.conv2d(inputs, kern, strides=[1,1,1,1], padding='SAME')
            bias = tf.get_variable('biases')
            return tf.nn.relu(tf.nn.bias_add(conv, bias), name=scope.name)


def _add_summaries(total, collection, name):
    """
    Add summaries for losses.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total: Total loss from loss().

    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average for the 
    averages = tf.train.ExponentialMovingAverage(0.9, name=name)
    averages_op = averages.apply(collection + [total])

    # Attach a scalar summary to all individual losses and the total loss.
    for l in collection + [total]:
      # Name each loss as '_raw' and name the moving average version of the loss
      # as the original loss name.
      tf.summary.scalar(l.op.name + '_raw', l)
      tf.summary.scalar(l.op.name, averages.average(l))
    return averages_op
