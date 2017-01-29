import operator
from functools import reduce

import tensorflow as tf


def weight_variable(shape):
    sigma=0.1
    weight = tf.Variable(tf.truncated_normal(
        shape,
        stddev=sigma,
    ), name='weight')
    tf.summary.histogram('weight', weight)
    return weight


def bias_variable(shape):
    bias = tf.Variable(tf.zeros(shape=shape), name='bias')
    tf.summary.histogram('bias', bias)
    return bias


def conv2d_layer(input_tensor, depth, window, stride=1, activation_fn=tf.nn.relu, pool=None, name=None):
    """Construct a convolutional layer which takes input_layer as input.
    input_layer -> output
    (batch_size, height, width, input_depth) -> (batch_size, height, width, depth)
    :param input_tensor: input tensor
    :param depth: number of convolution images
    :param stride:
    :param window: size of convolutional kernel (side length)
    :param pool: None for no pooling. (ksize, stride) otherwise.
    :param name:
    """

    with tf.name_scope(name):
        assert(input_tensor.get_shape().ndims == 4)
        w = weight_variable([window, window, input_tensor.get_shape().as_list()[-1], depth])
        conv = tf.nn.conv2d(input_tensor, w, strides=[1, stride, stride, 1], padding='SAME')
        b = bias_variable([depth])
        conv = tf.nn.bias_add(conv, b)
        with tf.name_scope('output/' + name):
            output = activation_fn(conv, name='activation')
            if pool is not None:
                (pool_ksize, pool_stride) = pool
                output = tf.nn.max_pool(output, ksize=[1, pool_ksize, pool_ksize, 1], strides=[1, pool_stride, pool_stride, 1], padding='SAME')
        tf.summary.histogram('activation', output)
        tf.add_to_collection(name, output)
        return output


def conv_to_ff_layer(input_tensor):
    """Collapse a convolutional layer into a single dimension (plus batch dimension).
    input -> output
    (batch_size, height, width, input_depth) -> (batch_size, height*width*input_depth)
    :param input_tensor:
    """
    with tf.name_scope('conv_to_ff_layer'):
        shape = input_tensor.get_shape().as_list()
        output_tensor = tf.reshape(input_tensor, [-1, reduce(operator.mul, shape[1:], 1)])
        return output_tensor


def ff_layer(input_tensor, depth, activation_fn=tf.nn.relu, dropout=None, name=None):
    """Construct a fully connected layer.
    input -> output
    (batch_size, input_depth) -> (batch_size, depth)
    :param input_tensor:
    :param depth: number of output nodes
    :param activation_fn:
    :param dropout: None if no dropout layer; keep_prob otherwise
    :param name:
    :param activation: boolean for whether to use the activation function (should be False for last layer)
    :param variables: dict with keys ff_w and ff_b to add weight and bias variables to
    """

    with tf.name_scope(name):
        assert(input_tensor.get_shape().ndims == 2)
        w = weight_variable([input_tensor.get_shape().as_list()[-1], depth])
        b = bias_variable([depth])
        output_tensor = tf.nn.bias_add(tf.matmul(input_tensor, w), b)
        if activation_fn is not None:
            output_tensor = activation_fn(output_tensor, name='activation')
        if dropout is not None:
            keep_prob = dropout
            output_tensor = tf.nn.dropout(output_tensor, keep_prob)
        tf.summary.histogram('activation', output_tensor)
        tf.add_to_collection(name, output_tensor)
        return output_tensor