"""Shamelessly cribbed from the online MIST tutorial.
See more details at
https://www.tensorflow.org/tutorials/mnist/pros/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import os
import matplotlib.pyplot as plt
from visualize import view_images, view_incorrect, one_hot_to_index
from constants import BATCH_SIZE, NUM_EPOCHS, TB_LOGS_DIR

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf


# def weight_variable(shape):
#     # weights initialized with small amount of noise for symmetry breaking, and to prevent 0 gradients
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
#
# def bias_variable(shape):
#     # bias initialized as slight positive number to avoid dead neurons.
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)


def weight_variable(shape, name=''):
    sigma=0.1
    with tf.name_scope('weight'):
        weight = tf.Variable(tf.truncated_normal(
            shape,
            stddev=sigma,
        ))
        tf.histogram_summary('%s/weight' % (name), weight)
        return weight


def bias_variable(shape, name=''):
    with tf.name_scope('bias'):
        bias = tf.Variable(tf.zeros(shape=shape))
        tf.histogram_summary('%s/bias' % (name), bias)
        return bias

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def conv_layer(input_layer, depth, window, stride=1, activation_fn=tf.nn.relu, pool=None, name=None):
    """Construct a convolutional layer which takes input_layer as input.
    input_layer -> output
    (batch_size, height, width, input_depth) -> (batch_size, height, width, depth)
    :param input_layer: input tensor
    :param depth: number of convolution images
    :param stride:
    :param window: size of convolutional kernel (side length)
    :param pool: None for no pooling. (ksize, stride) otherwise.
    :param name:
    """

    with tf.name_scope(name):
        assert(input_layer.get_shape().ndims == 4)
        w = weight_variable([window, window, input_layer.get_shape().as_list()[-1], depth], name)
        conv = tf.nn.conv2d(input_layer, w, strides=[1, stride, stride, 1], padding='SAME')
        b = bias_variable([depth], name)
        conv = tf.nn.bias_add(conv, b)
        with tf.name_scope('output/' + name):
            output = activation_fn(conv, name='activation')
            if pool is not None:
                (pool_ksize, pool_stride) = pool
                output = tf.nn.max_pool(output, ksize=[1, pool_ksize, pool_ksize, 1], strides=[1, pool_stride, pool_stride, 1], padding='SAME')
        tf.histogram_summary('%s/activation' % (name if name is not None else ''), output)
        tf.add_to_collection(name, output)
        return output

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')



# TODO: Save checkpoint file at end of run. Checkpoint file
# TODO: Enable recovery from specified checkpoint file.
def main(_):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    runId = timestamp

    ### Building the Graph
    # None indicates that the first dimension, corresponding to the batch size, can be of any size.
    x = tf.placeholder(tf.float32, [None, 784])
    # With tf.reshape, size of dimension with special value -1 computed so total size remains constant.
    x_image = tf.reshape(x, [-1,28,28,1])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Saving information to tensorboard.
    tf.summary.scalar('Cross Entropy', cross_entropy)
    tf.summary.scalar('Accuracy', accuracy)
    summary = tf.summary.merge_all()

    tensorboard_prefix = os.path.join(TB_LOGS_DIR, runId)
    tensorboard_train_prefix = os.path.join(tensorboard_prefix, 'training')

    with tf.Session() as sess:
        # Initialize summary write for TensorBoard
        os.makedirs(tensorboard_train_prefix)
        train_writer = tf.summary.FileWriter(tensorboard_train_prefix, graph=sess.graph)

        sess.run(tf.global_variables_initializer())
        num_iterations = NUM_EPOCHS * mnist.train.num_examples // BATCH_SIZE
        for step in range(num_iterations):
            batch = mnist.train.next_batch(BATCH_SIZE)
            if step % 100 == 0:
                train_feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0}
                train_accuracy = accuracy.eval(feed_dict=train_feed_dict)
                train_summary = summary.eval(feed_dict=train_feed_dict)
                print("step %d, training accuracy %g"%(step, train_accuracy))
                train_writer.add_summary(train_summary, step)
                train_writer.flush()
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print("test accuracy %g"%accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

        y_final = sess.run(y_conv, feed_dict={x: mnist.test.images,
                                         y_: mnist.test.labels, keep_prob: 1.0})
        view_incorrect(mnist.test.images, one_hot_to_index(y_final), one_hot_to_index(mnist.test.labels), 6, 8, 1)
        view_images(mnist.test.images, one_hot_to_index(y_final), one_hot_to_index(mnist.test.labels), 6, 8, 2)
        plt.show(block=True) # Eliminate use of this


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/data',
                        help='Directory for storing data')
    FLAGS = parser.parse_args()
    tf.app.run()