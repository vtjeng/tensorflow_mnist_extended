"""Shamelessly cribbed from the online MIST tutorial.
See more details at
https://www.tensorflow.org/tutorials/mnist/pros/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from constants import NUM_CHANNELS_CONV1, NUM_CHANNELS_CONV2, NUM_CHANNELS_FC1, WINDOW_1, WINDOW_2, POOL_1, POOL_2
from constants import BATCH_SIZE, NUM_EPOCHS, TB_LOGS_DIR, CHECKPOINT_DIR, EVAL_FREQUENCY, CHECKPOINT_FREQUENCY
from constants import CHECKPOINT_HOURS, CHECKPOINT_MAX_KEEP
from nn_util import fc_layer, conv2d_layer, conv_to_ff_layer
from visualize import view_images, view_incorrect, one_hot_to_index
import scipy.io as sio


"""
Exports the MNIST test set, with the images resized from 784 to 28*28 then downsampled to 14*14
"""
def main(_):
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    ### Building the Graph
    # None indicates that the first dimension, corresponding to the batch size, can be of any size.
    x = tf.placeholder(tf.float32, [None, 784], name='original_image')

    # With tf.reshape, size of dimension with special value -1 computed so total size remains constant.
    x_image = tf.reshape(x, [-1,28,28,1], name='flattened_image')
    x_resize = tf.nn.avg_pool(x_image, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME', name = 'input-resize')

    with tf.Session() as sess:
        # Initialize summary write for TensorBoard
        sess.run(tf.global_variables_initializer())
        x_resize_output = sess.run(x_resize, feed_dict={x: mnist.test.images})
        sio.savemat('mnist_test_data_resized.mat', {'x_resize': x_resize_output, 'y_': mnist.test.labels})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/data',
                        help='Directory for storing data')
    FLAGS = parser.parse_args()
    tf.app.run()