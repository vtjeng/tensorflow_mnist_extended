"""Shamelessly cribbed from the online MIST tutorial.
See more details at
https://www.tensorflow.org/tutorials/mnist/pros/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import scipy.io as sio
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

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