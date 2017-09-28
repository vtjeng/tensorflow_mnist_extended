"""
A pure TensorFlow implementation of a neural network. This can be
used as a drop-in replacement for a Keras model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from cleverhans_tutorials.tutorial_models import Layer

class Linear(Layer):

    def __init__(self, num_hid, name):
        self.num_hid = num_hid
        self.name = name

    def set_input_shape(self, input_shape):
        with tf.name_scope(self.name):
            batch_size, dim = input_shape
            self.input_shape = [batch_size, dim]
            self.output_shape = [batch_size, self.num_hid]
            init = tf.random_normal([dim, self.num_hid], dtype=tf.float32)
            init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=0,
                                                       keep_dims=True))
            print("linear layer", input_shape)
            self.W = tf.Variable(init, name='weight')
            self.b = tf.Variable(np.zeros((self.num_hid,)).astype('float32'), name='bias')

    def fprop(self, x):
        return tf.matmul(x, self.W) + self.b


class Conv2D(Layer):

    def __init__(self, output_channels, kernel_shape, strides, padding, name):
        self.__dict__.update(locals())
        del self.self

    def set_input_shape(self, input_shape):
        with tf.name_scope(self.name):
            batch_size, rows, cols, input_channels = input_shape
            kernel_shape = tuple(self.kernel_shape) + (input_channels,
                                                       self.output_channels)
            assert len(kernel_shape) == 4
            assert all(isinstance(e, int) for e in kernel_shape), kernel_shape
            init = tf.random_normal(kernel_shape, dtype=tf.float32)
            init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init),
                                                       axis=(0, 1, 2)))
            self.kernels = tf.Variable(init, name='weight')
            self.b = tf.Variable(
                np.zeros((self.output_channels,)).astype('float32'), name='bias')
            input_shape = list(input_shape)
            input_shape[0] = 1
            dummy_batch = tf.zeros(input_shape)
            dummy_output = self.fprop(dummy_batch)
            output_shape = [int(e) for e in dummy_output.get_shape()]
            output_shape[0] = 1
            self.output_shape = tuple(output_shape)
            print("conv2d layer", input_shape)

    def fprop(self, x):
        return tf.nn.conv2d(x, self.kernels, (1,) + tuple(self.strides) + (1,),
                            self.padding) + self.b

# TODO: Inherit from common Pool class
class AvgPool(Layer):

    def __init__(self, pool_ksize, pool_stride):
        self.ksize = [1, pool_ksize, pool_ksize, 1]
        self.strides = [1, pool_stride, pool_stride, 1]

    def set_input_shape(self, shape):
        self.input_shape = shape
        assert len(shape) == len(self.strides)
        self.output_shape = list()
        for i in range(len(shape)):
            if shape[i] is None:
                self.output_shape.append(None)
            else:
                self.output_shape.append(int(np.ceil(shape[i] / self.strides[i])))
        print("avgpool layer", shape)

    def get_output_shape(self):
        return self.output_shape

    def fprop(self, x):
        return tf.nn.avg_pool(x, ksize = self.ksize, strides = self.strides, padding="SAME")


class MaxPool(Layer):

    def __init__(self, pool_ksize, pool_stride):
        self.ksize = [1, pool_ksize, pool_ksize, 1]
        self.strides = [1, pool_stride, pool_stride, 1]

    def set_input_shape(self, shape):
        self.input_shape = shape
        assert len(shape) == len(self.strides)
        self.output_shape = list()
        for i in range(len(shape)):
            if shape[i] is None:
                self.output_shape.append(None)
            else:
                self.output_shape.append(int(np.ceil(shape[i] / self.strides[i])))
        print("maxpool layer", shape)

    def get_output_shape(self):
        return self.output_shape

    def fprop(self, x):
        return tf.nn.max_pool(x, ksize = self.ksize, strides = self.strides, padding="SAME")