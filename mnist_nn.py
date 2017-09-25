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

from constants import NUM_CHANNELS_CONV1, NUM_CHANNELS_CONV2, NUM_CHANNELS_FC1
from constants import BATCH_SIZE, NUM_EPOCHS, TB_LOGS_DIR, CHECKPOINT_DIR, EVAL_FREQUENCY, CHECKPOINT_FREQUENCY
from constants import CHECKPOINT_HOURS, CHECKPOINT_MAX_KEEP
from nn_util import ff_layer, conv2d_layer, conv_to_ff_layer
from visualize import view_images, view_incorrect, one_hot_to_index


# TODO: Save checkpoint file at end of run. Checkpoint file
# TODO: Enable recovery from specified checkpoint file.
def main(_):
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    runID = timestamp

    ### Building the Graph
    # None indicates that the first dimension, corresponding to the batch size, can be of any size.
    x = tf.placeholder(tf.float32, [None, 784], name='original_image')

    # With tf.reshape, size of dimension with special value -1 computed so total size remains constant.
    x_image = tf.reshape(x, [-1,28,28,1], name='flattened_image')
    (h_pool1, w_conv1, b_conv1) = conv2d_layer(x_image, depth=NUM_CHANNELS_CONV1, window=5, pool=(2, 2), name='conv1')
    (h_pool2, w_conv2, b_conv2) = conv2d_layer(h_pool1, depth=NUM_CHANNELS_CONV2, window=5, pool=(2, 2), name='conv2')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h_pool2_flat = conv_to_ff_layer(h_pool2)
    (h_fc1_drop, w_fc1, b_fc1) = ff_layer(h_pool2_flat, NUM_CHANNELS_FC1, name='ff1', dropout=keep_prob)
    (y, w_y, b_y) = ff_layer(h_fc1_drop, depth=10, name='ff2', activation_fn=None)

    y_ = tf.placeholder(tf.float32, [None, 10])
    with tf.name_scope('performance'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_), name='cross_entropy')
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # Saving information to Tensorboard.
    tf.summary.scalar('Cross Entropy', cross_entropy)
    tf.summary.scalar('Accuracy', accuracy)
    summary = tf.summary.merge_all()

    run_checkpoint_dir = os.path.join(CHECKPOINT_DIR, runID)
    if not os.path.exists(run_checkpoint_dir):
        os.makedirs(run_checkpoint_dir)
    saver = tf.train.Saver(
        max_to_keep=CHECKPOINT_MAX_KEEP,
        keep_checkpoint_every_n_hours=CHECKPOINT_HOURS,
    )

    run_tb_logs_dir = os.path.join(TB_LOGS_DIR, runID)
    tensorboard_train_prefix = os.path.join(run_tb_logs_dir, 'training') # TODO: improve naming

    with tf.Session() as sess:
        # Initialize summary write for TensorBoard
        os.makedirs(tensorboard_train_prefix)
        train_writer = tf.summary.FileWriter(tensorboard_train_prefix, graph=sess.graph)

        sess.run(tf.global_variables_initializer())
        num_iterations = NUM_EPOCHS * mnist.train.num_examples // BATCH_SIZE
        for step in range(num_iterations):
            batch = mnist.train.next_batch(BATCH_SIZE)
            if step % EVAL_FREQUENCY == 0:
                train_feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0}
                train_accuracy = accuracy.eval(feed_dict=train_feed_dict)
                train_summary = summary.eval(feed_dict=train_feed_dict)
                print("step %d, training accuracy %g"%(step, train_accuracy))
                train_writer.add_summary(train_summary, step)
                train_writer.flush()
            if step % CHECKPOINT_FREQUENCY == 0:
                checkpoint_file = os.path.join(run_checkpoint_dir, 'cp')
                print("\tSaving state in %s" % (checkpoint_file))
                # Step is automatically added by passing in the global_step option.
                saver.save(sess, checkpoint_file, global_step=step, write_meta_graph=True)
                print("\tSave success.\n")
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print("test accuracy %g" % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

        y_final = sess.run(y, feed_dict={x: mnist.test.images,
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