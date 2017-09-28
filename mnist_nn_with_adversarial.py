# shamelessly cribbed from https://github.com/tensorflow/cleverhans/blob/master/cleverhans_tutorials/mnist_tutorial_tf.py

"""
This tutorial shows how to generate some simple adversarial examples
and train a model using adversarial training using nothing but pure
TensorFlow.
It is very similar to mnist_tutorial_keras_tf.py, which does the same
thing but with a dependence on keras.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import time

import numpy as np
import scipy.io as sio
import pickle
import tensorflow as tf
from cleverhans.attacks import CarliniWagnerL2, FastGradientMethod
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans_tutorials.tutorial_models import MLP, ReLU, Softmax, Flatten
from util.cleverhans import AvgPool, Conv2D, MaxPool, Linear
from tensorflow.python.platform import flags

from constants import NUM_CHANNELS_CONV1, NUM_CHANNELS_CONV2, NUM_CHANNELS_FC1, WINDOW_1, WINDOW_2, POOL_1, POOL_2
from constants import CHECKPOINT_DIR


FLAGS = flags.FLAGS


def get_minified_mnist_model():
    """

    :return: A version of the mnist model parameterized by the values in constants.py
    """
    layers = [
        AvgPool(2, 2),
        Conv2D(NUM_CHANNELS_CONV1, (WINDOW_1, WINDOW_1), (1, 1), "SAME", "conv1", ),
        MaxPool(POOL_1, POOL_1),
        ReLU(),
        Conv2D(NUM_CHANNELS_CONV2, (WINDOW_2, WINDOW_2), (1, 1), "SAME", "conv2", ),
        MaxPool(POOL_2, POOL_2),
        ReLU(),
        Flatten(),
        Linear(NUM_CHANNELS_FC1, "fc1"),
        ReLU(),
        Linear(10, "fc2"),
        Softmax()
    ]

    model = MLP(layers, (None, 28, 28, 1))
    return model

def mnist(train_start=0, train_end=60000, test_start=0,
          test_end=10000, nb_epochs=6, batch_size=128,
          learning_rate=0.001,
          backprop_through_attack=False,
          adversarial_training=False,
          ):
    """
    MNIST cleverhans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param backprop_through_attack: If True, backprop through adversarial
                                    example construction process during
                                    adversarial training.
    :param adversarial_training: whether to conduct adversarial training.
    :return: an AccuracyReport object
    """
    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    runID = timestamp
    run_checkpoint_dir = os.path.join(CHECKPOINT_DIR, runID)
    if not os.path.exists(run_checkpoint_dir):
        os.makedirs(run_checkpoint_dir)

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Create TF session
    sess = tf.Session()

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Use label smoothing
    assert Y_train.shape[1] == 10
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': run_checkpoint_dir,
        'filename': 'cp_cleverhans'
    }

    fgsm_params = {
        'eps': 5,
        'ord': 2,
        'clip_min': 0.,
        'clip_max': 1.,
    }

    rng = np.random.RandomState([2017, 8, 30])

    # Redefine TF model graph
    model = get_minified_mnist_model()
    predictions = model(x)
    fgsm = FastGradientMethod(model, sess=sess)
    adversarial_predictions = fgsm.generate(x, **fgsm_params)
    if not backprop_through_attack:
        # For the fgsm attack used in this tutorial, the attack has zero
        # gradient so enabling this flag does not change the gradient.
        # For some other attacks, enabling this flag increases the cost of
        # training, but gives the defender the ability to anticipate how
        # the atacker will change their strategy in response to updates to
        # the defender's parameters.
        adversarial_predictions = tf.stop_gradient(adversarial_predictions)
    preds_2_adv = model(adversarial_predictions)

    def evaluate_2():
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                              args=eval_params)
        print('Test accuracy on legitimate examples: %0.4f' % accuracy)
        report.clean_eval = accuracy

        # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x, y, preds_2_adv, X_test,
                              Y_test, args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f' % accuracy)
        report.adv_eval = accuracy

    # Perform and evaluate adversarial training
    model_train(sess, x, y, predictions, X_train, Y_train,
                predictions_adv=preds_2_adv if adversarial_training else None, evaluate=evaluate_2,
                args=train_params, rng=rng, save=True)

    ### Spend time to find good adversarial examples using the CarliniWagner method.
    cwl_params = {
        'confidence': 5,
        'batch_size': 10,
        'learning_rate': 0.005,
        'binary_search_steps': 10,  # default 10
        'max_iterations': 20000,
        'abort_early': True,
        'initial_const': 0.01,
        'clip_min': 0.,
        'clip_max': 1.,
    }

    cwl = CarliniWagnerL2(model, sess=sess)
    x_adv_cwl = cwl.generate(x, **cwl_params)
    d = sess.run({'x': x, 'adv_x': x_adv_cwl}, feed_dict={x: X_test[:10], y: Y_test[:10]})
    sio.savemat(
        os.path.join(
            run_checkpoint_dir,
            os.path.join(run_checkpoint_dir, 'adversarial-examples.mat')
                .format('adv' if adversarial_training else 'clean')),
        d
    )
    pickle.dump(report, open(os.path.join(run_checkpoint_dir, 'report.pickle'), 'wb'))
    return report


def main(argv=None):
    mnist(
        nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.learning_rate,
        backprop_through_attack=FLAGS.backprop_through_attack,
        adversarial_training=FLAGS.adversarial_training,
    )


if __name__ == '__main__':
    flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_bool('backprop_through_attack', False,
                      ('If True, backprop through adversarial example '
                       'construction process during adversarial training'))
    flags.DEFINE_bool('adversarial_training', True,
                      ('If True, train with adversarial example generated via FGSM.'))

    tf.app.run()