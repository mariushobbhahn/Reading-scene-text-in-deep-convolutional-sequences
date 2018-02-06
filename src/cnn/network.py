#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

# Just import everything into current namespace
from tensorpack import *
from tensorpack.tfutils import summary
from tensorflow.python.platform import flags

from cnn import maxgroup
from utils.window import SlidingWindow
from data.utils import int_label_to_char


def _map_prediction(p):
    """
    Helper function to convert a vector with 36 dimensions to an character label.
    This function uses the index of the component with the highest value to determine the predicted character.
    :param p: The 36D vector with the prediction values.
    :return: A character from A to Z or 0 to 9.
    """
    max_index = 0
    max_value = p[0]

    for i in range(1, len(p)):
        value = p[i]

        if value > max_value:
            max_value = value
            max_index = i

    return int_label_to_char(max_index)


class CNN:
    """
    The convolutional neuronal network which is used to detect characters in images.
    """

    def __init__(self, inputs, model=None):
        """
        Create a new graph for the CNN with the given inputs.

        :param inputs: The input Tensors for this Model. Must be in shape NHWC.
        :param model: The location of a model to be loaded.
        If None is given, the graph will use default weights. Default is None.
        """
        self.model = model
        self._is_model_loaded = False
        self.in_image = inputs

        # The context manager `argscope` sets the default option for all the layers under
        # this context. Here we use convolution with shape 9x9
        with argscope(Conv2D,
                      padding='valid',
                      kernel_shape=9,
                      nl=tf.identity,
                      W_init=tf.contrib.layers.variance_scaling_initializer(0.001)):

            # output for RNN
            self.out_features = (LinearWrap(inputs).
                Conv2D('conv0', out_channel=96).
                maxgroup('max0', 2, 24, axis=3).
                Conv2D('conv1', out_channel=128).
                maxgroup('max1', 2, 16, axis=3).
                Conv2D('conv2', out_channel=256).
                maxgroup('max2', 2, 8, axis=3).
                Conv2D('conv3', kernel_shape=8, out_channel=512).
                maxgroup('max3', 4, 1, axis=3)())

            # output for training
            self.out_labels = (LinearWrap(self.out_features).
                Conv2D('conv4', kernel_shape=1, out_channel=144).
                maxgroup('max4', 4, 1, axis=3).
                FullyConnected('fc', out_dim=36, nl=tf.nn.relu, b_init=tf.contrib.layers.variance_scaling_initializer(1.0))())
                #pruneaxis('prune', 36)())

    def _load_model(self, session):
        if self.model and (not self._is_model_loaded):
            SaverRestore(self.model).init(session)
            self._is_model_loaded = True

    def _process_image(self, image, step_size, use_features):
        window = SlidingWindow(step_size=step_size)
        graph = self.out_features if use_features else self.out_labels

        # TODO is this really the graph?
        with tf.Session(graph=graph) as sess:
            self._load_model(sess)
            i = 0

            for slide in window.slides(image):
                print("process slide {}".format(i))
                i += 1

                #TODO is this the way to set the input?
                feed = {self.in_image: slide}

                # Calculate next feature
                yield sess.run(graph)

    def get_features_from_image(self, image, step_size=16):
        """
        Generates 128D feature vectors for every window.

        :param image: The image to be processed. Must have height = 32px and should be grayscale.
        :param step_size: The distance in pixels between every step.
        :yield: The predicted features of the current step.
        """
        self._process_image(image, step_size, True)

    def get_characters_from_image(self, image, step_size=16):
        """
        Processes the images and yields the recognized character for the current step.

        :param image: The image to be processed. Must hve height = 32px and should be grayscale.
        :param step_size: The distance in pixels between every step.
        :yield: The predicted character of the current step.
        """
        for p in self._process_image(image, step_size, False):
            yield _map_prediction(p)

