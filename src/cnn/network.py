#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import cv2

# Just import everything into current namespace
from tensorpack import *
from tensorpack.tfutils import summary
from tensorflow.python.platform import flags

from cnn import maxgroup
from tensorpack.predict import OfflinePredictor, PredictConfig
from utils.window import SlidingWindow
from data.utils import int_label_to_char
from data.utils import convert_image_to_array


def build_cnn(inputs):
    # In tensorflow, inputs to convolution function are assumed to be
    # NHWC. Add a single channel here.
    inputs = tf.expand_dims(inputs, 3)
    inputs = inputs * 2 - 1  # center the pixels values at zero

    # The context manager `argscope` sets the default option for all the layers under
    # this context. Here we use convolution with shape 9x9
    with argscope(Conv2D,
                  padding='valid',
                  kernel_shape=9,
                  nl=tf.nn.relu,
                  W_init=tf.contrib.layers.variance_scaling_initializer(0.001)):
        logits = (LinearWrap(inputs)
                  .Conv2D('conv0', out_channel=96)
                  .maxgroup('max0', 2, 24, axis=3)
                  .Conv2D('conv1', out_channel=128)
                  .maxgroup('max1', 2, 16, axis=3)
                  .Conv2D('conv2', out_channel=256)
                  .maxgroup('max2', 2, 8, axis=3)
                  .Conv2D('conv3', kernel_shape=8, out_channel=512)
                  .maxgroup('max3', 4, 1, axis=3)
                  .Conv2D('conv4', kernel_shape=1, out_channel=144)
                  .maxgroup('max4', 4, 1, axis=3)
                  .FullyConnected('fc', out_dim=36, nl=tf.nn.relu,
                                  b_init=tf.contrib.layers.variance_scaling_initializer(1.0))())

    return logits


def _tower_func(inputs):
    """
    Tower func for the predictor. Builds the default cnn graph and adds a softmax to the end.
    :param input: The input tensor
    """
    # In tensorflow, inputs to convolution function are assumed to be
    # NHWC. Add a single channel here.
    inputs = tf.expand_dims(inputs, 3)
    inputs = inputs * 2 - 1  # center the pixels values at zero

    # The context manager `argscope` sets the default option for all the layers under
    # this context. Here we use convolution with shape 9x9
    with argscope(Conv2D,
                  padding='valid',
                  kernel_shape=9,
                  nl=tf.nn.relu,
                  W_init=tf.contrib.layers.variance_scaling_initializer(0.001)):
        logits = (LinearWrap(inputs)
                  .Conv2D('conv0', out_channel=96)
                  .maxgroup('max0', 2, 24, axis=3)
                  .Conv2D('conv1', out_channel=128)
                  .maxgroup('max1', 2, 16, axis=3)
                  .Conv2D('conv2', out_channel=256)
                  .maxgroup('max2', 2, 8, axis=3)
                  .Conv2D('conv3', kernel_shape=8, out_channel=512)
                  .maxgroup('max3', 4, 1, axis=3)
                  .Conv2D('conv4', kernel_shape=1, out_channel=144)
                  .maxgroup('max4', 4, 1, axis=3)
                  .FullyConnected('fc', out_dim=36, nl=tf.nn.relu,
                                  b_init=tf.contrib.layers.variance_scaling_initializer(1.0))())

    #tf.identity(logits, name='labels')
    tf.nn.softmax(logits, name='labels')

    # logits = build_cnn(input)
    # logits = tf.identity(logits, name='labels')
    # tf.nn.softmax(logits, name='labels')


def _map_prediction(p):
    """
    Helper function to convert a vector with 36 dimensions to an character label.
    This function uses the index of the component with the highest value to determine the predicted character.
    :param p: The 36D vector with the prediction values.
    :return: A character from A to Z or 0 to 9.
    """
    max_index = 0
    max_value = p[0]
    threshold = 0.5

    for i in range(1, len(p)):
        value = p[i]

        if value > max_value:
            max_value = value
            max_index = i

    return "{} ({}%)".format(int_label_to_char(max_index), int(max_value * 100))


class CharacterPredictor(OfflinePredictor):
    """
    The CharacterPredictor can be used to
    """
    def __init__(self, model):
        config = PredictConfig(
            inputs_desc=[InputDesc(tf.float32, (None, 32, 32), 'input')],
            tower_func=_tower_func,
            session_init=SaverRestore(model),
            input_names=['input'],
            # TODO cannot choose max3. Fix this
            output_names=['labels'])

        super(CharacterPredictor, self).__init__(config)

    def _predict(self, image, step_size, output):
        # Resize image if needed
        # Check if height is 32px
        (h, w) = image.shape

        if h != 32:
            # Resize to 32px
            f = 32.0 / h
            image = cv2.resize(image, (int(f * w), 32), interpolation=cv2.INTER_AREA)

        window = SlidingWindow(step_size)

        for slide in reversed(list(window.slides(image))):
            slide = slide.reshape((1,  32, 32))

            yield self(slide)[0][output]


    def predict_features(self, image, step_size=16):
        """
        Uses the CNN to generate the predicated features inside the given image.

        :param image: The image to be proccessed. The size of the image may be adjusted.
        :param step_size: The step size in pixels by which the sliding window will be moved after each step.
        :yield: The 128D feature vector predicted in the current step
        """
        self._predict(image, step_size, 0)

    def predict_characters(self, image, step_size=16, map_to_char=False):
        """
        Uses the CNN to generate the predictaed characters inside the given image.

        :param image: The image to be proccessed. The size of the image may be adjusted.
        :param step_size: The step size in pixels by which the sliding window will be moved after each step.
        :param map_to_char: If set to true, the resulting 36D label predictions vector will be casted to the
        character label with the highest prediction value.
        :yield: The 36D prediciton vector or the character label predicted in the current step.
        """
        # TODO choose output 1 if tensor is fixed
        for p in self._predict(image, step_size, 0):
            yield _map_prediction(p) if map_to_char else p