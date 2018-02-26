#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import cv2
import numpy as np

# Just import everything into current namespace
from tensorpack import *
from tensorpack.tfutils import summary
from tensorpack.models.nonlin import Maxout
from tensorflow.python.platform import flags

from cnn.maxgroup import MaxGroup, maxgroup
from tensorpack.predict import OfflinePredictor, PredictConfig
from utils.window import SlidingWindow
from data.utils import int_label_to_char

# from tensorflow.python.layers import maxout
from data.utils import convert_image_to_array

WEIGHT_INIT_VARIANCE = 1
BIAS_INIT_VARIANCE = 1.0

def w_init_variance(k_size=9):
    return WEIGHT_INIT_VARIANCE / (k_size * k_size)

DEFAULT_NL = tf.nn.relu
FC_NL = tf.identity

def build_cnn(inputs):
    # In tensorflow, inputs to convolution function are assumed to be
    # NHWC. Add a single channel here.
    inputs = tf.expand_dims(inputs, 3)
    # inputs = tf.Print(inputs, [inputs], summarize=32)

    # activation = tf.identity
    #
    # logits = tf.layers.conv2d(inputs=inputs,
    #                           filters=96,
    #                           kernel_size=[9, 9],
    #                           padding='valid',
    #                           activation=activation,
    #                           name='conv0')
    #
    # logits = Maxout(num_units=48, name='max0').apply(logits)
    #
    # logits = tf.layers.conv2d(inputs=logits,
    #                           filters=128,
    #                           kernel_size=[9, 9],
    #                           padding='valid',
    #                           activation=activation,
    #                           name='conv1')
    #
    # logits = tf.contrib.layers.maxout(inputs=logits,
    #                           num_units=64,
    #                           axis=3,
    #                           name='max1')
    #
    #
    # logits = tf.layers.conv2d(inputs=logits,
    #                           filters=256,
    #                           kernel_size=[9, 9],
    #                           padding='valid',
    #                           activation=activation,
    #                           name='conv2')
    #
    # logits = tf.contrib.layers.maxout(inputs=logits,
    #                           num_units=128,
    #                           axis=3,
    #                           name='max1')
    #
    # logits = tf.layers.conv2d(inputs=logits,
    #                           filters=512,
    #                           kernel_size=[8, 8],
    #                           padding='valid',
    #                           activation=activation,
    #                           name='conv3')
    #
    # logits = tf.contrib.layers.maxout(inputs=logits,
    #                           num_units=128,
    #                           axis=3,
    #                           name='max1')
    #
    # logits = tf.layers.conv2d(inputs=logits,
    #                           filters=144,
    #                           kernel_size=[1, 1],
    #                           padding='valid',
    #                           activation=activation,
    #                           name='conv4')
    #
    # logits = tf.contrib.layers.maxout(inputs=logits,
    #                           num_units=36,
    #                           axis=3,
    #                           name='max1')
    #
    # logits = tf.reshape(tensor=logits,
    #                     shape=[1, 36],
    #                     name='prune')

    # The context manager `argscope` sets the default option for all the layers under
    # this context. Here we use convolution with shape 9x9
    with argscope(Conv2D,
                  padding='valid',
                  kernel_shape=9,
                  nl=DEFAULT_NL,
                  W_init=tf.contrib.layers.variance_scaling_initializer(WEIGHT_INIT_VARIANCE)):
        logits = (LinearWrap(inputs)
                  .Conv2D('conv0', out_channel=96)
                  .Maxout('max0', num_unit=2)
                  .Conv2D('conv1', out_channel=128)
                  .Maxout('max1', num_unit=2)
                  .Conv2D('conv2', out_channel=256)
                  .Maxout('max2', num_unit=2)
                  # .Conv2D('conv3', kernel_shape=8, out_channel=512)
                  # .Maxout('max3', num_unit=4)
                  .Conv2D('conv4', kernel_shape=1, out_channel=144)
                  .Maxout('max4', num_unit=4)
                  # .pruneaxis('prune')
                  .FullyConnected('fc', out_dim=36, nl=FC_NL)
                  # .Maxout('maxfc', num_unit=4)
                                  # ,b_init=tf.contrib.layers.variance_scaling_initializer(BIAS_INIT_VARIANCE))
                  ())
        #logits = tf.Print(logits, [logits], summarize=36)

        return logits


def _tower_func(inputs):
    """
    Tower func for the predictor. Builds the default cnn graph and adds a softmax to the end.
    :param input: The input tensor
    """

    logits = build_cnn(inputs)
    # logits = tf.Print(logits, [logits], summarize=36)
    logits = tf.identity(logits, name='labels')
    # tf.nn.softmax(logits, name='labels')


def _map_prediction(p):
    """
    Helper function to convert a vector with 36 dimensions to an character label.
    This function uses the index of the component with the highest value to determine the predicted character.
    :param p: The 36D vector with the prediction values.
    :return: A character from A to Z or 0 to 9.
    """
    max_index = np.argmax(p)

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
            # output_names=['max3/output', 'labels'])

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

        for slide in window.slides(image):
            slide = slide.reshape((1,  32, 32)).astype('float32')
            #print(self(slide))
            yield self(slide)[output][0]

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
