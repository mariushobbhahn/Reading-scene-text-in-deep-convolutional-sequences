#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

# Just import everything into current namespace
from tensorpack import *
from tensorpack.tfutils import summary
from tensorflow.python.platform import flags

from cnn import maxgroup
from utils.window import SlidingWindow


class CNN:
    def __init__(self, inputs, model=None):
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

            self.out_features = (LinearWrap(inputs).
                Conv2D('conv0', out_channel=96).
                maxgroup('max0', 2, 24, axis=3).
                Conv2D('conv1', out_channel=128).
                maxgroup('max1', 2, 16, axis=3).
                Conv2D('conv2', out_channel=256).
                maxgroup('max2', 2, 8, axis=3).
                Conv2D('conv3', kernel_shape=8, out_channel=512).
                maxgroup('max3', 4, 1, axis=3)())

            self.out_labels = (LinearWrap(self.out_features).
                Conv2D('conv4', kernel_shape=1, out_channel=144).
                maxgroup('max4', 4, 1, axis=3).
                # FullyConnected('fc', out_dim=36, nl=tf.nn.relu, b_init=tf.contrib.layers.variance_scaling_initializer(1.0))())
                pruneaxis('prune', 36)())

    def _load_model(self, session):
        if self.model and (not self._is_model_loaded):
            SaverRestore(self.model).init(session)
            self._is_model_loaded = True

    def process(self, image, window=None):
        if window is None:
            window = SlidingWindow(step_size=16)

        features = list()

        # TODO is this really the graph?
        with tf.Session(graph=self.out_features) as sess:
            self._load_model(sess)
            i = 0

            for slide in window.slides(image):
                print("process slide {}".format(i))
                i += 1

                #TODO is this teh way to set the input?
                feed = {self.in_image: slide}

                # Calculate next feature
                features.append(sess.run(self.out_features))

        return features
