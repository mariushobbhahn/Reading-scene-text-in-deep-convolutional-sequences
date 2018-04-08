#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: cifar10-resnet.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import argparse
import os
import data.utils
import config as cfg


from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.dataflow import dataset

import tensorflow as tf

from data.iiit5k import IIIT5KChar
from data.dataset import *
from cnn.network import build_cnn
from cnn.maxgroup import *

"""
CIFAR10 ResNet example. See:
Deep Residual Learning for Image Recognition, arxiv:1512.03385
This implementation uses the variants proposed in:
Identity Mappings in Deep Residual Networks, arxiv:1603.05027
I can reproduce the results on 2 TitanX for
n=5, about 7.1% val error after 67k steps (20.4 step/s)
n=18, about 5.95% val error after 80k steps (5.6 step/s, not converged)
n=30: a 182-layer network, about 5.6% val error after 51k steps (3.4 step/s)
This model uses the whole training set instead of a train-val split.
To train:
    ./cifar10-resnet.py --gpu 0,1
"""

BATCH_SIZE = 128
NUM_UNITS = 18


class C10_Default_Model(ModelDesc):

    def __init__(self, n, steps_per_epoch=1):
        super(C10_Default_Model, self).__init__()
        self.n = n
        self.steps_per_epoch = steps_per_epoch

    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, 32,32), 'input'),
                InputDesc(tf.int32, (None,), 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = image / 255.0 # convert uint8 -> float
        image = image * 2
        image = image - 1
        # In tensorflow, inputs to convolution function are assumed to be
        # NHWC. Add a single channel here.
        image = tf.expand_dims(image, 3)
        assert tf.test.is_gpu_available()
        image = tf.transpose(image, [0, 3, 1, 2])

        def residual(name, l, increase_dim=False, first=False):
            shape = l.get_shape().as_list()
            in_channel = shape[1]

            if increase_dim:
                out_channel = in_channel * 2
                stride1 = 2
            else:
                out_channel = in_channel
                stride1 = 1

            with tf.variable_scope(name):
                b1 = l if first else BNReLU(l)
                c1 = Conv2D('conv1', b1, out_channel, strides=stride1, activation=BNReLU)
                c2 = Conv2D('conv2', c1, out_channel)
                if increase_dim:
                    l = AvgPooling('pool', l, 2)
                    l = tf.pad(l, [[0, 0], [in_channel // 2, in_channel // 2], [0, 0], [0, 0]])

                l = c2 + l
                return l

        with argscope([Conv2D, AvgPooling, BatchNorm, GlobalAvgPooling], data_format='channels_first'), \
                argscope(Conv2D, use_bias=False, kernel_size=3,
                         kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
            l = Conv2D('conv0', image, 16, activation=BNReLU)
            l = residual('res1.0', l, first=True)
            for k in range(1, self.n):
                l = residual('res1.{}'.format(k), l)
            # 32,c=16

            l = residual('res2.0', l, increase_dim=True)
            for k in range(1, self.n):
                l = residual('res2.{}'.format(k), l)
            # 16,c=32

            l = residual('res3.0', l, increase_dim=True)
            for k in range(1, self.n):
                l = residual('res3.' + str(k), l)
            l = BNReLU('bnlast', l)
            # 8,c=64
            l = GlobalAvgPooling('gap', l)

        logits = tf.contrib.layers.fully_connected(l, 36, activation_fn=tf.identity)
#        label = tf.Print(label, [label])

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        
        result = tf.argmax(logits, dimension=1, output_type=tf.int32)
        correct = tf.cast(tf.equal(result, label), tf.float32, name='correct')
        accuracy = tf.reduce_mean(correct, name='accuracy')
        train_error = tf.reduce_mean(1 - correct, name='train_error')
        summary.add_moving_summary(train_error, accuracy)

        # weight decay on all W of fc layers
#        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
#                                          480000, 0.2, True)
#        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost)

#        add_param_summary(('.*/W', ['histogram']))   # monitor W
        return tf.identity(cost, name='cost')

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt



def get_data(unique=False, sub_data=None, batch_size=128):
    ds_train = data.utils.load_lmdb(IIIT5KChar('train', unique=unique))
    ds_test = data.utils.load_lmdb(IIIT5KChar('test', unique=unique))

    if unique:
        print("Use one data point per label")
        ds_train = UniqueData(ds_train)
        # for unique set, run validation on same data
        ds_test = UniqueData(data.utils.load_lmdb(IIIT5KChar('train', unique=unique)))

    if sub_data:
        print("Uses only {} data points".format(sub_data))
        ds_train = SubData(ds_train, sub_data)

    # check if train data should be dumped.
    if cfg.DUMP_DIR:
        print("dump data")
        data.utils.dump_data(ds_train, cfg.DUMP_DIR)

    # Batch data
    print("Use batch size {}".format(batch_size))
    ds_train = BatchData(ds_train, batch_size)
    ds_test = BatchData(ds_test, 2 * batch_size, remainder=True)

    return ds_train, ds_test


def train_resnet(unique=False, sub_data=None, batch_size=None):
    print("train resnet")
    logger.set_logger_dir(cfg.TRAIN_LOG_DIR)

    dataset_train, dataset_test = get_data(unique=unique,
                    sub_data=sub_data,
                    batch_size=batch_size)

    c = TrainConfig(
        model=C10_Default_Model(n=NUM_UNITS),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            ScalarStats(['cross_entropy_loss', 'accuracy'])),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.1), (20, 0.01), (30, 0.001), (75, 0.0002)])
        ],
        max_epoch=100,
#        session_init=SaverRestore(args.load) if args.load else None
    )
    launch_train_with_config(c, SimpleTrainer())