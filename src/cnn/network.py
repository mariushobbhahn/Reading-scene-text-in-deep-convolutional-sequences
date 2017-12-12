#!/usr/bin/env python
# -*- coding: utf-8 -*-
import config
import os
import argparse
import tensorflow as tf

# Just import everything into current namespace
from tensorpack import *
from tensorpack.tfutils import summary
from data import dataset

IMAGE_SIZE = 32


class Model(ModelDesc):
    def _get_inputs(self):
        """
        Define all the inputs (with type, shape, name) that
        the graph will need.
        """
        return [InputDesc(tf.float32, (None, IMAGE_SIZE, IMAGE_SIZE), 'input'),
                InputDesc(tf.int32, (None,), 'label')]

    def _build_graph(self, inputs):
        """This function should build the model which takes the input variables
        and define self.cost at the end"""

        # inputs contains a list of input variables defined above
        image, label = inputs

        # In tensorflow, inputs to convolution function are assumed to be
        # NHWC. Add a single channel here.
        image = tf.expand_dims(image, 3)
        image = image * 2 - 1   # center the pixels values at zero


        # The context manager `argscope` sets the default option for all the layers under
        # this context. Here we use convolution with shape 9x9
        with argscope(Conv2D, padding='valid', kernel_shape=9, nl=tf.nn.relu):
            logits = (LinearWrap(image).
                        Conv2D('conv0', out_channel=96).
                        #Maxout('max0', num_unit=2).
                        Conv2D('conv1', out_channel=128).
                        #Maxout('max1', num_unit=2).
                        Conv2D('conv2', out_channel=256).
                        #Maxout('max2', num_unit=2).
                        Conv2D('conv3', kernel_shape=8, out_channel=512).
                        #Maxout('max3', num_unit=4).
                        Conv2D('conv4', kernel_shape=1, out_channel=144)())
                        #FullyConnected('fc', out_dim=10, nl=tf.identity)())
                        #Maxout('max4', num_unit=4))



        #softmax = logits.Softmax('prob')
        tf.nn.softmax(logits, name='prob')   # a Bx10 with probabilities

        # a vector of length B with loss of each sample
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')  # the average cross-entropy loss

        correct = tf.cast(tf.nn.in_top_k(logits, label, 1), tf.float32, name='correct')
        accuracy = tf.reduce_mean(correct, name='accuracy')

        # This will monitor training error (in a moving_average fashion):
        # 1. write the value to tensosrboard
        # 2. write the value to stat.json
        # 3. print the value after each epoch
        train_error = tf.reduce_mean(1 - correct, name='train_error')
        summary.add_moving_summary(train_error, accuracy)

        # Use a regex to find parameters to apply weight decay.
        # Here we apply a weight decay on all W (weight matrix) of all fc layers
        wd_cost = tf.multiply(1e-5,
                              regularize_cost('fc.*/W', tf.nn.l2_loss),
                              name='regularize_loss')
        self.cost = tf.add_n([wd_cost, cost], name='total_cost')
        summary.add_moving_summary(cost, wd_cost, self.cost)

        # monitor histogram of all weight (of conv and fc layers) in tensorboard
        summary.add_param_summary(('.*/W', ['histogram', 'rms']))

    def _get_optimizer(self):
        lr = tf.train.exponential_decay(
            learning_rate=1e-3,
            global_step=get_global_step_var(),
            decay_steps=468 * 10,
            decay_rate=0.3, staircase=True, name='learning_rate')
        # This will also put the summary in tensorboard, stat.json and print in terminal
        # but this time without moving average
        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr)


def get_data():
    train = BatchData(dataset.IIIT5K('train', char_data=True), 128)
    test = BatchData(dataset.IIIT5K('test', char_data=True), 256, remainder=True)
    return train, test


def get_config():
    dataset_train, dataset_test = get_data()
    # How many iterations you want in each epoch.
    # This is the default value, don't actually need to set it in the config
    steps_per_epoch = dataset_train.size()

    # get the config which contains everything necessary in a training
    return TrainConfig(
        model=Model(),
        dataflow=dataset_train,  # the DataFlow instance for training
        callbacks=[
            ModelSaver(),   # save the model after every epoch
            MaxSaver('validation_accuracy'),  # save the model with highest accuracy (prefix 'validation_')
            InferenceRunner(    # run inference(for validation) after every epoch
                dataset_test,   # the DataFlow instance used for validation
                ScalarStats(['cross_entropy_loss', 'accuracy'])),
        ],
        steps_per_epoch=steps_per_epoch,
        max_epoch=100,
    )


if __name__ == '__main__':
    print("start network: server={}".format(config.IS_SERVER))

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # automatically setup the directory for logging
    logger.auto_set_dir()

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)

    # SimpleTrainer is slow, this is just a demo.
    # You can use QueueInputTrainer instead
    launch_train_with_config(config, SimpleTrainer())
