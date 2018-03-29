import config as cfg
import tensorflow as tf
import data.utils
import os
import numpy as np

# Just import everything into current namespace
from tensorpack import *
from tensorpack.tfutils import summary
from tensorflow.python.platform import flags

from cnn.network import CharacterPredictor
from data.iiit5k import IIIT5K
from data.dataset import *
from data.predicted import PredictFeatures
from rnn.rnn_network import *


def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    l = tf.reduce_sum(used, 1)
    l = tf.cast(l, tf.int32)

    return l


class TrainRNNModel(ModelDesc):
    def __init__(self, max_length=36):
        self.max_length = max_length
        self.input_vector_size = 128

    def inputs(self):
        return [tf.placeholder(tf.float32, [None, None, 128], 'feat'),  # features
                tf.placeholder(tf.int64, [None, None], 'labelidx'),  # label is b x maxlen, sparse
                tf.placeholder(tf.int32, [None], 'labelvalue'),
                tf.placeholder(tf.int64, [None], 'labelshape'),
                tf.placeholder(tf.int32, [None], 'seqlen'),  # b
                ]


    def _build_graph(self, inputs):
        """This function should build the model which takes the input variables
        and define self.cost at the end"""

        # inputs contains a list of input variables defined above
        feat, labelidx, labelvalue, labelshape, seqlen = inputs

        label = tf.SparseTensor(labelidx, labelvalue, labelshape)

        #features, label = inputs
        #seq_length = length(features)

        #build the graph
        logits = build_rnn(feat, seqlen)

        # Cost function
        loss = tf.nn.ctc_loss(label, logits, seqlen, time_major=False)
        cost = tf.reduce_mean(loss, name='cost')

        # transpose to fit major_time
        logits = tf.transpose(logits, (1, 0, 2))

        isTrain = get_current_tower_context().is_training
        if isTrain:
            # beam search is too slow to run in training
            predictions = tf.to_int32(
                tf.nn.ctc_greedy_decoder(logits, seqlen)[0][0])
        else:
            predictions = tf.to_int32(
                tf.nn.ctc_beam_search_decoder(logits, seqlen)[0][0])

        err = tf.edit_distance(predictions, label, normalize=True)
        err.set_shape([None])
        err = tf.reduce_mean(err, name='error')
        summary.add_moving_summary(err, cost)


        print()

        #accuracy = tf.reduce_mean(1 - err, name='accuracy')
        #summary.add_moving_summary(err, accuracy)


        self.cost = tf.identity(cost, name='total_cost')
        #summary.add_moving_summary(cost, self.cost)

        return cost

        # """CTC"""
        # print("Logits: {}".format(logits.shape))
        # decoded, log_probs = tf.nn.ctc_beam_search_decoder(inputs=logits,
        #                                                    sequence_length=seqlen)  # log prob will not be used afterwards
        #
        # # print("Decoded length: {}".format(len(decoded)))
        #
        # # for d in decoded:
        #    # print("Indices {}, Values {}, Shape {}, ".format(d.indices, d.values, d.dense_shape))
        #
        #
        # decoded = decoded[0]
        #
        # log_probs = tf.Print(log_probs,
        #                      [decoded.indices, decoded.values],
        #                      message="Decoded: ")
        #
        # # print the predicted labels for the first data point in each step.
        # # out = tf.Print(decoded[0].indices,
        # #                   [tf.argmax(decoded, axis=1, name='prediction')],
        # #                   # [tf.nn.softmax(decoded, name='sm')],
        # #                   summarize=8,
        # #                   message="prediction: ")
        #
        # # label = tf.Print(label,
        # #                  [label],
        # #                  summarize=8,
        # #                  message="labels: ")
        #
        #
        # # a vector of length B with loss of each sample
        # # cost = tf.nn.ctc_loss(
        # #     labels=decoded,
        # #     inputs=logits,
        # #     sequence_length=seqlen
        # # )
        #
        # optimizer = self._get_optimizer()
        # cost = optimizer.minimize(cost)
        #
        # result = tf.argmax(decoded, dimension=1, output_type=tf.int32) #is this the correct output type?
        # correct = tf.cast(tf.equal(result, label), tf.float32, name='correct') #is this the correct output type?
        # accuracy = tf.reduce_mean(correct, name='accuracy')
        #
        # # This will monitor training error (in a moving_average fashion):
        # # 1. write the value to tensorboard
        # # 2. write the value to stat.json
        # # 3. print the value after each epoch
        # train_error = tf.reduce_mean(1 - correct, name='train_error')
        # summary.add_moving_summary(train_error, accuracy)
        #
        # self.cost = tf.identity(cost, name='total_cost')
        # summary.add_moving_summary(cost, self.cost)

    def _get_optimizer(self):
        #we use the adam optimizer for simplicity
        #as described in the paper we use learning rate 0.0001 and momentum term of 0.9
        return tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9)



def get_config(data, max_epoch=1500, run_inference=True):
    dataset_train, dataset_test = data

    # How many iterations you want in each epoch.
    # This is the default value, don't actually need to set it in the config
    steps_per_epoch = dataset_train.size()

    model = TrainRNNModel()
    model.steps_per_epoch = steps_per_epoch
    model.max_epoch = max_epoch

    callbacks = [
            ModelSaver(),   # save the model after every epoch
            MaxSaver('validation_accuracy')  # save the model with highest accuracy (prefix 'validation_')
        ]


    # check if inference runner is enabled
    if run_inference:
        callbacks.append(
            InferenceRunner(    # run inference(for validation) after every epoch
                dataset_test,   # the DataFlow instance used for validation
                ScalarStats(['total_cost', 'error'])))

    # get the config which contains everything necessary in a training
    return TrainConfig(
        model=model,
        dataflow=dataset_train,  # the DataFlow instance for training
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        max_epoch=max_epoch,
        # session_init=SaverRestore(os.path.join(cfg.RES_DIR, 'model/schulte/max-validation_accuracy'))
    )


def get_data(model, step_size, unique, sub_data, batch_size):
    ds_train = data.utils.load_lmdb(IIIT5K('train', unique=unique))
    ds_test = data.utils.load_lmdb(IIIT5K('test', unique=unique))

    predictor = CharacterPredictor(model)

    ds_train = PredictFeatures(ds_train, predictor, step_size=step_size)
    ds_test = PredictFeatures(ds_test, predictor, step_size=step_size)

    ds_train.name = "IIIT5K_train_features_{}".format(step_size)
    ds_test.name = "IIIT5K_test_features_{}".format(step_size)

    ds_train = data.utils.load_lmdb(ds_train)
    ds_test = data.utils.load_lmdb(ds_test)

    if unique:
        print("Use one data point per label")
        ds_train = UniqueData(ds_train)
        # for unique set, run validation on same data
        ds_test = UniqueData(data.utils.load_lmdb(IIIT5K('train', unique=unique)))

    if sub_data:
        print("Uses only {} data points".format(sub_data))
        ds_train = SubData(ds_train, sub_data)

    # check if train data should be dumped.
    if cfg.DUMP_DIR:
        print("dump data")
        data.utils.dump_data(ds_train, cfg.DUMP_DIR)

    # Batch data
    print("Use batch size {}".format(batch_size))
    #ds_train = BatchData(ds_train, batch_size)
    #ds_test = BatchData(ds_test, 2 * batch_size, remainder=True)

    ds_train = BatchedFeatures(ds_train, batch_size)
    ds_test = BatchedFeatures(ds_test, batch_size)

    return ds_train, ds_test


def train_rnn(model, step_size, unique, sub_data, batch_size):

    # automatically setup the directory for logging
    logger.set_logger_dir(cfg.TRAIN_LOG_DIR)

    data = (ds_train, ds_test) = get_data(model, step_size, unique, sub_data, batch_size)

    max_length = 0

    # for (features, label) in ds_test.get_data():
    #     if len(features) > max_length:
    #         max_length = len(features)
    #         print("Max len: {}".format(max_length))


    config = get_config(data, run_inference=True)

    # TODO change trainer
    launch_train_with_config(config, SimpleTrainer())
