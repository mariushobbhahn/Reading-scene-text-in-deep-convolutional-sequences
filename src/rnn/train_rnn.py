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

class TrainRNNModel(ModelDesc):

    def __init__(self, image_size=32):
        self.image_size = int(image_size)

    def _get_inputs(self):
        """
        Define all the inputs (with type, shape, name) that
        the graph will need.
        """
        # TODO Input is a sequence of 128D vectors and a n-length label
        return [InputDesc(tf.float32, (None, 128), 'input'),
                InputDesc(tf.int32, (None,), 'label')]

    def _build_graph(self, inputs):
        """This function should build the model which takes the input variables
        and define self.cost at the end"""

        # inputs contains a list of input variables defined above
        features, label = inputs
        # constant for the length of this particular sequence
        sequence_length = len(image)

        logits = build_RNN(image)

        """CTC"""
        decoded, log_probs = tf.nn.ctc_beam_search_decoder(inputs=logits,
                                                           sequence_length=seq_length)  # log prob will not be used afterwards

        # print the predicted labels for the first data point in each step.
        decoded = tf.Print(decoded,
                          [tf.argmax(decoded, dimension=1, name='prediction')],
                          # [tf.nn.softmax(decoded, name='sm')],
                          summarize=8,
                          message="prediction: ")

        label = tf.Print(label,
                         [label],
                         summarize=8,
                         message="labels: ")


        # a vector of length B with loss of each sample
        cost = tf.nn.ctc_loss(
            labels=label,
            inputs=decoded,
            sequence_length=sequence_length
        )

        optimizer = self._get_optimizer()
        cost = optimizer.minimize(cost)

        result = tf.argmax(decoded, dimension=1, output_type=tf.int32) #is this the correct output type?
        correct = tf.cast(tf.equal(result, label), tf.float32, name='correct') #is this the correct output type?
        accuracy = tf.reduce_mean(correct, name='accuracy')

        # This will monitor training error (in a moving_average fashion):
        # 1. write the value to tensorboard
        # 2. write the value to stat.json
        # 3. print the value after each epoch
        train_error = tf.reduce_mean(1 - correct, name='train_error')
        summary.add_moving_summary(train_error, accuracy)

        self.cost = tf.identity(cost, name='total_cost')
        summary.add_moving_summary(cost, self.cost)

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
                ScalarStats(['cross_entropy_loss', 'accuracy'])))

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

    predictor = CharacterPredictor(model)

    ds_train = PredictFeatures(ds_train, predictor, step_size=step_size)
    ds_test = PredictFeatures(ds_test, predictor, step_size=step_size)

    return ds_train, ds_test


def train_rnn(model, step_size=16, unique=False, sub_data=None, batch_size=None):
    data = (ds_train, ds_test) = get_data(model, step_size, unique, sub_data, batch_size)

    max_length = 0

    # for (features, label) in ds_test.get_data():
    #     if len(features) > max_length:
    #         max_length = len(features)
    #         print("Max len: {}".format(max_length))


    config = get_config(data, run_inference=True)

    # TODO change trainer
    launch_train_with_config(config, SimpleTrainer())