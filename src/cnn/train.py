import config as cfg
import tensorflow as tf
import data.utils

# Just import everything into current namespace
from tensorpack import *
from tensorpack.tfutils import summary
from tensorflow.python.platform import flags

from data.iiit5k import IIIT5KChar
from data.dataset import *
from cnn.network import build_cnn
from cnn.maxgroup import *

class TrainCNNModel(ModelDesc):

    def __init__(self, image_size=32):
        self.image_size = image_size

    def _get_inputs(self):
        """
        Define all the inputs (with type, shape, name) that
        the graph will need.
        """
        return [InputDesc(tf.float32, (None, self.image_size, self.image_size), 'input'),
                InputDesc(tf.int32, (None,), 'label')]

    def _build_graph(self, inputs):
        """This function should build the model which takes the input variables
        and define self.cost at the end"""

        # inputs contains a list of input variables defined above
        image, label = inputs

        logits = build_cnn(image)

        # print the predicted labels for the first data point in each step.
        logits = tf.Print(logits, [tf.nn.softmax(logits, name='sm')], summarize=36)
        label = tf.Print(label, [label], summarize=1)

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
        # wd_cost = tf.multiply(1e-5,
        #                      regularize_cost('fc.*/W', tf.nn.l2_loss),
        #                      name='regularize_loss')
        self.cost = tf.add_n([cost], name='total_cost')
        summary.add_moving_summary(cost, self.cost)


        # monitor histogram of all weight (of conv and fc layers) in tensorboard
        summary.add_param_summary(('.*/W', ['histogram', 'rms']))

    def _get_optimizer(self):
        # decay every x epoches by lr_decay_rate
        lr = tf.train.exponential_decay(
            learning_rate=1e-3,
            global_step=get_global_step_var(),
            decay_steps=10 * self.steps_per_epoch  ,  # 74 * 5,
            decay_rate=self.lr_decay_rate, staircase=True, name='learning_rate')
        # This will also put the summary in tensorboard, stat.json and print in terminal
        # but this time without moving average
        tf.summary.scalar('lr', lr)

        return tf.train.AdamOptimizer(lr)


def get_data(unique=False, sub_data=None, batch_size=128):
    ds_train = data.utils.load_lmdb(IIIT5KChar('train'))
    ds_test = data.utils.load_lmdb(IIIT5KChar('test'))

    if unique:
        print("Use one data point per label")
        ds_train = UniqueData(ds_train)
        ds_test = UniqueData(ds_test)

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


def get_config(data, max_epoch=1000, lr_decay_rate=0.98, run_inference=True):
    dataset_train, dataset_test = data

    # How many iterations you want in each epoch.
    # This is the default value, don't actually need to set it in the config
    steps_per_epoch = dataset_train.size()
    # steps_per_epoch = 2 #TODO: remove this for actual training

    model = TrainCNNModel()
    model.steps_per_epoch = steps_per_epoch
    model.max_epoch = max_epoch
    model.lr_decay_rate = lr_decay_rate

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
#        session_init=SaverRestore('train_log/jan0121-200909/checkpoint')
    )


def train(unique=False, sub_data=None, batch_size=None):
    print("train cnn")

    # automatically setup the directory for logging
    logger.set_logger_dir(cfg.TRAIN_LOG_DIR)

    data = get_data(unique=unique,
                    sub_data=sub_data,
                    batch_size=batch_size)

    c = get_config(data)

    #TODO change trainer
    launch_train_with_config(c, SimpleTrainer())
