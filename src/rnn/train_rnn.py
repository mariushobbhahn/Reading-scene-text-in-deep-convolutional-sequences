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

def get_data(model, step_size, unique, sub_data, batch_size):
    ds_train = data.utils.load_lmdb(IIIT5K('train', unique=unique))
    ds_test = data.utils.load_lmdb(IIIT5K('test', unique=unique))

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
    #ds_train = BatchData(ds_train, batch_size)
    #ds_test = BatchData(ds_test, 2 * batch_size, remainder=True)

    predictor = CharacterPredictor(model)

    ds_train = PredictFeatures(ds_train, predictor, step_size=step_size)
    ds_test  = PredictFeatures(ds_test, predictor, step_size=step_size)

    return ds_train, ds_test


def train_rnn(model, step_size=16, unique=False, sub_data=None, batch_size=None):
    (ds_train, ds_test) = get_data(model, step_size, unique, sub_data, batch_size)

    for (features, label) in ds_train.get_data():
        print('{}: {}'.format(label, features))
