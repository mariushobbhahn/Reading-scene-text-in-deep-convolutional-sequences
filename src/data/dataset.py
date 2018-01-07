import os
import config
import cv2
import itertools

from tensorpack.dataflow import *
from data.iiit5k import IIIT5KHelper
from tensorpack.dataflow.dftools import dump_dataflow_to_lmdb


DUMP=False

def _lmdb_file(name, train_or_test, char_data):
    str = "_chars" if char_data else ""
    return name + "_" + train_or_test + str + ".lmdb"


def _lmdb_path(name, train_or_test, char_data):
    return os.path.join(config.DATA_DIR, _lmdb_file(name, train_or_test, char_data))


def _load_or_create_ds(helper, shuffle):
    """
    Uses a heper data set to load or create a lmdb data set.
    :param helper: Helper data set to create lmdb if needed.
    :param shuffle: If `true` the returned dataset will be shuffled.
    :return: A lmdb data set
    """
    path = _lmdb_path(helper.name, helper.train_or_test, helper.is_char_data)

    # Check if lmdb exists
    if not os.path.exists(path):
        dump_dataflow_to_lmdb(helper, path)

    # Load lmdb
    ds = LMDBData(path, shuffle=False)

    # shuffle if needed
    if shuffle:
        ds = LocallyShuffleData(ds, 50000)

    ds = PrefetchData(ds, 5000, 1)

    # Decode images
    ds = LMDBDataPoint(ds)
    ds = MapDataComponent(ds, lambda x: cv2.imdecode(x, cv2.IMREAD_GRAYSCALE), 0)
    ds = PrefetchDataZMQ(ds, 25)

    return ds


def dump_helper(helper, output_dir=None, count=100):
    """
    Dumps the given amout of images form the helper data.

    :param helper:
    :param output_dir:
    :param count:
    :return:
    """
    if output_dir is None:
        s = "_char" if helper.is_char_data else ""
        output_dir = os.path.join(config.DATA_DIR, "dump_{}_{}{}".format(helper.name, helper.train_or_test, s))

    data = itertools.islice(helper.get_data(), count)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Dump to {}".format(output_dir))

    for raw, label in data:
        img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        file = os.path.join(output_dir, "{}.png".format(label))
        cv2.imwrite(file, img)


def IIIT5K(train_or_test, char_data=False, shuffle=False):
    """
    Creates or loads the IIIT5K data set.

    :param train_or_test: controls which subset is loaded.
    :param char_data: if 'True' a version of the data set for character recognition is loaded.
    :param shuffle: if `True` the elements of the data set will be shuffled.
    :return:
    """
    helper = IIIT5KHelper(train_or_test, char_data)

    if DUMP:
        dump_helper(helper)
        
    return _load_or_create_ds(helper, shuffle)
