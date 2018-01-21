import os.path
import config
import cv2
import numpy as np
import shutil

from tensorpack.dataflow.dftools import *
from tensorpack.dataflow import *
from data.sub_data import SubData


def char_to_int_label(char):
    """
    Converts a character to an integer label.
    :param char:
    :return:
    """

    # ascii index
    index = int(ord(char.lower()))

    # is character?
    if index > 60:
        index = index - 75

    return index - 22


def int_label_to_char(label):
    """
    Converts a integer label to the corresponding character
    :param label: the integer label
    :return: The corresponding character for the label.
    """

    # Is a label for a numer?
    if label > 25:
        # Cast number to string
        return str(label - 26)
    else:
        # Else lookup ascii character
        return chr(label + 97)


def load_lmdb(named_df):
    """
    Loads a LMDBDataFlow for the given named data set.

    :param named_df: The NamedDataSet for which the LMDB file should be loaded.
    :param reuse: reuses .mdb file from previous run. Default is True
    :return: A LMDBDataFlow which contains the same data points as the named df.
    """
    # The path where the .mdb file should be located
    mdb_file = os.path.join(config.DATA_DIR, named_df.get_name() + ".mdb")

    # remove old file to force recreation
    if config.REMOVE_LMDB and os.path.exists(mdb_file):
        print("Remove lmdb")
        os.remove(mdb_file)

    # check if mdb must be created
    if not os.path.exists(mdb_file):
        print("Create lmdb...")
        df = MapData(named_df, lambda p: (convert_image_to_array(p[0]), p[1]))
        dump_dataflow_to_lmdb(df, mdb_file)
        print("Done")

    # Load lmdb and convert data points to images
    ds = LMDBData(mdb_file)
    ds = LMDBDataPoint(ds)
    ds = MapDataComponent(ds, lambda x: cv2.imdecode(x, cv2.IMREAD_GRAYSCALE), 0)

    return ds


def convert_image_to_array(img):
    success, raw = cv2.imencode(".png", img)

    if success:
        return np.asarray(bytearray(raw), dtype='uint8')
    else:
        raise RuntimeError("Failed to convert image to png!")


def dump_data(df, dir, count=100, start=0, step=1):
    """
    Dumps part of the given data flow into the given dir.
    The files will be named "{index}_{label}.png" where label is the human-readable character.

    :param df: The data flow to be dumped
    :param dir: The directory in which the dump will be saved. (Will be removed before dumping)
    :param count: The number of data points to be dumped.
    :param start: The index of the first data point to be dumped.
    :param step: The step between points.
    """
    print("Start dump")
    df = SubData(df, count, start, step)

    # remove old dump
    if os.path.exists(dir):
        shutil.rmtree(dir)

    os.makedirs(dir)
    index = 0

    for (img, label) in df.get_data():
        char = int_label_to_char(label)

        file = os.path.join(dir, "{}_{}.png".format(index, char))
        index += 1
        cv2.imwrite(file, img)
