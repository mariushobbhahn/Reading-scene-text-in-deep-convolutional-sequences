import os.path
import config
import cv2

from tensorpack.dataflow.dftools import *
from tensorpack.dataflow import *


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
    # The path where the .mdb file should be located
    mdb_file = os.path.join(config.DATA_DIR, named_df.get_name() + ".mdb")

    # check if mdb must be created
    if not os.path.exists(mdb_file):
        df = MapData(named_df, lambda img, label: (img, label))
        dump_dataflow_to_lmdb(named_df, mdb_file)

    # Load lmdb and convert data points to images
    ds = LMDBData(mdb_file)
    ds = LMDBDataPoint(ds)
    ds = MapDataComponent(ds, lambda x: cv2.imdecode(x, cv2.IMREAD_GRAYSCALE), 0)

    return ds


