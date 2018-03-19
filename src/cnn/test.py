
import tensorflow as tf
import cv2
import numpy as np

# Just import everything into current namespace
from tensorpack import *
from tensorpack.tfutils import summary
from tensorpack.models.nonlin import Maxout
from tensorflow.python.platform import flags

from data.iiit5k import IIIT5KChar
from cnn.network import CharacterPredictor
from data.utils import int_label_to_char, load_lmdb

# from tensorflow.python.layers import maxout
from data.utils import convert_image_to_array


def print_wrong(model):

    ds = load_lmdb(IIIT5KChar('test'))
    p = CharacterPredictor(model)

    ds.reset_state()

    for img, label in ds.get_data():
        img = img.reshape((1, 32, 32)).astype('float32')
        predicted_label = np.argmax(p(img)[1][0])
        print("Predicted {} for {}".format(predicted_label, label))


def test(path, model):
    print_wrong(model)


    print("Load image from {}".format(path))
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Failed to load image!")
        exit()

    for char in predictor.predict_characters(img, step_size=8, map_to_char=True):
        char = int_label_to_char(np.argmax(char))
        print("Found character: {}".format(char))
