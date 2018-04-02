
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
from rnn.rnn_network import FeaturePredictor




def test(path, cnn_model, rnn_model):

    cnn = CharacterPredictor(cnn_model)
    rnn = FeaturePredictor(rnn_model)

    print("Load image from {}".format(path))
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Failed to load image!")
        exit()

    feats = list(cnn.predict_features(img, step_size=8))
    label = rnn.predict_label(feats)

    print("Label: {}".format(label))
