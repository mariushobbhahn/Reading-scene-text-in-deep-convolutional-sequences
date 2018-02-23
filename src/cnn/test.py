import cnn.network
import cv2
import numpy as np

from cnn.network import CharacterPredictor
from data.utils import *


def test(path, model):
    predictor = CharacterPredictor(model)

    print("Load image from {}".format(path))
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Failed to load image!")
        exit()

    for char in predictor.predict_characters(img, step_size=5, map_to_char=False):
        char = int_label_to_char(np.argmax(char))
        print("Found character: {}".format(char))
