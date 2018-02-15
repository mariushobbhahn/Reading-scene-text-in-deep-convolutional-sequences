import cnn.network
import cv2
import numpy as np

from cnn.network import CharacterPredictor


def test(path, model):
    predictor = CharacterPredictor(model)

    print("Load image from {}".format(path))
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Failed to load image!")
        exit()

    for char in predictor.predict_characters(img, step_size=8, map_to_char=False):
        print("Found character: {}".format(char))
