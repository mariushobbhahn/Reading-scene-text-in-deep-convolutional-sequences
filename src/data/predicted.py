import itertools

from tensorpack.dataflow import *
from tensorpack.dataflow.base import DataFlow


class PredictFeatures(DataFlow):

    def __init__(self, dataflow, predictor, step_size=16):
        self.wrapped = dataflow
        self.step_size = step_size
        self.predictor = predictor
        self.reset_state()

    def size(self):
        return self.wrapped.size()

    def reset_state(self):
        self.wrapped.reset_state()

    def get_data(self):
        # Predict the features in every image
        for img, label in self.wrapped.get_data():
            f = self.predictor.predict_features(img, self.step_size)
            yield list(f), label

