import itertools

import numpy as np
from tensorpack.dataflow import *
from tensorpack.dataflow.base import DataFlow


class PredictFeatures(DataFlow):

    def __init__(self, dataflow, predictor, step_size=16, list_size=36):
        self.wrapped = dataflow
        self.step_size = step_size
        self.predictor = predictor
        self.reset_state()
        self.list_size = list_size

    def size(self):
        return self.wrapped.size()

    def reset_state(self):
        self.wrapped.reset_state()

    def get_data(self):
        # Predict the features in every image
        for img, label in self.wrapped.get_data():
            f = np.array(list(self.predictor.predict_features(img, self.step_size)))

            num_vectors = f.shape[0]

            # print("Feature shape: {}".format(f.shape))

            # print("Type: {}".format(f.dtype))

            # Ignore data points with less feature vectors than characters.
            if num_vectors < len(label):
                continue

            # if num_vectors < self.list_size:
            #     padding = np.zeros(dtype=np.float32, shape=(self.list_size - num_vectors, 128))
            #     # print("Padding shape: {}".format(padding.shape))
            #     f = np.append(f, padding, axis=0)

            # print("Final shape: {}".format(f.shape))

            yield f, label

class MultiPredictFeatures(DataFlow):

    def __init__(self, dataflow, predictors, step_size=16, list_size=36):
        self.wrapped = dataflow
        self.step_size = step_size
        self.predictors = predictors
        self.reset_state()
        self.list_size = list_size

    def size(self):
        return self.wrapped.size()

    def reset_state(self):
        self.wrapped.reset_state()

    def get_data(self):
        # Predict the features in every image
        for img, label in self.wrapped.get_data():
            # call all n predictors and create a sequence of 128*n-D feature vectors
            f = np.array([list(predictor.predict_features(img, self.step_size)) for predictor in self.predictors])
            
            # TODO f = predictor.predict_features for predictor in predictors
            num_vectors = f.shape[1]

            # print("Feature shape: {}".format(f.shape))

            # print("Type: {}".format(f.dtype))

            # Ignore data points with less feature vectors than characters.
            if num_vectors < len(label):
                continue

            f = np.concatenate(f, axis=1)
            # if num_vectors < self.list_size:
            #     padding = np.zeros(dtype=np.float32, shape=(self.list_size - num_vectors, 128))
            #     # print("Padding shape: {}".format(padding.shape))
            #     f = np.append(f, padding, axis=0)

            # print("Final shape: {}".format(f.shape))
            
            if len(f) < len(label):
                print("not enough features: {}/{}".format(f.shape, label))
                continue

            yield f, label

