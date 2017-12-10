import os.path
import config
import cv2
import numpy as np

from abc import abstractmethod

from tensorpack.dataflow.dftools import dump_dataflow_to_lmdb
from tensorpack.dataflow.base import RNGDataFlow


__all__ = ['DataSet', 'lmdb_file', 'lmdb_path']


def lmdb_file(dataset, mode):
    return dataset.DATASET_NAME + "_" + mode + ".lmdb"


def lmdb_path(dataset, mode):
    return os.path.join(config.DATA_DIR, lmdb_file(dataset, mode))


class DataSet(RNGDataFlow):
    """
       Produces [image, label] in any dataset,
    """

    DATASET_NAME = None


    def __init__(self, train_or_test, shuffle=True, dir=None):
        """
            Args:
                train_or_test (str): either 'train' or 'test'
                shuffle (bool): shuffle the dataset
                dir (str): the location of the dataset
        """
        assert train_or_test in ['train', 'test']

        # check if dir was set
        if dir is None:
            self.dir = self.get_default_data_location()
        else:
            self.dir = dir

        # Init other vars
        self.train_or_test = train_or_test
        self.shuffle = shuffle

        # Load labels and images
        self.files, self.labels = self.load_data()

        # reset data set
        self.reset_state()


    def get_default_data_location(self):
        return os.path.join(config.DATA_DIR, self.DATASET_NAME)


    @abstractmethod
    def load_data(self):
        """
            Load the data from the self.dir
            Returns:
                 A list with file paths to the images and a list of labels.
        """


    def size(self):
        return len(self.files)

    def get_data(self):
        idxs = list(range(self.size()))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            with open(self.files[k], 'rb') as f:
                jpeg = f.read()
            jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
            label = self.labels[k]
            yield [jpeg, label]


    def isTrainData(self):
        """
            Checks if the train data set was loaded.
        """
        return self.train_or_test == 'train'


    def isTestData(self):
        """
            Checks if the test data set was loaded.
        """
        return self.train_or_test == 'test'

    def dump_to_lmdb(self, dir):
        file_name = lmdb_file(self, self.train_or_test)
        dump_dataflow_to_lmdb(self, os.path.join(dir, file_name))

