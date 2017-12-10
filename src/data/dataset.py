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
            # Load image as grayscale
            file = self.files[k]
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

            # Check if height is 32px
            h = img.shape[0]

            if h != 32:
                # Resize to 32px
                f = 32.0 / h
                img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)

            # Encode image data
            success, raw = cv2.imencode(".png", img)

            if success:
                png = np.asarray(bytearray(raw), dtype='uint8')
                label = self.labels[k]
                yield [png, label]

    def is_train_data(self):
        """
            Checks if the train data set was loaded.
        """
        return self.train_or_test == 'train'

    def is_test_data(self):
        """
            Checks if the test data set was loaded.
        """
        return self.train_or_test == 'test'

    def dump_to_lmdb(self, dir):
        file_name = lmdb_file(self, self.train_or_test)
        dump_dataflow_to_lmdb(self, os.path.join(dir, file_name))

