import itertools

from tensorpack.dataflow import *
from tensorpack.dataflow.base import DataFlow
from tensorpack.dataflow.dftools import dump_dataflow_to_lmdb


class NamedDataFlow(DataFlow):
    """
        Subclass of DataFlow which has a unique name
    """
    def __init__(self, name):
        self._name = name

    def size(self):
        pass

    def get_data(self):
        pass

    def get_name(self):
        """
        Returns the name of this data set. The name can depend on the train or test config.
        :return:
        """
        return self._name


class SubData(DataFlow):
    """
        DataFlow wrapper which only contains a subset of the wrapped data points.
    """
    def __init__(self, data, count, start=0, step=1):
        self.start = start
        self.step = step
        self.count = count
        self.data = data
        self.reset_state()

    def reset_state(self):
        self.data.reset_state()

    def get_data(self):
        elem = itertools.islice(self.data.get_data(), self.start, self.start + self.count * self.step, self.step)
        for img, label in elem:
            yield [img, label]

    def size(self):
        return self.count


class UniqueData(DataFlow):
    """
        DataFlow wrapper which contains one data point per label.
    """
    def __init__(self, data):
        self.data = data
        # Count will be init lazy
        self.count = None
        self.reset_state()

    def reset_state(self):
        self.data.reset_state()

    def get_data(self):
        """
            yields one data point per label.
        """
        self.reset_state()
        known_labels = set()
        index = 0
        lastindex = -10

        for img, label in self.data.get_data():
            index = index + 1
            if label not in known_labels and index > lastindex + 6:
                known_labels.add(label)
                yield [img, label]

    def size(self):
        if self.count is None:
            # count data points
            self.count = sum(1 for _ in self.get_data())

        return self.count
