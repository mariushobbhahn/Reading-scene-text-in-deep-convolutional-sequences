import itertools
from tensorpack.dataflow import DataFlow


class SubData(DataFlow):
    def __init__(self, data, count, start=0, step=1):
        self.start = start
        self.step = step
        self.count = count
        self.data = data
        self.reset_state()

    def get_data(self):
        elem = itertools.islice(self.data.get_data(), self.start, self.start + self.count * self.step, self.step)
        for img, label in elem:
            yield [img, label]

    def size(self):
        return self.count
