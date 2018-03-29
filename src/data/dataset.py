import itertools

import numpy as np
from tensorpack.dataflow import *
from tensorpack.dataflow.base import DataFlow, ProxyDataFlow
from tensorpack.dataflow.dftools import dump_dataflow_to_lmdb
import data.utils

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


def _batch_feature(feats):
    # pad to the longest in the batch
    maxlen = max([k.shape[0] for k in feats])
    bsize = len(feats)
    ret = np.zeros((bsize, maxlen, feats[0].shape[1]))
    for idx, feat in enumerate(feats):
        ret[idx, :feat.shape[0], :] = feat
    return ret


def _sparse_label(labels):
    batchsize = len(labels)
    maxlen = max([len(label) for label in labels])

    shape = (batchsize, maxlen)
    indices = []
    values = []

    current_batch = 0

    for batch, label in enumerate(labels):

        for idx, char in enumerate(label):
            indices.append([batch, idx])
            values.append(data.utils.char_to_int_label(char))

    indices = np.asarray(indices)
    values = np.asarray(values)
    return (indices, values, shape)


class BatchedFeatures(ProxyDataFlow):

    def __init__(self, ds, batch):
        self.batch = batch
        self.ds = ds

    def size(self):
        return self.ds.size() // self.batch

    def get_data(self):
        itr = self.ds.get_data()
        for _ in range(self.size()):
            feats = []
            labs = []
            for b in range(self.batch):
                feat, lab = next(itr)

                if feat.shape[0] < len(lab):
                    b -= 1
                    continue

                feats.append(feat)
                labs.append(lab)
            batchfeat = _batch_feature(feats)
            batchlab = _sparse_label(labs)
            seqlen = np.asarray([k.shape[0] for k in feats])

            # print("Process label: {}".format(labs[0]))
            # print("Features: {}, label: {}".format(batchfeat.shape, batchlab))

            yield [batchfeat, batchlab[0], batchlab[1], batchlab[2], seqlen]
