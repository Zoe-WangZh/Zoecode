# -*- coding: utf-8 -*-

# import h5py
import numpy as np
import scipy.io as sio


def read_matdata(filepath, keys):
    data = {}
    f = sio.loadmat(filepath)
    # f = h5py.File(filepath, 'r')
    for key in keys:
        data[key] = f[key]
    return data


class Dataset(object):
    def __init__(self, inputs):
        self._data = inputs
        self._num_examples = self._data.shape[0]
        self._indices = np.arange(self._num_examples)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def data(self):
        return self._data

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def shuffle(self):
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._indices = perm

    def get_portiondata(self, indices):
        return self._data[indices]

    def get_subset(self, ratio, shuffle=True):
        ratio = ratio / np.sum(ratio)
        num_total = self.num_examples
        num_each = (num_total * ratio).astype(int)
        ends = np.cumsum(num_each)
        ends[-1] = num_total
        starts = np.copy(ends)
        starts[1:]  = starts[0:-1]
        starts[0] = 0
        if shuffle: self.shuffle()
        return [Dataset(self.get_portiondata(self._indices[start:end])) for (start, end) in (starts, ends)]

    def next_batch(self, batch_size, shuffle=True):
        '''Return the next `batch_size` examples from this data set.'''
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            self.shuffle()
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            indices_rest_part = self._indices[start:self._num_examples]
            # Shuffle the data
            if shuffle: self.shuffle()
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            indices_new_part = self._indices[start:end]
            batch = self.get_portiondata(np.concatenate((indices_rest_part, indices_new_part), axis=0))
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            batch = self.get_portiondata(self._indices[start:end])
        return batch