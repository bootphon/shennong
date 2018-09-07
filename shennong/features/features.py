"""Provides the Features class that handles features data"""


class Features(object):
    def __init__(self, data, labels, times, parameters):
        self._data = data
        self._labels = labels
        self._times = times
        self._parameters = parameters

        assert self.is_valid()

    def is_valid(self):
        """Return True if data, labels and times are consitent"""
        assert self._data.ndim == 2

        nframes = self._data.shape[0]
        assert self._times.ndim == 1
        assert self._times.shape[0] == nframes

        nlabels = self._data.shape[1]
        assert self._labels.ndim == 1
        assert self._labels.shape[0] == nlabels

        return True
