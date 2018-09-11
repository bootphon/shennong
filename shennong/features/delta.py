"""Compute deltas on existing features

Uses the Kaldi implementation

"""

import kaldi.feat.functions
import kaldi.matrix
import numpy as np

from shennong.features.processor import FeaturesProcessor
from shennong.features.features import Features


class DeltaProcessor(FeaturesProcessor):
    def __init__(self, order=2, window=2):
        self._options = kaldi.feat.functions.DeltaFeaturesOptions()
        self.order = order
        self.window = window

    @property
    def order(self):
        """Order of delta computation"""
        return self._options.order

    @order.setter
    def order(self, value):
        self._options.order = value

    @property
    def window(self):
        """Parameter controlling window for delta computation

        The actual window size for each delta order is 1 + 2 *
        `window`. The behavior at the edges is to replicate the first
        or last frame.

        """
        return self._options.window

    @window.setter
    def window(self, value):
        self._options.window = value

    def labels(self):
        raise ValueError(
            'labels are created from the input features given to `process()`')

    def times(self, nframes):
        raise ValueError(
            'times are created from the input features given to `process()`')

    def process(self, features):
        """Compute deltas on `features` with the specified options

        Parameters
        ----------
        features : Features, shape = [nframes, nlabels]
            The input features on which to compute the deltas

        Returns
        -------
        deltas : Features, shape = [nframes, nlabels * `order`]
            The computed deltas with as much orders as specified.

        """
        data = kaldi.matrix.SubMatrix(
            kaldi.feat.functions.compute_deltas(
                self._options, kaldi.matrix.SubMatrix(features.data))).numpy()

        labels = []
        for o in range(self.order):
            labels += [l + 'delta_{}'.format(o) for l in features.labels()]
        labels = np.asarray(labels)

        return Features(data, labels, features.times(), self.parameters())
