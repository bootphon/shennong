"""Compute time derivatives on existing features

Uses the Kaldi implementation (see [kaldi-delta]_):

    :class:`~shennong.features.features.Features` -->
    DeltaPostProcessor -->
    :class:`~shennong.features.features.Features`

Examples
--------

>>> import numpy as np
>>> from shennong.audio import Audio
>>> from shennong.features.processor.mfcc import MfccProcessor
>>> from shennong.features.postprocessor.delta import DeltaPostProcessor
>>> audio = Audio.load('./test/data/test.wav')
>>> mfcc = MfccProcessor().process(audio)

Initialize the delta processor and compute first and second order time
derivatives of MFCC features:

>>> processor = DeltaPostProcessor(order=2)
>>> delta = processor.process(mfcc)

The resulting matrice is the concatenation of the original features,
their first and second order derivatives:

>>> nmfcc = mfcc.shape[1]
>>> delta.shape[1] == nmfcc * 3
True
>>> original = delta.data[:, :nmfcc]
>>> np.array_equal(original, mfcc.data)
True
>>> first_order = delta.data[:, nmfcc:2*nmfcc]
>>> second_order = delta.data[:, 2*nmfcc:]
>>> original.shape == first_order.shape == second_order.shape
True

References
----------

.. [kaldi-delta] http://kaldi-asr.org/doc/feature-functions_8h.html

"""

import copy
import kaldi.feat.functions
import kaldi.matrix

from shennong.features import Features
from shennong.features.postprocessor.base import FeaturesPostProcessor


class DeltaPostProcessor(FeaturesPostProcessor):
    def __init__(self, order=2, window=2):
        super().__init__()

        self._options = kaldi.feat.functions.DeltaFeaturesOptions()
        self.order = order
        self.window = window

    @property
    def name(self):
        return 'delta'

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
        if not 0 < value < 1000:
            raise ValueError(
                'window must be in [1, 999], it is {}'.format(value))
        self._options.window = value

    @property
    def ndims(self):
        raise ValueError(
            'output dimension for delta processor depends on input')

    def get_properties(self, features):
        ndims = (self.order + 1) * features.ndims
        properties = copy.deepcopy(features.properties)
        properties[self.name] = {
            'order': self.order,
            'window': self.window}

        if 'pipeline' not in properties:
            properties['pipeline'] = []

        properties['pipeline'].append({
            'name': self.name,
            'columns': [0, ndims - 1]})

        return properties

    def process(self, features):
        """Compute deltas on `features` with the specified options

        Parameters
        ----------
        features : Features, shape = [nframes, ncols]
            The input features on which to compute the deltas

        Returns
        -------
        deltas : Features, shape = [nframes, ncols * (`order` + 1)]
            The computed deltas with as much orders as specified. The
            output features are the concatenation of the input
            `features` and it's time derivative at each orders.

        """
        data = kaldi.matrix.SubMatrix(
            kaldi.feat.functions.compute_deltas(
                self._options, kaldi.matrix.SubMatrix(features.data))).numpy()

        return Features(
            data,
            features.times,
            self.get_properties(features))
