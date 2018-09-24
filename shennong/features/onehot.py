"""One hot encoding of time-aligned phones

One hot features are built from a time alignement of the spoken
phonemes. They come in two flavours:

* `OneHotProcessor` simply encode phones in an alignment into on hot
  vectors

* `FramedOneHotProcessor` includes the alignment into windowed frames
  before doing the one hot encoding


    *Alignement* ---> {Framed}OneHotProcessor ---> *Features*

"""

import collections
import numpy as np
import math

from shennong.features.features import Features
from shennong.features.processor import FeaturesProcessor
import shennong.window


class OneHotProcessor(FeaturesProcessor):
    """Simple version of one hot features encoding

    The `OneHotProcessor` directly converts an alignment to
    `Features`.

    """
    def __init__(self, phones=None):
        if phones is None:
            self.phones = None
        else:
            self.phones = sorted(set(phones))

    def parameters(self):
        return {'phones': self.phones}

    def times(self):
        raise NotImplementedError(
            'times are computed from alignment by the process() method')

    def process(self, alignment):
        # if no phones list specified, take them from the alignment
        phones = self.phones
        if phones is None:
            phones = sorted(set(alignment.phones))

        # build a bijection phone <-> onehot index
        phone2index = {p: i for i, p in enumerate(phones)}

        # initialize the data matrix with zeros, dtype is int8 because
        # np.bool_ are stored as bytes as well TODO should data be a
        # scipy.sparse matrix?
        data = np.zeros(
            (alignment.phones.shape[0], len(phone2index)),
            dtype=np.int8)

        # fill the data with onehot encoding of phones
        for i, p in enumerate(alignment.phones):
            data[i, phone2index[p]] = 1

        # times are simply (tstop - tstart) / 2
        times = alignment._times.mean(axis=1)

        return Features(data, times, properties={
            'phone2index': phone2index})


class FramedOneHotProcessor(FeaturesProcessor):
    def __init__(self, sample_rate=16000,
                 frame_shift=0.01, frame_length=0.025,
                 window_type='povey', blackman_coeff=0.42):
        self.sample_rate = sample_rate
        self.frame_shift = frame_shift
        self.frame_length = frame_length
        self.window_type = window_type
        self.blackman_coeff = blackman_coeff

    def parameters(self):
        return {
            'sample_rate': self.sample_rate,
            'frame_shitf': self.frame_shift,
            'frame_length': self.frame_length,
            'window_type': self.window_type,
            'blackman_coeff': self.blackman_coeff}

    def times(self, nframes):
        return np.arange(nframes) * self.frame_shift + self.frame_length / 2.0

    def _frames(self, alignment):
        """Returns the features frames as an array of (tstart, tstop) pairs"""
        # number of frames in the resulting features
        nframes = math.ceil(
            (alignment.duration() - self.frame_length) / self.frame_shift)

        # generate the frames boundaries as (tstart, tstop) pairs
        tstart = np.arange(nframes) * self.frame_shift
        tstop = tstart + self.frame_length
        return np.vstack((tstart, tstop)).T

    def process(self, alignment):
        # get the features frames as (tstart, tstop) pairs
        frames = self._frames(alignment)

        # generate the onehot vectors, one per frame
        data = np.zeros((frames.shape[0], len(alignment.phones_set)))
        for i, (tstart, tstop) in enumerate(frames):
            # read the phones for the current frame at the given
            # sample rate
            phones = list(alignment.at_sample_rate(
                self.sample_rate, tstart=tstart, tstop=tstop))

            nphones = len(phones)
            if nphones != 0:
                # the window function used to compute the phones weights
                window = shennong.window.window(
                    nphones, type=self.window_type,
                    blackman_coef=self.blackman_coeff)

                # compute the weight of each phone
                weight = collections.defaultdict(int)
                for i, w in enumerate(window):
                    weight[phones[i]] += w

                winner = 0  # TODO
                data[i, winner] = 1

        return Features(
            data, self.times(frames.shape[0]), properties=self.parameters())
