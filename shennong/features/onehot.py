"""One hot encoding of time-aligned phones

One hot features are built from a time alignement of the spoken
phonemes. They come in two flavours:

* :class:`OneHotProcessor` simply encode phones in an alignment into
  on hot vectors

* :class:`FramedOneHotProcessor` includes the alignment into windowed
  frames before doing the one hot encoding


    :class:`~shennong.alignment.Alignment` ---> {Framed}OneHotProcessor \
    ---> :class:`~shennong.features.features.Features`

"""

import collections
import math
import operator

import numpy as np

from shennong.core.processor import FeaturesProcessor
from shennong.features.features import Features
import shennong.core.window


class _OneHotBase(FeaturesProcessor):
    def __init__(self, phones=None):
        if phones is None:
            self.phones = None
        else:
            self.phones = sorted(set(phones))

    def parameters(self):
        return {'phones': self.phones}

    def _phones_set(self, alignment):
        # if no phones list specified, take them from the alignment
        if self.phones is None:
            return alignment.get_phones_inventory()
        else:
            errors = [p for p in set(alignment.phones) if p not in self.phones]
            if errors != []:
                raise ValueError(
                    'following phones are in alignment but not defined in the '
                    'onehot features processor: {}'.format(errors))
        return self.phones

    def _phone2index(self, alignment):
        phones = self._phones_set(alignment)
        return {p: i for i, p in enumerate(sorted(phones))}


class OneHotProcessor(_OneHotBase):
    """Simple version of one hot features encoding

    The `OneHotProcessor` directly converts an :class:`Alignment` to
    :class:`features.Features` while preserving the timestamps of the original
    alignment.

    Parameters
    ----------
    phones : sequence, optional
        The phones composing the alignment. Specify the phones if you
        want to have consistant one-hot vectors accross different
        :class:`Features`. By default the phones are extracted from
        the alignment in :func:`process`.
    times : {'mean', 'onset', 'offset'}, optional
        The features timestamps are either the alignments `onsets`,
        `offsets` or :math:`\\frac{onset + offset}{2}` if `mean` is
        choosen. Default is `mean`.

    Raises
    ------
    ValueError
        If `times` is not `mean`, `onset` or `offset`

    """
    def __init__(self, phones=None, times='mean'):
        super().__init__(phones=phones)

        if times is 'onsets':
            times = 'onset'
        if times is 'offsets':
            times = 'offset'
        if times is 'means':
            times = 'mean'
        if times not in ('mean', 'onset', 'offset'):
            raise ValueError(
                'times must be "mean", "onset" or "offset" but is'
                .format(times))
        self._times = times

    def parameters(self):
        params = super().parameters()
        params.update({'times': self._times})
        return params

    def process(self, alignment):
        # build a bijection phone <-> onehot index
        phone2index = self._phone2index(alignment)

        # initialize the data matrix with zeros, dtype is int8 because
        # np.bool_ are stored as bytes as well TODO should data be a
        # scipy.sparse matrix?
        data = np.zeros(
            (alignment.phones.shape[0], len(phone2index)), dtype=np.int8)

        # fill the data with onehot encoding of phones
        for i, p in enumerate(alignment.phones):
            data[i, phone2index[p]] = 1

        # times are simply (onset + offset) / 2 if 'mean' as
        # parameters, else 'onset' or 'offset
        if self._times is 'mean':
            times = alignment._times.mean(axis=1)
        elif self._times is 'onset':
            times = alignment.onsets
        else:
            times = alignment.offsets

        prop = self.parameters()
        prop.update({'phone2index': phone2index})

        return Features(data, times, properties=prop)


class FramedOneHotProcessor(_OneHotBase):
    def __init__(self, phones=None, sample_rate=16000,
                 frame_shift=0.01, frame_length=0.025,
                 window_type='povey', blackman_coeff=0.42):
        super().__init__(phones=phones)

        self.sample_rate = sample_rate
        self.frame_shift = frame_shift
        self.frame_length = frame_length
        self.window_type = window_type
        self.blackman_coeff = blackman_coeff

    def parameters(self):
        params = super().parameters()
        params.update({
            'sample_rate': self.sample_rate,
            'frame_shitf': self.frame_shift,
            'frame_length': self.frame_length,
            'window_type': self.window_type,
            'blackman_coeff': self.blackman_coeff})
        return params

    def process(self, alignment):
        # build a bijection phone <-> onehot index
        phone2index = self._phone2index(alignment)

        # sample the alignment at the requested sample rate
        sampled = alignment.at_sample_rate(self.sample_rate)

        shift = self.frame_shift * self.sample_rate
        length = self.frame_length * self.sample_rate

        k = math.ceil((sampled.shape[0] - length) / shift)
        frames = np.repeat(np.arange(k), 2).reshape(k, 2)
        frames = (frames * shift + (0, length)).astype(np.int)
        assert frames[0, 0] == 0
        assert frames[-1, 1] <= sampled.shape[0]

        # allocate the features data
        data = np.zeros((k, len(phone2index)), dtype=np.int8)

        # allocate the window function
        window = shennong.core.window.window(
            length, type=self.window_type,
            blackman_coeff=self.blackman_coeff)

        for i, (onset, offset) in enumerate(frames):
            framed = sampled[onset:offset]
            # the frame is made of a single phone, no needs to compute
            # a window function
            if np.all(framed[0] == framed[1:]):
                winner = framed[0]
            else:
                # several phones in the frame, compute the weights

                weights = collections.defaultdict(int)
                for j, w in enumerate(window):
                    weights[framed[j]] += w

                # the winner phone has the biggest weight
                winner = sorted(
                    weights.items(),
                    key=operator.itemgetter(1),
                    reverse=True)[0][0]

            data[i, phone2index[winner]] = 1

        prop = self.parameters()
        prop.update({'phone2index': phone2index})

        return Features(
            data,
            frames.mean(axis=1) / self.sample_rate,
            properties=prop)
