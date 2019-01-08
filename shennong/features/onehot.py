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
import operator

import numpy as np

import shennong.core.window
from shennong.core.processor import FeaturesProcessor
from shennong.core.frames import Frames
from shennong.features.features import Features


class _OneHotBase(FeaturesProcessor):
    def __init__(self, phones=None):
        if phones is None:
            self.phones = None
        else:
            self.phones = sorted(set(phones))

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
    :class:`features.Features` while preserving the timestamps of the
    original alignment.

    Parameters
    ----------
    phones : sequence, optional
        The phones composing the alignment. Specify the phones if you
        want to have consistant one-hot vectors accross different
        :class:`Features`. By default the phones are extracted from
        the alignment in :meth:`process`.
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

        # check `times` is correct
        if times not in ('mean', 'onset', 'offset'):
            raise ValueError(
                'times must be "mean", "onset" or "offset" but is'
                .format(times))
        self.times = times

    def process(self, alignment):
        # build a bijection phone <-> onehot index
        phone2index = self._phone2index(alignment)

        # initialize the data matrix with zeros, TODO should data be a
        # scipy.sparse matrix?
        data = np.zeros(
            (alignment.phones.shape[0], len(phone2index)), dtype=np.bool)

        # fill the data with onehot encoding of phones
        for i, p in enumerate(alignment.phones):
            data[i, phone2index[p]] = 1

        # times are simply (onset + offset) / 2 if 'mean' as
        # parameters, else 'onset' or 'offset
        if self.times is 'mean':
            times = alignment._times.mean(axis=1)
        elif self.times is 'onset':
            times = alignment.onsets
        else:
            times = alignment.offsets

        # add the phones index to the features proerties allows to
        # reconstruct the phones sequence from the onehot vectors
        prop = self.get_params()
        prop.update({'phone2index': phone2index})

        return Features(data, times, properties=prop)


class FramedOneHotProcessor(_OneHotBase):
    """One-hot encoding on framed signals

    Computes the one-hot encoding on framed signals (i.e. on
    overlapping time windows)

    Parameters
    ----------
    phones : sequence, optional
        The phones composing the alignment. Specify the phones if you
        want to have consistant one-hot vectors accross different
        :class:`Features`. By default the phones are extracted from
        the alignment in :func:`process`.
    sample_rate : int, optional
        Sample frequency used for frames, in Hz, default to 16kHz
    frame_shift : float, optional
        Frame shift in seconds, default to 10ms
    frame_length : float, optional
        Frame length in seconds, default to 25ms
    window_type : {'povey', 'hanning', 'hamming', 'rectangular', 'blackman'}
        The type of the window, default is 'povey' (like hamming but
        goes to zero at edges)
    blackman_coeff : float, optional
        The constant coefficient for generalized Blackman window, used
        only when `window_type` is 'blackman', default is 0.42.

    """
    def __init__(self, phones=None, sample_rate=16000,
                 frame_shift=0.01, frame_length=0.025,
                 window_type='povey', blackman_coeff=0.42):
        super().__init__(phones=phones)

        self.frame = Frames(
            sample_rate=sample_rate,
            frame_shift=frame_shift,
            frame_length=frame_length)

        self.window_type = window_type
        self.blackman_coeff = blackman_coeff

    @property
    def sample_rate(self):
        """The processor operation sample rate

        Must match the sample rate of the signal specified in
        `process`

        """
        return self.frame.sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        self.frame.sample_rate = value

    @property
    def frame_shift(self):
        """Frame shift in seconds"""
        return self.frame.frame_shift

    @frame_shift.setter
    def frame_shift(self, value):
        self.frame.frame_shift = value

    @property
    def frame_length(self):
        """Frame length in seconds"""
        return self.frame.frame_length

    @frame_length.setter
    def frame_length(self, value):
        self.frame.frame_length = value

    def process(self, alignment):
        # build a bijection phone <-> onehot index
        phone2index = self._phone2index(alignment)

        # sample the alignment at the requested sample rate
        sampled = alignment.at_sample_rate(self.frame.sample_rate)

        # get the frames as pairs (istart:istop)
        nframes = self.frame.nframes(sampled.shape[0])
        frame_boundaries = self.frame.boundaries(nframes)

        # allocate the features data
        data = np.zeros(
            (frame_boundaries.shape[0], len(phone2index)), dtype=np.bool)

        # allocate the window function
        window = shennong.core.window.window(
            self.frame.samples_per_frame, type=self.window_type,
            blackman_coeff=self.blackman_coeff)

        for i, (onset, offset) in enumerate(frame_boundaries):
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

        prop = self.get_params()
        prop.update({'phone2index': phone2index})

        return Features(
            data,
            frame_boundaries.mean(axis=1) / self.frame.sample_rate,
            properties=prop)
