"""One hot encoding of time-aligned tokens

One hot features are built from a time alignement of the spoken
tokenmes. They come in two flavours:

* :class:`OneHotProcessor` simply encode tokens in an alignment into
  on hot vectors

* :class:`FramedOneHotProcessor` includes the alignment into windowed
  frames before doing the one hot encoding


    :class:`~shennong.alignment.Alignment` ---> {Framed}OneHotProcessor \
    ---> :class:`~shennong.features.features.Features`

"""

import collections
import operator

import numpy as np

import shennong.features.window
from shennong.features import Features
from shennong.features.frames import Frames
from shennong.features.processor.base import FeaturesProcessor


class _OneHotBase(FeaturesProcessor):
    def __init__(self, tokens=None):
        self.tokens = tokens

    @property
    def tokens(self):
        return self._tokens

    @tokens.setter
    def tokens(self, value):
        if value is None:
            self._tokens = None
        else:
            self._tokens = sorted(set(value))

    def _tokens_set(self, alignment):
        # if no tokens list specified, take them from the alignment
        if self.tokens is None:
            return alignment.get_tokens_inventory()
        else:
            errors = [p for p in set(alignment.tokens) if p not in self.tokens]
            if errors != []:
                raise ValueError(
                    'following tokens are in alignment but not defined in the '
                    'onehot features processor: {}'.format(errors))
        return self.tokens

    def _token2index(self, alignment):
        tokens = self._tokens_set(alignment)
        return {p: i for i, p in enumerate(sorted(tokens))}


class OneHotProcessor(_OneHotBase):
    """Simple version of one hot features encoding

    The `OneHotProcessor` directly converts an :class:`Alignment` to
    :class:`features.Features` while preserving the timestamps of the
    original alignment.

    Parameters
    ----------
    tokens : sequence, optional
        The tokens composing the alignment. Specify the tokens if you
        want to have consistant one-hot vectors accross different
        :class:`Features`. By default the tokens are extracted from
        the alignment in :meth:`process`.

    """
    def __init__(self, tokens=None):
        super().__init__(tokens=tokens)

    def process(self, alignment):
        # build a bijection token <-> onehot index
        token2index = self._token2index(alignment)

        # initialize the data matrix with zeros, TODO should data be a
        # scipy.sparse matrix?
        data = np.zeros(
            (alignment.tokens.shape[0], len(token2index)), dtype=np.bool)

        # fill the data with onehot encoding of tokens
        for i, p in enumerate(alignment.tokens):
            data[i, token2index[p]] = 1

        # add the tokens index to the features proerties allows to
        # reconstruct the tokens sequence from the onehot vectors
        prop = self.get_params()
        prop.update({'token2index': token2index})

        return Features(data, alignment.times, properties=prop)


class FramedOneHotProcessor(_OneHotBase):
    """One-hot encoding on framed signals

    Computes the one-hot encoding on framed signals (i.e. on
    overlapping time windows)

    Parameters
    ----------
    tokens : sequence, optional
        The tokens composing the alignment. Specify the tokens if you
        want to have consistant one-hot vectors accross different
        :class:`Features`. By default the tokens are extracted from
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
    def __init__(self, tokens=None, sample_rate=16000,
                 frame_shift=0.01, frame_length=0.025,
                 window_type='povey', blackman_coeff=0.42):
        super().__init__(tokens=tokens)

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
        # build a bijection token <-> onehot index
        token2index = self._token2index(alignment)

        # sample the alignment at the requested sample rate
        sampled = alignment.at_sample_rate(self.frame.sample_rate)

        # get the frames as pairs (istart:istop)
        nframes = self.frame.nframes(sampled.shape[0])
        frame_boundaries = self.frame.boundaries(nframes)

        # allocate the features data
        data = np.zeros(
            (frame_boundaries.shape[0], len(token2index)), dtype=np.bool)

        # allocate the window function
        window = shennong.features.window.window(
            self.frame.samples_per_frame, type=self.window_type,
            blackman_coeff=self.blackman_coeff)

        for i, (onset, offset) in enumerate(frame_boundaries):
            framed = sampled[onset:offset]
            # the frame is made of a single token, no needs to compute
            # a window function
            if np.all(framed[0] == framed[1:]):
                winner = framed[0]
            else:
                # several tokens in the frame, compute the weights

                weights = collections.defaultdict(int)
                for j, w in enumerate(window):
                    weights[framed[j]] += w

                # the winner token has the biggest weight
                winner = sorted(
                    weights.items(),
                    key=operator.itemgetter(1),
                    reverse=True)[0][0]

            data[i, token2index[winner]] = 1

        prop = self.get_params()
        prop.update({'token2index': token2index})

        return Features(
            data,
            frame_boundaries / self.frame.sample_rate,
            properties=prop)
