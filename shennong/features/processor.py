"""Provides abstract base classes for the features extraction models"""

import abc
import numpy as np

from kaldi.feat.window import FrameExtractionOptions
from kaldi.feat.mel import MelBanksOptions


class FeaturesProcessor(metaclass=abc.ABCMeta):
    """Base class of all the features extraction models"""
    @abc.abstractmethod
    def parameters(self):
        """Returns a dict of all the processor's parameters"""
        pass

    @abc.abstractmethod
    def labels(self):
        """Returns the name of the columns given by the `process` method"""
        pass

    @abc.abstractmethod
    def times(self, nframes):
        """Returns the time labels of the rows given by the `process` method"""
        pass

    @abc.abstractmethod
    def process(self, signal):
        """Returns some features processed from an input `signal`"""
        pass


class MelFeaturesProcessor(FeaturesProcessor):
    """A base class for mel-based features processors

    The mel-based features are MFCC, PLP and filterbanks. The class
    implement common options for processing those features.

    References
    ----------

    .. [kaldi-frame]
      http://kaldi-asr.org/doc/structkaldi_1_1FrameExtractionOptions.html

    .. [kaldi-mel]
      http://kaldi-asr.org/doc/structkaldi_1_1MelBanksOptions.html

    """
    def __init__(self, sample_rate=16000, frame_shift=0.01,
                 frame_length=0.025, dither=1.0, preemph_coeff=0.97,
                 remove_dc_offset=True, window_type='povey',
                 round_to_power_of_two=True, blackman_coeff=0.42,
                 snip_edges=True, num_bins=23, low_freq=20,
                 high_freq=0, vtln_low=100, vtln_high=-500):
        # frame extraction options
        self._frame_options = FrameExtractionOptions()
        self.sample_rate = sample_rate
        self.frame_shift = frame_shift
        self.frame_length = frame_length
        self.dither = dither
        self.preemph_coeff = preemph_coeff
        self.remove_dc_offset = remove_dc_offset
        self.window_type = window_type
        self.round_to_power_of_two = round_to_power_of_two
        self.blackman_coeff = blackman_coeff
        self.snip_edges = snip_edges

        # mel banks options
        self._mel_options = MelBanksOptions()
        self.num_bins = num_bins
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.vtln_low = vtln_low
        self.vtln_high = vtln_high

    @property
    def sample_rate(self):
        """Waveform sample frequency in Hertz

        Must match the sample rate of the signal specified in
        `process`

        """
        return self._frame_options.samp_freq

    @sample_rate.setter
    def sample_rate(self, value):
        self._frame_options.samp_freq = value

    @property
    def frame_shift(self):
        """Frame shift in seconds"""
        return self._frame_options.frame_shift_ms / 1000.0

    @frame_shift.setter
    def frame_shift(self, value):
        self._frame_options.frame_shift_ms = value * 1000.0

    @property
    def frame_length(self):
        """Frame length in seconds"""
        return self._frame_options.frame_length_ms / 1000.0

    @frame_length.setter
    def frame_length(self, value):
        self._frame_options.frame_length_ms = value * 1000.0

    @property
    def dither(self):
        """Amount of dithering

        0.0 means no dither

        """
        return self._frame_options.dither

    @dither.setter
    def dither(self, value):
        self._frame_options.dither = value

    @property
    def preemph_coeff(self):
        """Coefficient for use in signal preemphasis"""
        return self._frame_options.preemph_coeff

    @preemph_coeff.setter
    def preemph_coeff(self, value):
        self._frame_options.preemph_coeff = value

    @property
    def remove_dc_offset(self):
        """If True, subtract mean from waveform on each frame"""
        return self._frame_options.remove_dc_offset

    @remove_dc_offset.setter
    def remove_dc_offset(self, value):
        self._frame_options.remove_dc_offset = value

    @property
    def window_type(self):
        """Type of window

        Must be 'hamming', 'hanning', 'povey', 'rectangular' or
        'blackman'

        """
        return self._frame_options.window_type

    @window_type.setter
    def window_type(self, value):
        windows = ['hamming', 'hanning', 'povey', 'rectangular', 'blackman']
        if value not in windows:
            raise ValueError(
                'window type must be in {}, it is {}'.format(windows, value))
        self._frame_options.window_type = value

    @property
    def round_to_power_of_two(self):
        """If true, round window size to power of two

        This is done by zero-padding input to FFT

        """
        return self._frame_options.round_to_power_of_two

    @round_to_power_of_two.setter
    def round_to_power_of_two(self, value):
        self._frame_options.round_to_power_of_two = value

    @property
    def blackman_coeff(self):
        """Constant coefficient for generalized Blackman window"""
        return self._frame_options.blackman_coeff

    @blackman_coeff.setter
    def blackman_coeff(self, value):
        self._frame_options.blackman_coeff = value

    @property
    def snip_edges(self):
        """If true, output only frames that completely fit in the file

        When True the number of frames depends on the `frame_length`.
        If false, the number of frames depends only on the
        `frame_shift`, and we reflect the data at the ends.

        """
        return self._frame_options.snip_edges

    @snip_edges.setter
    def snip_edges(self, value):
        self._frame_options.snip_edges = value

    @property
    def num_bins(self):
        """Number of triangular mel-frequency bins"""
        return self._mel_options.num_bins

    @num_bins.setter
    def num_bins(self, value):
        self._mel_options.num_bins = value

    @property
    def low_freq(self):
        """Low cutoff frequency for mel bins in Hertz"""
        return self._mel_options.low_freq

    @low_freq.setter
    def low_freq(self, value):
        self._mel_options.low_freq = value

    @property
    def high_freq(self):
        """High cutoff frequency for mel bins in Hertz

        If `high_freq` < 0, offset from the Nyquist frequency

        """
        return self._mel_options.high_freq

    @high_freq.setter
    def high_freq(self, value):
        self._mel_options.high_freq = value

    @property
    def vtln_low(self):
        """Low inflection point in piecewise linear VTLN warping function

        In Hertz

        """
        return self._mel_options.vtln_low

    @vtln_low.setter
    def vtln_low(self, value):
        self._mel_options.vtln_low = value

    @property
    def vtln_high(self):
        """High inflection point in piecewise linear VTLN warping function

        In Hertz. If `vtln_high` < 0, offset from `high_freq`

        """
        return self._mel_options.vtln_high

    @vtln_high.setter
    def vtln_high(self, value):
        self._mel_options.vtln_high = value

    def parameters(self):
        return {
            'sample_rate': self.sample_rate,
            'frame_shift': self.frame_shift,
            'frame_length': self.frame_length,
            'dither': self.dither,
            'preemph_coeff': self.preemph_coeff,
            'remove_dc_offset': self.remove_dc_offset,
            'window_type': self.window_type,
            'round_to_power_of_two': self.round_to_power_of_two,
            'blackman_coeff': self.blackman_coeff,
            'snip_edges': self.snip_edges,
            'num_bins': self.num_bins,
            'low_freq': self.low_freq,
            'high_freq': self.high_freq,
            'vtln_low': self.vtln_low,
            'vtln_high': self.vtln_high}

    def times(self, nframes):
        """Returns the time label for the rows given by the `process` method"""
        return np.arange(nframes) * self.frame_shift + self.frame_length / 2.0

    @abc.abstractmethod
    def labels(self):
        pass

    @abc.abstractmethod
    def process(self, signal):
        pass
