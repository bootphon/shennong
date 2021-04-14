"""This module implements the speech features extraction models (processors)

A speech features processor takes an audio signal as input and output features:

    :class:`~shennong.audio.Audio` --> FeaturesProcessor -->
    :class:`~shennong.features.Features`

"""

import abc
import kaldi.feat.window
import kaldi.feat.mel
import joblib
import numpy as np

from shennong import Features, FeaturesCollection
from shennong.base import BaseProcessor
from shennong.utils import get_njobs


class FeaturesProcessor(BaseProcessor, metaclass=abc.ABCMeta):
    """Base class of all the features extraction models"""
    @abc.abstractproperty
    def name(self):  # pragma: nocover
        """Name of the processor"""

    @abc.abstractproperty
    def ndims(self):  # pragma: nocover
        """Dimension of the output features frames"""

    def get_properties(self, **kwargs):
        """Return the processors properties as a dictionary"""
        params = self.get_params()
        params.update(kwargs)
        return {
            'pipeline': [
                {'name': self.name, 'columns': [0, self.ndims-1]}],
            self.name: params}

    @abc.abstractmethod
    def process(self, signal):
        """Returns features processed from an input `signal`

        Parameters
        ----------
        signal: :class`~shennong.audio.Audio`
            The input audio signal to process features on

        Returns
        -------
        features: :class:`~shennong.features.Features`
            The computed features

        """

    def process_all(self, utterances, njobs=None, **kwargs):
        """Returns features processed from several input `signals`

        This function processes the features in parallel jobs.

        Parameters
        ----------
        utterances: :class`~shennong.uttterances.Utterances`
            The utterances on which to process features on.
        njobs: int, optional
            The number of parallel jobs to run in background. Default
            to the number of CPU cores available on the machine.
        **kwargs: dict, optional
            Extra arguments to be forwarded to the `process` method. Keys must
            be the same as for `signals`.

        Returns
        -------
        features: :class:`~shennong.features_collection.FeaturesCollection`
            The computed features on each input signal. The keys of
            output `features` are the keys of the input `signals`.

        Raises
        ------
        ValueError
            If the `njobs` parameter is <= 0 or if an entry is missing in
            optioanl kwargs.

        """
        # checks the number of background jobs
        njobs = get_njobs(njobs, log=self.log)

        # check the extra arguments
        for name, value in kwargs.items():
            if not isinstance(value, dict):
                raise ValueError(
                    f'argument "{name}" is not a dict')
            if value.keys() != utterances.by_name().keys():
                raise ValueError(
                    f'utterances and "{name}" have different names')

        def _process_one(utterance, **kwargs):
            return utterance.name, self.process(
                utterance.load_audio(),
                **{k: v[utterance.name] for k, v in kwargs.items()})

        verbose = 8 if self.log.getEffectiveLevel() > 10 else 0

        return FeaturesCollection(joblib.Parallel(
            n_jobs=njobs, verbose=verbose, prefer='threads')(
                joblib.delayed(_process_one)(utt, **kwargs)
                for utt in utterances))


class FramesProcessor(FeaturesProcessor, metaclass=abc.ABCMeta):
    """A base class for frame based features processors.

    Wrap the kaldi frames implementation. See [kaldi-frame]_.

    References
    ----------

    .. [kaldi-frame]
       http://kaldi-asr.org/doc/structkaldi_1_1FrameExtractionOptions.html

    """
    def __init__(self, sample_rate=16000, frame_shift=0.01,
                 frame_length=0.025, dither=1.0, preemph_coeff=0.97,
                 remove_dc_offset=True, window_type='povey',
                 round_to_power_of_two=True, blackman_coeff=0.42,
                 snip_edges=True):
        super().__init__()

        # frame extraction options
        self._frame_options = kaldi.feat.window.FrameExtractionOptions()
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

    @property
    def sample_rate(self):
        """Waveform sample frequency in Hertz

        Must match the sample rate of the signal specified in
        `process`

        """
        return np.float32(self._frame_options.samp_freq)

    @sample_rate.setter
    def sample_rate(self, value):
        self._frame_options.samp_freq = value

    @property
    def frame_shift(self):
        """Frame shift in seconds"""
        return np.float32(self._frame_options.frame_shift_ms / 1000.0)

    @frame_shift.setter
    def frame_shift(self, value):
        self._frame_options.frame_shift_ms = value * 1000.0

    @property
    def frame_length(self):
        """Frame length in seconds"""
        return np.float32(self._frame_options.frame_length_ms / 1000.0)

    @frame_length.setter
    def frame_length(self, value):
        self._frame_options.frame_length_ms = value * 1000.0

    @property
    def dither(self):
        """Amount of dithering

        0.0 means no dither

        """
        return np.float32(self._frame_options.dither)

    @dither.setter
    def dither(self, value):
        self._frame_options.dither = value

    @property
    def preemph_coeff(self):
        """Coefficient for use in signal preemphasis"""
        return np.float32(self._frame_options.preemph_coeff)

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
        """Constant coefficient for generalized Blackman window

        Used only if `window_type` is 'blackman'

        """
        return np.float32(self._frame_options.blackman_coeff)

    @blackman_coeff.setter
    def blackman_coeff(self, value):
        self._frame_options.blackman_coeff = value

    @property
    def snip_edges(self):
        """If true, output only frames that completely fit in the file

        When True the number of frames depends on the `frame_length`.
        If False, the number of frames depends only on the
        `frame_shift`, and we reflect the data at the ends.

        """
        return self._frame_options.snip_edges

    @snip_edges.setter
    def snip_edges(self, value):
        self._frame_options.snip_edges = value

    def times(self, nframes):
        """Returns the times label for the rows given by :func:`process`"""
        return np.vstack((
            np.arange(nframes) * self.frame_shift,
            np.arange(nframes) * self.frame_shift + self.frame_length)).T


class MelFeaturesProcessor(FramesProcessor):
    """A base class for mel-based features processors

    The mel-based features are MFCC, PLP and filterbanks. The class
    implement common options for processing those features. See
    [kaldi-mel]_ and [kaldi-frame-2]_.

    References
    ----------

    .. [kaldi-frame-2]
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
        # init of FramesProcessor parent
        super().__init__(
            sample_rate=sample_rate,
            frame_shift=frame_shift,
            frame_length=frame_length,
            dither=dither,
            preemph_coeff=preemph_coeff,
            remove_dc_offset=remove_dc_offset,
            window_type=window_type,
            round_to_power_of_two=round_to_power_of_two,
            blackman_coeff=blackman_coeff,
            snip_edges=snip_edges)

        # mel banks options
        self._mel_options = kaldi.feat.mel.MelBanksOptions()
        self.num_bins = num_bins
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.vtln_low = vtln_low
        self.vtln_high = vtln_high

    @property
    def num_bins(self):
        """Number of triangular mel-frequency bins

        The minimal number of bins is 3

        """
        return self._mel_options.num_bins

    @num_bins.setter
    def num_bins(self, value):
        self._mel_options.num_bins = value

    @property
    def low_freq(self):
        """Low cutoff frequency for mel bins in Hertz"""
        return np.float32(self._mel_options.low_freq)

    @low_freq.setter
    def low_freq(self, value):
        self._mel_options.low_freq = value

    @property
    def high_freq(self):
        """High cutoff frequency for mel bins in Hertz

        If `high_freq` < 0, offset from the Nyquist frequency

        """
        return np.float32(self._mel_options.high_freq)

    @high_freq.setter
    def high_freq(self, value):
        self._mel_options.high_freq = value

    @property
    def vtln_low(self):
        """Low inflection point in piecewise linear VTLN warping function

        In Hertz

        """
        return np.float32(self._mel_options.vtln_low)

    @vtln_low.setter
    def vtln_low(self, value):
        self._mel_options.vtln_low = value

    @property
    def vtln_high(self):
        """High inflection point in piecewise linear VTLN warping function

        In Hertz. If `vtln_high` < 0, offset from `high_freq`

        """
        return np.float32(self._mel_options.vtln_high)

    @vtln_high.setter
    def vtln_high(self, value):
        self._mel_options.vtln_high = value

    def process(self, signal, vtln_warp=1.0):
        """Compute features with the specified options

        Do an optional feature-level vocal tract length normalization
        (VTLN) when `vtln_warp` != 1.0.

        Parameters
        ----------
        signal : Audio, shape = [nsamples, 1]
            The input audio signal to compute the features on, must be
            mono
        vtln_warp : float, optional
            The VTLN warping factor to be applied when computing
            features. Be 1.0 by default, meaning no warping is to be
            done.

        Returns
        -------
        features : `Features`, shape = [nframes, `ndims`]
            The computed features, output will have as many rows as there
            are frames (depends on the specified options `frame_shift`
            and `frame_length`).

        Raises
        ------
        ValueError
            If the input `signal` has more than one channel (i.e. is
            not mono). If `sample_rate` != `signal.sample_rate`.

        """
        return self._process(self._kaldi_processor, signal, vtln_warp)

    def _process(self, cls, signal, vtln_warp):
        """Inner process method common to all Kaldi Mel processors"""
        # ensure the signal is correct
        if signal.nchannels != 1:
            raise ValueError(
                'signal must have one dimension, but it has {}'
                .format(signal.nchannels))

        if self.sample_rate != signal.sample_rate:
            raise ValueError(
                'processor and signal mismatch in sample rates: '
                '{} != {}'.format(self.sample_rate, signal.sample_rate))

        # we need to forward options (because the assignation here is
        # done by copy, not by reference. If the user do 'p =
        # Processor(); p.dither = 0', this is forwarded to Kaldi here)
        self._options.frame_opts = self._frame_options
        self._options.mel_opts = self._mel_options

        # force 16 bits integers
        signal = signal.astype(np.int16).data
        data = kaldi.matrix.SubMatrix(
            cls(self._options).compute(
                kaldi.matrix.SubVector(signal), vtln_warp)).numpy()

        return Features(
            data,
            self.times(data.shape[0]),
            properties=self.get_properties(vtln_warp=vtln_warp))
