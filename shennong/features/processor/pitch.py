"""Provides classes to extract pitch from an audio (speech) signal

This modules provides the classes PitchProcessor and
PitchPostProcessor which respectively computes the pitch from raw
speech and turns it into suitable features: it produces pitch and
probability-of-voicing estimates for use as features in automatic
speech recognition systems

Uses the Kaldi implementation of pitch extraction and postprocessing
(see [Ghahremani2014]_ and [kaldi-pitch]_).

    :class:`AudioData` ---> PitchProcessor ---> PitchPostProcessor \
    ---> :class:`Features`

Examples
--------

>>> from shennong.audio import AudioData
>>> from shennong.features.processor.pitch import (
...     PitchProcessor, PitchPostProcessor)
>>> audio = AudioData.load('./test/data/test.wav')

Initialize a pitch processor with some options. Options can be
specified at construction, or after:

>>> processor = PitchProcessor(frame_shift=0.01, frame_length=0.025)
>>> processor.sample_rate = audio.sample_rate
>>> processor.min_f0 = 20
>>> processor.max_f0 = 500

Options can also being passed as a dictionnary:

>>> options = {
...     'sample_rate': audio.sample_rate,
...     'frame_shift': 0.01, 'frame_length': 0.025,
...     'min_f0': 20, 'max_f0': 500}
>>> processor = PitchProcessor(**options)

Compute the pitch with the specified options, the output is an
instance of `Features`:

>>> pitch = processor.process(audio)
>>> type(pitch)
<class 'shennong.features.features.Features'>
>>> pitch.shape
(140, 2)

The pitch post-processor works in the same way, input is the pitch,
output are features usable by speech processing tools:

>>> postprocessor = PitchPostProcessor()  # use default options
>>> postpitch = postprocessor.process(pitch)
>>> postpitch.shape
(140, 3)

References
----------

.. [Ghahremani2014] `A Pitch Extraction Algorithm Tuned for Automatic
     Speech Recognition, Pegah Ghahremani, Bagher BabaAli, Daniel
     Povey, Korbinian Riedhammer, Jan Trmal and Sanjeev Khudanpur,
     ICASSP 2014`

.. [kaldi-pitch] http://kaldi-asr.org/doc/pitch-functions_8h.html

"""

import kaldi.feat.pitch
import kaldi.matrix
import numpy as np

from shennong.features import Features
from shennong.features.processor.base import FeaturesProcessor


class PitchProcessor(FeaturesProcessor):
    """Extracts the (NCCF, pitch) per frame from a speech signal

    The output will have as many rows as there are frames, and two
    columns corresponding to (NCCF, pitch). NCCF is the Normalized
    Cross Correlation Function.

    """
    def __init__(self, sample_rate=16000, frame_shift=0.01,
                 frame_length=0.025, min_f0=50, max_f0=400,
                 soft_min_f0=10, penalty_factor=0.1,
                 lowpass_cutoff=1000, resample_freq=4000,
                 delta_pitch=0.005, nccf_ballast=7000,
                 lowpass_filter_width=1, upsample_filter_width=5):
        self._options = kaldi.feat.pitch.PitchExtractionOptions()
        self.sample_rate = sample_rate
        self.frame_shift = frame_shift
        self.frame_length = frame_length
        self.min_f0 = min_f0
        self.max_f0 = max_f0
        self.soft_min_f0 = soft_min_f0
        self.penalty_factor = penalty_factor
        self.lowpass_cutoff = lowpass_cutoff
        self.resample_freq = resample_freq
        self.delta_pitch = delta_pitch
        self.nccf_ballast = nccf_ballast
        self.lowpass_filter_width = lowpass_filter_width
        self.upsample_filter_width = upsample_filter_width

    @property
    def sample_rate(self):
        """Waveform sample frequency in Hertz

        Must match the sample rate of the signal specified in `process`

        """
        return self._options.samp_freq

    @sample_rate.setter
    def sample_rate(self, value):
        self._options.samp_freq = value

    @property
    def frame_shift(self):
        """Frame shift in seconds"""
        return self._options.frame_shift_ms / 1000.0

    @frame_shift.setter
    def frame_shift(self, value):
        self._options.frame_shift_ms = value * 1000.0

    @property
    def frame_length(self):
        """Frame length in seconds"""
        return self._options.frame_length_ms / 1000.0

    @frame_length.setter
    def frame_length(self, value):
        self._options.frame_length_ms = value * 1000.0

    @property
    def min_f0(self):
        """Minimum F0 to search for in Hertz"""
        return self._options.min_f0

    @min_f0.setter
    def min_f0(self, value):
        self._options.min_f0 = value

    @property
    def max_f0(self):
        """Maximum F0 to search for in Hertz"""
        return self._options.max_f0

    @max_f0.setter
    def max_f0(self, value):
        self._options.max_f0 = value

    @property
    def soft_min_f0(self):
        """Minimum F0 to search, applied in soft way, in Hertz

        Must not exceed `min_f0`

        """
        return self._options.soft_min_f0

    @soft_min_f0.setter
    def soft_min_f0(self, value):
        self._options.soft_min_f0 = value

    @property
    def penalty_factor(self):
        """Cost factor for F0 change"""
        return np.float32(self._options.penalty_factor)

    @penalty_factor.setter
    def penalty_factor(self, value):
        self._options.penalty_factor = value

    @property
    def lowpass_cutoff(self):
        """Cutoff frequency for low-pass filter, in Hertz"""
        return self._options.lowpass_cutoff

    @lowpass_cutoff.setter
    def lowpass_cutoff(self, value):
        self._options.lowpass_cutoff = value

    @property
    def resample_freq(self):
        """Frequency that we down-sample the signal to, in Hertz

        Must be more than twice `lowpass_cutoff`

        """
        return self._options.resample_freq

    @resample_freq.setter
    def resample_freq(self, value):
        self._options.resample_freq = value

    @property
    def delta_pitch(self):
        """Smallest relative change in pitch that the algorithm measures"""
        return np.float32(self._options.delta_pitch)

    @delta_pitch.setter
    def delta_pitch(self, value):
        self._options.delta_pitch = value

    @property
    def nccf_ballast(self):
        """Increasing this factor reduces NCCF for quiet frames

        This helps ensuring pitch continuity in unvoiced regions

        """
        return self._options.nccf_ballast

    @nccf_ballast.setter
    def nccf_ballast(self, value):
        self._options.nccf_ballast = value

    @property
    def lowpass_filter_width(self):
        """Integer that determines filter width of lowpass filter

        More gives sharper filter

        """
        return self._options.lowpass_filter_width

    @lowpass_filter_width.setter
    def lowpass_filter_width(self, value):
        self._options.lowpass_filter_width = value

    @property
    def upsample_filter_width(self):
        """Integer that determines filter width when upsampling NCCF"""
        return self._options.upsample_filter_width

    @upsample_filter_width.setter
    def upsample_filter_width(self, value):
        self._options.upsample_filter_width = value

    def times(self, nframes):
        """Returns the time label for the rows given by the `process` method"""
        return np.arange(nframes) * self.frame_shift + self.frame_length / 2.0

    def process(self, signal):
        """Extracts the (NCCF, pitch) from a given speech `signal`

        Parameters
        ----------
        signal : AudioData
            The speech signal on which to estimate the pitch. The
            signal's sample rate must match the sample rate specified
            in the `PitchProcessor` options.

        Returns
        -------
        raw_pitch_features : Features, shape = [nframes, 2]
            The output array has as many rows as there are frames
            (depends on the specified options `frame_shift` and
            `frame_length`), and two columns corresponding to (NCCF,
            pitch).

        Raises
        ------
        ValueError
            If the input `signal` has more than one channel (i.e. is
            not mono). If `sample_rate` != `signal.sample_rate`.

        """
        if signal.nchannels != 1:
            raise ValueError(
                'audio signal must have one channel, but it has {}'
                .format(signal.nchannels))

        if self.sample_rate != signal.sample_rate:
            raise ValueError(
                'processor and signal mismatch in sample rates: '
                '{} != {}'.format(self.sample_rate, signal.sample_rate))

        # force 16 bits integers
        signal = signal.astype(np.int16).data
        data = kaldi.matrix.SubMatrix(
            kaldi.feat.pitch.compute_kaldi_pitch(
                self._options, kaldi.matrix.SubVector(signal))).numpy()

        return Features(
            data, self.times(data.shape[0]), properties=self.get_params())


class PitchPostProcessor(FeaturesProcessor):
    """Processes the raw (NCCF, pitch) computed by the PitchProcessor

    Turns the raw pitch quantites into usable features. By default it
    will output three-dimensional features, (POV-feature,
    mean-subtracted-log-pitch, delta-of-raw-pitch), but this is
    configurable in the options. The number of rows of "output" will
    be the number of frames (rows) in "input", i.e. the number of
    frames. The number of columns will be the number of different
    types of features requested (by default, 3; 4 is the max). The
    four parameters `add_pov_feature`, `add_normalized_log_pitch`,
    `add_delta_pitch`, `add_raw_log_pitch` determine which features we
    create; by default we create the first three.

    POV stands for Probability of Voicing.

    """
    def __init__(self, pitch_scale=2.0, pov_scale=2.0, pov_offset=0.0,
                 delta_pitch_scale=10.0, delta_pitch_noise_stddev=0.005,
                 normalization_left_context=75,
                 normalization_right_context=75,
                 delta_window=2, delay=0,
                 add_pov_feature=True, add_normalized_log_pitch=True,
                 add_delta_pitch=True, add_raw_log_pitch=False):
        self._options = kaldi.feat.pitch.ProcessPitchOptions()
        self.pitch_scale = pitch_scale
        self.pov_scale = pov_scale
        self.pov_offset = pov_offset
        self.delta_pitch_scale = delta_pitch_scale
        self.delta_pitch_noise_stddev = delta_pitch_noise_stddev
        self.normalization_left_context = normalization_left_context
        self.normalization_right_context = normalization_right_context
        self.delta_window = delta_window
        self.delay = delay
        self.add_pov_feature = add_pov_feature
        self.add_normalized_log_pitch = add_normalized_log_pitch
        self.add_delta_pitch = add_delta_pitch
        self.add_raw_log_pitch = add_raw_log_pitch

    @property
    def pitch_scale(self):
        """Scaling factor for the final normalized log-pitch value"""
        return self._options.pitch_scale

    @pitch_scale.setter
    def pitch_scale(self, value):
        self._options.pitch_scale = value

    @property
    def pov_scale(self):
        """Scaling factor for final probability of voicing feature"""
        return self._options.pov_scale

    @pov_scale.setter
    def pov_scale(self, value):
        self._options.pov_scale = value

    @property
    def pov_offset(self):
        """This can be used to add an offset to the POV feature

        Intended for use in Kaldi's online decoding as a substitute
        for CMV (cepstral mean normalization)

        """
        return self._options.pov_offset

    @pov_offset.setter
    def pov_offset(self, value):
        self._options.pov_offset = value

    @property
    def delta_pitch_scale(self):
        """Term to scale the final delta log-pitch feature"""
        return self._options.delta_pitch_scale

    @delta_pitch_scale.setter
    def delta_pitch_scale(self, value):
        self._options.delta_pitch_scale = value

    @property
    def delta_pitch_noise_stddev(self):
        """Standard deviation for noise we add to the delta log-pitch

        The stddev is added before scaling. Should be about the same
        as delta-pitch option to pitch creation. The purpose is to get
        rid of peaks in the delta-pitch caused by discretization of
        pitch values.

        """
        return np.float32(self._options.delta_pitch_noise_stddev)

    @delta_pitch_noise_stddev.setter
    def delta_pitch_noise_stddev(self, value):
        self._options.delta_pitch_noise_stddev = value

    @property
    def normalization_left_context(self):
        """Left-context (in frames) for moving window normalization"""
        return self._options.normalization_left_context

    @normalization_left_context.setter
    def normalization_left_context(self, value):
        self._options.normalization_left_context = value

    @property
    def normalization_right_context(self):
        """Right-context (in frames) for moving window normalization"""
        return self._options.normalization_right_context

    @normalization_right_context.setter
    def normalization_right_context(self, value):
        self._options.normalization_right_context = value

    @property
    def delta_window(self):
        """Number of frames on each side of central frame"""
        return self._options.delta_window

    @delta_window.setter
    def delta_window(self, value):
        self._options.delta_window = value

    @property
    def delay(self):
        """Number of frames by which the pitch information is delayed"""
        return self._options.delay

    @delay.setter
    def delay(self, value):
        self._options.delay = value

    @property
    def add_pov_feature(self):
        """If true, the warped NCCF is added to output features"""
        return self._options.add_pov_feature

    @add_pov_feature.setter
    def add_pov_feature(self, value):
        self._options.add_pov_feature = value

    @property
    def add_normalized_log_pitch(self):
        """If true, the normalized log-pitch is added to output features

         Normalization is done with POV-weighted mean subtraction over
         1.5 second window.

        """
        return self._options.add_normalized_log_pitch

    @add_normalized_log_pitch.setter
    def add_normalized_log_pitch(self, value):
        self._options.add_normalized_log_pitch = value

    @property
    def add_delta_pitch(self):
        """If true, time derivative of log-pitch is added to output features"""
        return self._options.add_delta_pitch

    @add_delta_pitch.setter
    def add_delta_pitch(self, value):
        self._options.add_delta_pitch = value

    @property
    def add_raw_log_pitch(self):
        """If true, time derivative of log-pitch is added to output features"""
        return self._options.add_raw_log_pitch

    @add_raw_log_pitch.setter
    def add_raw_log_pitch(self, value):
        self._options.add_raw_log_pitch = value

    def process(self, raw_pitch):
        """Post process a raw pitch data as specified by the options

        Parameters
        ----------
        raw_pitch : Features, shape = [n, 2]
            The pitch as extracted by the `PitchProcessor.process`
            method

        Returns
        -------
        pitch : Features, shape = [n, 1 2 3 or 4]
            The post-processed pitch usable as speech features. The
            output columns are 'pov_feature', 'normalized_log_pitch',
            delta_pitch' and 'raw_log_pitch', in that order,if their
            respective options are set to True.

        Raises
        ------
        ValueError
            If `raw_pitch` has not exactly two columns. If all the
            following options are False: 'add_pov_feature',
            'add_normalized_log_pitch', 'add_delta_pitch' and
            'add_raw_log_pitch' (at least one of them must be True).

        """
        # check at least one required option is True
        if not (self.add_pov_feature or self.add_normalized_log_pitch
                or self.add_delta_pitch or self.add_raw_log_pitch):
            raise ValueError(
                'at least one of the following options must be True: '
                'add_pov_feature, add_normalized_log_pitch, '
                'add_delta_pitch, add_raw_log_pitch')

        if raw_pitch.shape[1] != 2:
            raise ValueError(
                'data shape must be (_, 2), but it is (_, {})'
                .format(raw_pitch.shape[1]))

        data = kaldi.matrix.SubMatrix(
            kaldi.feat.pitch.process_pitch(
                self._options, kaldi.matrix.SubMatrix(raw_pitch.data))).numpy()

        return Features(
            data, raw_pitch.times, properties=self.get_params())
