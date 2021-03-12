"""Provides classes to extract pitch from an audio (speech) signal
using the CREPE model (see [Kim2018]_). Integrates the CREPE package
(see [crepe-repo]_) into shennong API and provides postprocessing
to turn the raw pitch into usable features, using
:class:`~shennong.processor.pitch.PitchPostProcessor`.

The maximum value of the output of the neural network is
used as a heuristic estimate of the voicing probability (POV).

Examples
--------

>>> from shennong.audio import Audio
>>> from shennong.processor.crepepitch import (
...     CrepePitchProcessor, CrepePitchPostProcessor)
>>> audio = Audio.load('./test/data/test.wav')

Initialize a pitch processor with some options. Options can be
specified at construction, or after:

>>> processor = CrepePitchProcessor(
...   model_capacity='tiny', frame_shift=0.01)

Compute the pitch with the specified options, the output is an
instance of :class:`~shennong.features.Features`:

>>> pitch = processor.process(audio)
>>> type(pitch)
<class 'shennong.features.Features'>
>>> pitch.shape
(140, 2)

The pitch post-processor works in the same way, input is the pitch,
output are features usable by speech processing tools:

>>> postprocessor = CrepePitchPostProcessor()  # use default options
>>> postpitch = postprocessor.process(pitch)
>>> postpitch.shape
(140, 3)

References
----------

.. [Kim2018]
    CREPE: A Convolutional Representation for Pitch Estimation
    Jong Wook Kim, Justin Salamon, Peter Li, Juan Pablo Bello.
    Proceedings of the IEEE International Conference on Acoustics, Speech,
    and Signal Processing (ICASSP), 2018. https://arxiv.org/abs/1802.06182

.. [crepe-repo]
    https://github.com/marl/crepe

"""

import copy
import functools
import logging
import os
import warnings

import crepe
import hmmlearn
import numpy as np
import scipy.optimize
import scipy.interpolate
import scipy.signal

from shennong import Features
from shennong.processor.base import FeaturesProcessor
from shennong.processor.pitch import PitchPostProcessor


def _nccf_to_pov(x):
    y = (
        -5.2 + 5.4 * np.exp(7.5 * (x - 1)) + 4.8 * x - 2 *
        np.exp(-10 * x) + 4.2 * np.exp(20 * (x - 1)))
    return 1 / (1 + np.exp(-y))


def predict_voicing(confidence):
    """Find the Viterbi path for voiced versus unvoiced frames.

    Adapted from https://github.com/sannawag/crepe.

    Parameters
    ----------
    confidence : np.ndarray [shape=(N,)]
        voicing confidence array, i.e. the confidence in the presence of
        a pitch

    Returns
    -------
    voicing_states : np.ndarray [shape=(N,)]
        HMM predictions for each frames state, 0 if unvoiced, 1 if
        voiced
    """
    # fix the model parameters because we are not optimizing the model
    model = hmmlearn.hmm.GaussianHMM(n_components=2)

    # uniform prior on the voicing confidence
    model.startprob_ = np.array([0.5, 0.5])

    #  mean and variance for unvoiced and voiced states
    model.means_ = np.array([[0.0], [1.0]])
    model.covars_ = np.array([[0.25], [0.25]])

    # transition probabilities inducing continuous voicing state
    model.transmat_ = np.array([[0.99, 0.01], [0.01, 0.99]])

    model.n_features = 1

    # find the Viterbi path
    return np.array(
        model.predict(confidence.reshape(-1, 1), [len(confidence)]))


class CrepePitchProcessor(FeaturesProcessor):
    """Extracts the (POV, pitch) per frame from a speech signal

    This processor uses the pre-trained CREPE model.

    The output will have as many rows as there are frames, and two
    columns corresponding to (POV, pitch). POV is the Probability of
    Voicing.

    """
    def __init__(self, model_capacity='full', viterbi=True, center=True,
                 frame_shift=0.01, frame_length=0.025):
        super().__init__()

        self.model_capacity = model_capacity
        self.viterbi = viterbi
        self.center = center
        self.frame_shift = frame_shift
        self.frame_length = frame_length

    @property
    def name(self):
        return 'crepe'

    @property
    def model_capacity(self):
        """String specifying the model capacity to use

        Must be 'tiny', 'small', 'medium', 'large' or 'full'

        """
        return self._model_capacity

    @model_capacity.setter
    def model_capacity(self, value):
        if value not in ['tiny', 'small', 'medium', 'large', 'full']:
            raise ValueError(f'Model capacity {value} is not recognized.')
        self._model_capacity = value

    @property
    def viterbi(self):
        """Whether to apply viterbi smoothing to the estimated pitch curve"""
        return self._viterbi

    @viterbi.setter
    def viterbi(self, value):
        self._viterbi = bool(value)

    @property
    def center(self):
        """Whether to center the window on the current frame"""
        return self._center

    @center.setter
    def center(self, value):
        self._center = bool(value)

    @property
    def frame_shift(self):
        """"Frame shift in seconds for running pitch estimation"""
        return self._frame_shift

    @frame_shift.setter
    def frame_shift(self, value):
        self._frame_shift = value

    @property
    def frame_length(self):
        """Frame length in seconds"""
        return self._frame_length

    @frame_length.setter
    def frame_length(self, value):
        self._frame_length = value

    @property
    def sample_rate(self):
        """CREPE operates at 16kHz"""
        return 16000

    @property
    def ndims(self):
        return 2

    def times(self, nframes):
        """Returns the time label for the rows given by :func:`process`"""
        return np.vstack((
            np.arange(nframes) * self.frame_shift,
            np.arange(nframes) * self.frame_shift + self.frame_length)).T

    def process(self, audio):
        """Extracts the (POV, pitch) from a given speech ``audio`` using CREPE.

        Parameters
        ----------
        audio : Audio
            The speech signal on which to estimate the pitch. Will be
            transparently resampled at 16kHz if needed.

        Returns
        -------
        raw_pitch_features : Features, shape = [nframes, 2]
            The output array has two columns corresponding to (POV,
            pitch). The output from the `crepe` module is reshaped to
            match the specified options `frame_shift` and `frame_length`.

        Raises
        ------
        ValueError
            If the input `signal` has more than one channel (i.e. is
            not mono).

        """
        if audio.nchannels != 1:
            raise ValueError(
                f'audio must have one channel but has {audio.nchannels}')

        if audio.sample_rate != self.sample_rate:
            self.log.debug('resampling audio to 16 kHz')
            audio = audio.resample(self.sample_rate)

        # tensorflow verbosity
        if self.log.level == logging.DEBUG:  # pragma: nocover
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
            verbose = 2
        else:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            verbose = 0

        with warnings.catch_warnings():
            # tensorflow (used by crepe) issues irrelevant warnings, we just
            # ignore them
            warnings.simplefilter('ignore')

            _, frequency, confidence, _ = crepe.predict(
                audio.data,
                audio.sample_rate,
                model_capacity=self.model_capacity,
                viterbi=self.viterbi,
                center=self.center,
                step_size=int(self.frame_shift * 1000),
                verbose=verbose)

        # number of samples in the resampled signal
        hop_length = np.round(self.sample_rate * self.frame_shift).astype(int)
        nsamples = 1 + int((
            audio.shape[0] - self.frame_length * self.sample_rate)
                           / hop_length)

        # scipy method issues warnings we want to inhibit
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=FutureWarning)
            data = scipy.signal.resample(
                np.array([confidence, frequency]).T, nsamples)

        # hack needed beacause resample confidence
        data[data[:, 0] < 1e-2, 0] = 0
        data[data[:, 0] > 1, 0] = 1

        return Features(
            data, self.times(data.shape[0]), properties=self.get_properties())


class CrepePitchPostProcessor(PitchPostProcessor):
    """Processes the raw (POV, pitch) computed by the CrepePitchProcessor

    Turns the raw pitch quantities into usable features. Converts the POV into
    NCCF usable by :class:`PitchPostProcessor`, then removes the pitch at
    frames with the worst POV (according to the `pov_threshold` or the
    `proportion_voiced` option) and replace them with interpolated values, and
    finally sends this (NCCF, pitch) pair to
    :func:`shennong.processor.pitch.PitchPostProcessor.process`.

    """
    def __init__(self, pitch_scale=2.0, delta_pitch_scale=10.0,
                 delta_pitch_noise_stddev=0.005,
                 normalization_left_context=75, normalization_right_context=75,
                 delta_window=2, delay=0,
                 add_pov_feature=True, add_normalized_log_pitch=True,
                 add_delta_pitch=True, add_raw_log_pitch=False):
        super().__init__(
            pitch_scale=pitch_scale,
            delta_pitch_scale=delta_pitch_scale,
            delta_pitch_noise_stddev=delta_pitch_noise_stddev,
            normalization_left_context=normalization_left_context,
            normalization_right_context=normalization_right_context,
            delta_window=delta_window,
            delay=delay,
            add_pov_feature=add_pov_feature,
            add_normalized_log_pitch=add_normalized_log_pitch,
            add_delta_pitch=add_delta_pitch,
            add_raw_log_pitch=add_raw_log_pitch)

    @property
    def name(self):
        return 'crepe postprocessing'

    def get_properties(self, features):
        properties = copy.deepcopy(features.properties)
        properties['crepe'][self.name] = self.get_params()
        properties['pipeline'][0]['columns'] = [0, self.ndims - 1]
        return properties

    def process(self, crepe_pitch):
        """Post process a raw pitch data as specified by the options

        Parameters
        ----------
        crepe_pitch : Features, shape = [n, 2]
            The pitch as extracted by the `CrepePitchProcessor.process`
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
            If after interpolation some pitch values are not positive.
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

        if crepe_pitch.shape[1] != 2:
            raise ValueError(
                'data shape must be (_, 2), but it is (_, {})'
                .format(crepe_pitch.shape[1]))

        # Interpolate pitch values for unvoiced frames
        to_remove = predict_voicing(crepe_pitch.data[:, 0]) == 0
        if np.all(to_remove):
            raise ValueError('No voiced frames')

        data = crepe_pitch.data[:, 1].copy()
        indexes_to_keep = np.where(~to_remove)[0]
        first, last = indexes_to_keep[0], indexes_to_keep[-1]
        first_value, last_value = data[first], data[last]

        interp = scipy.interpolate.interp1d(
            indexes_to_keep, data[indexes_to_keep],
            fill_value='extrapolate')
        data[to_remove] = interp(np.where(to_remove)[0])
        data[:first] = first_value
        data[last:] = last_value

        if not np.all(data > 0):
            raise ValueError(
                'Not all pitch values are positive: issue with '
                'extracted pitch or interpolation')

        # Converts POV into NCCF
        nccf = []
        for sample in crepe_pitch.data[:, 0]:
            if sample in [0, 1]:
                nccf.append(sample)
            else:
                nccf.append(scipy.optimize.bisect(functools.partial(
                    lambda x, y: _nccf_to_pov(x)-y, y=sample), 0, 1))

        return super(CrepePitchPostProcessor, self).process(
            Features(np.vstack((nccf, data)).T,
                     crepe_pitch.times,
                     crepe_pitch.properties))
