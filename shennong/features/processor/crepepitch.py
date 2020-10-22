"""Provides classes to extract pitch from an audio (speech) signal
using the CREPE model (see [Kim2018]_). Integrates the CREPE package
(see [crepe-repo]_) into shennong API and provides postprocessing
to turn the raw pitch into usable features, using
:class:`~shennong.features.processor.pitch.PitchPostProcessor`.

The maximum value of the output of the neural network is
used as a heuristic estimate of the voicing probability (POV).


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
import crepe
import functools
import numpy as np
import scipy.optimize
import scipy.interpolate

from shennong.features import Features
from shennong.features.processor.base import FeaturesProcessor
from shennong.features.processor.pitch import PitchPostProcessor


def _nccf_to_pov(x):
    y = -5.2 + 5.4*np.exp(7.5*(x - 1)) + 4.8*x - 2 * \
        np.exp(-10*x)+4.2*np.exp(20*(x-1))
    return 1/(1+np.exp(-y))


class CrepePitchProcessor(FeaturesProcessor):
    """Extracts the (POV, pitch) per frame from a speech signal
    using the CREPE model.

    The output will have as many rows as there are frames, and two
    columns corresponding to (POV, pitch). POV is the Probability of
    Voicing.

    """
    sample_rate = 16000

    def __init__(self, model_capacity='full',
                 viterbi=True, center=True, step_size=10):
        self.model_capacity = model_capacity
        self.viterbi = viterbi
        self.center = center
        self.step_size = step_size

    @property
    def name(self):
        return 'crepe'

    @property
    def model_capacity(self):
        """String specifying the model capacity"""
        return self._model_capacity

    @model_capacity.setter
    def model_capacity(self, value):
        if value not in ['tiny', 'small', 'medium', 'large', 'full']:
            raise ValueError(f'Model capacity {value} is not recognized.')
        self._model_capacity = value

    @property
    def viterbi(self):
        """Apply viterbi smoothing to the estimated pitch curve"""
        return self._viterbi

    @viterbi.setter
    def viterbi(self, value):
        self._viterbi = bool(value)

    @property
    def center(self):
        """
        - If `True` (default), the signal `audio` is padded so that frame
          `D[:, t]` is centered at `audio[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `audio[t * hop_length]`
        """
        return self._center

    @center.setter
    def center(self, value):
        self._center = bool(value)

    @property
    def step_size(self):
        """The step size in milliseconds for running pitch estimation"""
        return self._step_size

    @step_size.setter
    def step_size(self, value):
        self._step_size = int(value)

    @property
    def ndims(self):
        return 2

    def process(self, audio):
        if audio.nchannels != 1:
            raise ValueError(
                'audio signal must have one channel, but it has {}'
                .format(audio.nchannels))

        if audio.sample_rate != self.sample_rate:
            audio = audio.resample(self.sample_rate)

        time, frequency, confidence, _ = \
            crepe.predict(audio.data,
                          audio.sample_rate,
                          model_capacity=self.model_capacity,
                          viterbi=self.viterbi,
                          center=self.center,
                          step_size=self.step_size,
                          verbose=1)

        return Features(
            np.array([confidence, frequency]).T, time,
            properties=self.get_properties())


class CrepePitchPostProcessor(PitchPostProcessor):
    """Processes the raw (POV, pitch) computed by the :class:`CrepePitchProcessor`
    using :class:`PitchPostProcessor`.

    Turns the raw pitch quantities into usable features. Converts the POV into
    NCCF usable by :class:`PitchPostProcessor`, then removes the pitch at
    frames with the worst POV (according to the `pov_threshold` or the
    `proportion_voiced` option) and replace them with interpolated values,
    and finally gives the new (NCCF, pitch) to the
    :func:`~shennong.features.processor.pitch.PitchProcessor.process`
    method of :class:`PitchPostProcessor`
    """

    def __init__(self, pov_threshold=0.5,
                 proportion_voiced=None,
                 pitch_scale=2.0,
                 delta_pitch_scale=10.0,
                 delta_pitch_noise_stddev=0.005,
                 normalization_left_context=75,
                 normalization_right_context=75,
                 delta_window=2, delay=0,
                 add_normalized_log_pitch=True,
                 add_delta_pitch=True,
                 add_raw_log_pitch=False):
        super().__init__(
            pitch_scale=pitch_scale,
            delta_pitch_scale=delta_pitch_scale,
            delta_pitch_noise_stddev=delta_pitch_noise_stddev,
            normalization_left_context=normalization_left_context,
            normalization_right_context=normalization_right_context,
            delta_window=delta_window,
            delay=delay,
            add_normalized_log_pitch=add_normalized_log_pitch,
            add_delta_pitch=add_delta_pitch,
            add_raw_log_pitch=add_raw_log_pitch)
        self.pov_threshold = pov_threshold
        self.proportion_voiced = proportion_voiced

    @property
    def name(self):
        return 'crepe postprocessing'

    @property
    def pov_threshold(self):
        """All frames with a POV below this threshold are considered
        unvoiced"""
        return self._pov_threshold

    @pov_threshold.setter
    def pov_threshold(self, value):
        if value < 0 or value >= 1:
            raise ValueError('POV threshold must be in [0, 1)')
        self._pov_threshold = value

    @property
    def proportion_voiced(self):
        """Proportion of voiced frames. Optionnal, overrides ``pov_threshold``
        if provided"""
        return self._proportion_voiced

    @proportion_voiced.setter
    def proportion_voiced(self, value):
        if value is not None:
            self._proportion_voiced = np.float32(value)
        else:
            self._proportion_voiced = None

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
        # proportion_voiced overrides pov_threshold if provided
        if self.proportion_voiced is not None:
            self.pov_threshold = np.quantile(
                crepe_pitch.data[:, 0], 1-self.proportion_voiced)

        # Converts POV into NCCF
        data = crepe_pitch.data[:, 1].copy()
        nccf = [scipy.optimize.bisect(
            functools.partial(lambda x, y: _nccf_to_pov(x)-y, y=y), 0, 1)
            for y in crepe_pitch.data[:, 0]]

        # Interpolate pitch values where pov < pov_threshold
        to_remove = crepe_pitch.data[:, 0] < self.pov_threshold
        interp = scipy.interpolate.interp1d(
            np.where(~to_remove)[0], data[~to_remove],
            fill_value='extrapolate')
        data[to_remove] = interp(np.where(to_remove)[0])

        if not np.all(data > 0):
            raise ValueError('Problem')

        return super(CrepePitchPostProcessor, self).process(
            Features(np.vstack((nccf, data)).T,
                     crepe_pitch.times,
                     crepe_pitch.properties))
