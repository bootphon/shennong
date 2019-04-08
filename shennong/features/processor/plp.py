# coding: utf-8

"""Provides the PlpProcessor class to extract PLP features

Extract PLP (Perceptual Linear Predictive analysis of speech) from an
audio signal. Uses the Kaldi implementation (see [Hermansky1990]_ and
[kaldi-plp]_).

    :class:`AudioData` ---> PlpProcessor ---> :class:`Features`

Examples
--------

>>> from shennong.audio import AudioData
>>> from shennong.features.processor.plp import PlpProcessor
>>> audio = AudioData.load('./test/data/test.wav')

Initialize the PLP processor with some options. Options can be
specified at construction, or after:

>>> processor = PlpProcessor()
>>> processor.sample_rate = audio.sample_rate
>>> processor.low_freq = 20
>>> processor.high_freq = -100  # nyquist - 100
>>> processor.compress_factor = 1/3

Compute the PLP features with the specified options, the output is an
instance of `Features`:

>>> plp = processor.process(audio)
>>> type(plp)
<class 'shennong.features.features.Features'>
>>> plp.shape[1] == processor.num_ceps
True

References
----------

.. [Hermansky1990] `Perceptual linear predictive (PLP) analysis of
     speech, H. Hermansky, Journal of the Acoustical Society of
     America, vol. 87, no. 4, pages 1738–1752 (1990)`

.. [kaldi-plp] http://kaldi-asr.org/doc/feat.html#feat_plp

"""

import kaldi.feat.plp
import kaldi.matrix
import numpy as np

from shennong.features.processor.base import MelFeaturesProcessor


class PlpProcessor(MelFeaturesProcessor):
    """Perceptive linear predictive features"""
    def __init__(self, sample_rate=16000, frame_shift=0.01,
                 frame_length=0.025, dither=1.0, preemph_coeff=0.97,
                 remove_dc_offset=True, window_type='povey',
                 round_to_power_of_two=True, blackman_coeff=0.42,
                 snip_edges=True, num_bins=23, low_freq=20,
                 high_freq=0, vtln_low=100, vtln_high=-500,
                 lpc_order=12, num_ceps=13, use_energy=True,
                 energy_floor=0.0, raw_energy=True,
                 compress_factor=1.0/3.0, cepstral_lifter=22,
                 cepstral_scale=1.0, htk_compat=False):
        # Forward options to MelFeaturesProcessor
        super().__init__(sample_rate, frame_shift, frame_length,
                         dither, preemph_coeff, remove_dc_offset, window_type,
                         round_to_power_of_two, blackman_coeff, snip_edges,
                         num_bins, low_freq, high_freq, vtln_low, vtln_high)

        self._options = kaldi.feat.plp.PlpOptions()
        self._options.frame_opts = self._frame_options
        self._options.mel_opts = self._mel_options

        self.lpc_order = lpc_order
        self.num_ceps = num_ceps
        self.use_energy = use_energy
        self.energy_floor = energy_floor
        self.raw_energy = raw_energy
        self.compress_factor = compress_factor
        self.cepstral_lifter = cepstral_lifter
        self.cepstral_scale = cepstral_scale
        self.htk_compat = htk_compat

        self._kaldi_processor = kaldi.feat.plp.Plp

    @property
    def lpc_order(self):
        """Order of LPC analysis in PLP computation"""
        return self._options.lpc_order

    @lpc_order.setter
    def lpc_order(self, value):
        self._options.lpc_order = value

    @property
    def num_ceps(self):
        """Number of cepstra in PLP computation (including C0)

        Should be smaller or equal to `lpc_order` + 1.

        """
        return self._options.num_ceps

    @num_ceps.setter
    def num_ceps(self, value):
        if int(value) > self.lpc_order + 1:
            raise ValueError(
                'We must have num_ceps <= lpc_order+1, but {} > {}+1'.format(
                    int(value), self.lpc_order))
        self._options.num_ceps = int(value)

    @property
    def use_energy(self):
        """Use energy (instead of C0) for zeroth PLP feature"""
        return self._options.use_energy

    @use_energy.setter
    def use_energy(self, value):
        self._options.use_energy = value

    @property
    def energy_floor(self):
        """Floor on energy (absolute, not relative) in PLP computation"""
        return self._options.energy_floor

    @energy_floor.setter
    def energy_floor(self, value):
        self._options.energy_floor = value

    @property
    def raw_energy(self):
        """If true, compute energy before preemphasis and windowing"""
        return self._options.raw_energy

    @raw_energy.setter
    def raw_energy(self, value):
        self._options.raw_energy = value

    @property
    def compress_factor(self):
        """Compression factor in PLP computation"""
        return np.float32(self._options.compress_factor)

    @compress_factor.setter
    def compress_factor(self, value):
        self._options.compress_factor = value

    @property
    def cepstral_lifter(self):
        """Constant that controls scaling of PLPs"""
        return self._options.cepstral_lifter

    @cepstral_lifter.setter
    def cepstral_lifter(self, value):
        self._options.cepstral_lifter = value

    @property
    def cepstral_scale(self):
        """Scaling constant in PLP computation"""
        return self._options.cepstral_scale

    @cepstral_scale.setter
    def cepstral_scale(self, value):
        self._options.cepstral_scale = value

    @property
    def htk_compat(self):
        """If True, get closer to HTK PLP features

        Put energy or C0 last.

        Warnings
        --------
        Not sufficient to get HTK compatible features (need to change
        other parameters)

        """
        return self._options.htk_compat

    @htk_compat.setter
    def htk_compat(self, value):
        self._options.htk_compat = value

    @property
    def ndims(self):
        return self.num_ceps
