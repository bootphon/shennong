"""Extract MFCC features from an audio (speech) signal"""

import numpy as np

from kaldi.feat import mfcc
from kaldi.matrix import SubVector, SubMatrix

from shennong.features.features import Features
from shennong.features.processor import MelFeaturesProcessor


class MfccProcessor(MelFeaturesProcessor):
    def __init__(self, sample_rate=16000, frame_shift=0.01,
                 frame_length=0.025, dither=1.0, preemph_coeff=0.97,
                 remove_dc_offset=True, window_type='povey',
                 round_to_power_of_two=True, blackman_coeff=0.42,
                 snip_edges=True, num_bins=23, low_freq=20,
                 high_freq=0, vtln_low=100, vtln_high=-500,
                 num_ceps=13, use_energy=True, energy_floor=0.0,
                 raw_energy=True, cepstral_lifter=22.0,
                 htk_compat=False):
        # Forward options to MelFeaturesProcessor
        super().__init__(sample_rate, frame_shift, frame_length,
                         dither, preemph_coeff, remove_dc_offset, window_type,
                         round_to_power_of_two, blackman_coeff, snip_edges,
                         num_bins, low_freq, high_freq, vtln_low, vtln_high)

        self._options = mfcc.MfccOptions()
        self._options.frame_opts = self._frame_options
        self._options.mel_opts = self._mel_options

        self.num_ceps = num_ceps
        self.use_energy = use_energy
        self.energy_floor = energy_floor
        self.raw_energy = raw_energy
        self.cepstral_lifter = cepstral_lifter
        self.htk_compat = htk_compat

    @property
    def num_ceps(self):
        """Number of cepstra in MFCC computation (including C0)"""
        return self._options.num_ceps

    @num_ceps.setter
    def num_ceps(self, value):
        self._options.num_ceps = value

    @property
    def use_energy(self):
        """Use energy (instead of C0) in MFCC computation"""
        return self._options.use_energy

    @use_energy.setter
    def use_energy(self, value):
        self._options.use_energy = value

    @property
    def energy_floor(self):
        """Floor on energy (absolute, not relative) in MFCC computation"""
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
    def cepstral_lifter(self):
        """Constant that controls scaling of MFCCs"""
        return self._options.cepstral_lifter

    @cepstral_lifter.setter
    def cepstral_lifter(self, value):
        self._options.cepstral_lifter = value

    @property
    def htk_compat(self):
        """If True, get closer to HTK MFCC features

        Put energy or C0 last and use a factor of sqrt(2) on C0.
        Warning: not sufficient to get HTK compatible features (need
        to change other parameters)

        """
        return self._options.htk_compat

    @htk_compat.setter
    def htk_compat(self, value):
        self._options.htk_compat = value

    def parameters(self):
        params = super().parameters()
        params.update({
            'num_ceps': self.num_ceps,
            'use_energy': self.use_energy,
            'energy_floor': self.energy_floor,
            'raw_energy': self.raw_energy,
            'cepstral_lifter': self.cepstral_lifter,
            'htk_compat': self.htk_compat})
        return params

    def times(self, nframes):
        """Returns the time label for the rows given by the `process` method"""
        return np.arange(nframes) * self.frame_shift + self.frame_length / 2.0

    def process(self, signal):
        if signal.ndim != 1:
            raise ValueError(
                'signal must have one dimension, but it has {}'
                .format(signal.ndim))
        processor = mfcc.Mfcc(self._options)
        vtln_warp = 1.0  # TODO fixme
        data = processor.compute_features(
            SubVector(signal), self.sample_rate, vtln_warp)

        return Features(
            data, self.labels(), self.times(data.shape[0]), self.parameters())
