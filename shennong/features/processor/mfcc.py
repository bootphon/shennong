"""Extraction of MFCC features from audio signals

Extract MFCC (Mel Frequency Cepstral Coeficients) from an audio
signal. Uses the Kaldi implementation (see [kaldi-mfcc]_):

    :class:`AudioData` ---> MfccProcessor ---> :class:`Features`

Examples
--------

>>> from shennong.audio import AudioData
>>> from shennong.features.processor.mfcc import MfccProcessor
>>> audio = AudioData.load('./test/data/test.wav')

Initialize the MFCC processor with some options. Options can be
specified at construction, or after:

>>> processor = MfccProcessor(sample_rate=audio.sample_rate)
>>> processor.window_type = 'hanning'
>>> processor.low_freq = 20
>>> processor.high_freq = -100  # nyquist - 100
>>> processor.use_energy = False  # use C0 instead

Compute the MFCC features with the specified options, the output is an
instance of `Features`:

>>> mfcc = processor.process(audio)
>>> type(mfcc)
<class 'shennong.features.features.Features'>
>>> mfcc.shape[1] == processor.num_ceps
True

References
----------

.. [kaldi-mfcc] http://kaldi-asr.org/doc/feat.html#feat_mfcc

"""

import kaldi.feat.mfcc
import kaldi.matrix

from shennong.features.processor.base import MelFeaturesProcessor


class MfccProcessor(MelFeaturesProcessor):
    """Mel Frequency Cepstral Coeficients"""
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
            snip_edges=snip_edges,
            num_bins=num_bins,
            low_freq=low_freq,
            high_freq=high_freq,
            vtln_low=vtln_low,
            vtln_high=vtln_high)

        self._options = kaldi.feat.mfcc.MfccOptions()
        self._options.frame_opts = self._frame_options
        self._options.mel_opts = self._mel_options

        self.num_ceps = num_ceps
        self.use_energy = use_energy
        self.energy_floor = energy_floor
        self.raw_energy = raw_energy
        self.cepstral_lifter = cepstral_lifter
        self.htk_compat = htk_compat

        self._kaldi_processor = kaldi.feat.mfcc.Mfcc

    @property
    def num_ceps(self):
        """Number of cepstra in MFCC computation (including C0)

        Must be smaller of equal to `num_bins`

        """
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

        Warning: Not sufficient to get HTK compatible features (need
        to change other parameters).

        """
        return self._options.htk_compat

    @htk_compat.setter
    def htk_compat(self, value):
        self._options.htk_compat = value

    @property
    def ndims(self):
        return self.num_ceps
