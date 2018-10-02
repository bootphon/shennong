"""Provides the FilterbankProcessor class to extract filterbank features

Extract mel-filterbank features from an audio signal. Use the Kaldi
implementation (see [kaldi-fbank]_).

    *AudioData* ---> FilterbankProcessor ---> *Features*


Examples
--------

>>> from shennong.audio import AudioData
>>> from shennong.features.filterbank import FilterbankProcessor
>>> audio = AudioData.load('./test/data/test.wav')

Initialize the filterbank processor with some options and compute the
features:

>>> processor = FilterbankProcessor(sample_rate=audio.sample_rate)
>>> processor.use_energy = False
>>> fbank = processor.process(audio)
>>> fbank.shape
(142, 23)

Using energy adds a column to the output:

>>> processor.use_energy = True
>>> fbank = processor.process(audio)
>>> fbank.shape
(142, 24)

References
----------

.. [kaldi-fbank] http://kaldi-asr.org/doc/structkaldi_1_1FbankOptions.html

"""

import kaldi.feat.fbank
import kaldi.matrix

from shennong.features.features import Features
from shennong.core.processor import MelFeaturesProcessor


class FilterbankProcessor(MelFeaturesProcessor):
    def __init__(self, sample_rate=16000, frame_shift=0.01,
                 frame_length=0.025, dither=1.0, preemph_coeff=0.97,
                 remove_dc_offset=True, window_type='povey',
                 round_to_power_of_two=True, blackman_coeff=0.42,
                 snip_edges=True, num_bins=23, low_freq=20,
                 high_freq=0, vtln_low=100, vtln_high=-500,
                 use_energy=False, energy_floor=0.0, raw_energy=True,
                 htk_compat=False, use_log_fbank=True, use_power=True):
        # Forward options to MelFeaturesProcessor
        super().__init__(sample_rate, frame_shift, frame_length,
                         dither, preemph_coeff, remove_dc_offset, window_type,
                         round_to_power_of_two, blackman_coeff, snip_edges,
                         num_bins, low_freq, high_freq, vtln_low, vtln_high)

        self._options = kaldi.feat.fbank.FbankOptions()
        self._options.frame_opts = self._frame_options
        self._options.mel_opts = self._mel_options

        self.use_energy = use_energy
        self.energy_floor = energy_floor
        self.raw_energy = raw_energy
        self.htk_compat = htk_compat
        self.use_log_fbank = use_log_fbank
        self.use_power = use_power

    @property
    def use_energy(self):
        """Add an extra dimension with energy to the filterbank output"""
        return self._options.use_energy

    @use_energy.setter
    def use_energy(self, value):
        self._options.use_energy = value

    @property
    def energy_floor(self):
        """Floor on energy (absolute, not relative) in filterbanks"""
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
    def htk_compat(self):
        """If True, get closer to HTK filterbank features

        Put energy last.

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
    def use_log_fbank(self):
        """If true, produce log-filterbank, else produce linear"""
        return self._options.use_log_fbank

    @use_log_fbank.setter
    def use_log_fbank(self, value):
        self._options.use_log_fbank = value

    @property
    def use_power(self):
        """If true, use power, else use magnitude"""
        return self._options.use_power

    @use_power.setter
    def use_power(self, value):
        self._options.use_power = value

    def parameters(self):
        params = super().parameters()
        params.update({
            'use_energy': self.use_energy,
            'energy_floor': self.energy_floor,
            'raw_energy': self.raw_energy,
            'htk_compat': self.htk_compat,
            'use_log_fbank': self.use_log_fbank,
            'use_power': self.use_power})
        return params

    def process(self, signal, vtln_warp=1.0):
        """Compute filterbank features with the specified options

        Do an optional feature-level vocal tract length normalization
        (VTLN) when `vtln_warp` != 1.0.

        Parameters
        ----------
        signal : AudioData, shape = [nsamples, 1]
            The input audio signal to compute the features on, must be
            mono
        vtln_warp : float, optional
            The VTLN warping factor to be applied when computing
            features. Be 1.0 by default, meaning no warping is to be
            done.

        Returns
        -------
        fbank : Features, shape = [nframes, `num_bins` + energy]
            The computed MFCCs, output will have as many rows as there
            are frames (depends on the specified options `frame_shift`
            and `frame_length`), and as many columns as there are
            mel-frequency bins (the `num_ceps` option), plus one
            additional column if `use_energy` is True.

        Raises
        ------
        ValueError
            If the input `signal` has more than one channel (i.e. is
            not mono). If `sample_rate` != `signal.sample_rate`.

        """
        if signal.nchannels != 1:
            raise ValueError(
                'signal must have one dimension, but it has {}'
                .format(signal.nchannels))

        if self.sample_rate != signal.sample_rate:
            raise ValueError(
                'processor and signal mismatche in sample rates: '
                '{} != {}'.format(self.sample_rate, signal.sample_rate))

        data = kaldi.matrix.SubMatrix(
            kaldi.feat.fbank.Fbank(self._options).compute(
                kaldi.matrix.SubVector(signal.data), vtln_warp)).numpy()

        return Features(
            data, self.times(data.shape[0]), self.parameters())
