"""Extraction of spectrogram from audio signals

Extract spectrogram (log of the power spectrum) from an audio
signal. Uses the Kaldi implementation (see [kaldi-spec]_):

    :class:`~shennong.audio.Audio` ---> SpectrogramProcessor \
    ---> :class:`~shennong.features.Features`

Examples
--------

>>> from shennong.audio import Audio
>>> from shennong.processor.spectrogram import SpectrogramProcessor
>>> audio = Audio.load('./test/data/test.wav')

Initialize the spectrogram processor with some options and compute the
features:

>>> processor = SpectrogramProcessor(sample_rate=audio.sample_rate)
>>> processor.window_type = 'hanning'
>>> spect = processor.process(audio)
>>> spect.shape
(140, 257)


References
----------

.. [kaldi-spec] http://kaldi-asr.org/doc/classkaldi_1_1SpectrogramComputer.html

"""

import numpy as np
import kaldi.feat.spectrogram

from shennong import Features
from shennong.processor.base import FramesProcessor


class SpectrogramProcessor(FramesProcessor):
    """Spectogram"""
    def __init__(self, sample_rate=16000, frame_shift=0.01,
                 frame_length=0.025, dither=1.0,
                 preemph_coeff=0.97, remove_dc_offset=True,
                 window_type='povey', round_to_power_of_two=True,
                 blackman_coeff=0.42, snip_edges=True,
                 energy_floor=0.0, raw_energy=True):
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

        self._options = kaldi.feat.spectrogram.SpectrogramOptions()
        self._options.frame_opts = self._frame_options

        self.energy_floor = energy_floor
        self.raw_energy = raw_energy

    @property
    def name(self):
        return 'spectrogram'

    @property
    def ndims(self):
        return int(self._frame_options.padded_window_size() / 2 + 1)

    @property
    def energy_floor(self):
        return self._options.energy_floor

    @energy_floor.setter
    def energy_floor(self, value):
        self._options.energy_floor = value

    @property
    def raw_energy(self):
        return self._options.raw_energy

    @raw_energy.setter
    def raw_energy(self, value):
        self._options.raw_energy = bool(value)

    def process(self, signal, vtln_warp=1.0):
        """Compute spectrogram with the specified options

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

        # force 16 bits integers
        signal = signal.astype(np.int16).data
        data = kaldi.matrix.SubMatrix(
            kaldi.feat.spectrogram.Spectrogram(self._options).compute(
                kaldi.matrix.SubVector(signal), vtln_warp)).numpy()

        return Features(
            data, self.times(data.shape[0]), properties=self.get_properties())
