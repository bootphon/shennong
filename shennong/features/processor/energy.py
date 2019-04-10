"""Extraction of energy from audio signals

    :class:`~shennong.audio.AudioData` --> EnergyProcessor
    --> :class:`~shennong.features.features.Features`

Computes the energy on window frames extracted from an audio
signal. This algorithm is identical to the first coefficient of
:class:`~shennong.features.processor.mfcc.MfccProcessor` or
:class:`~shennong.features.processor.plp.PlpProcessor`.

Examples
--------

>>> from shennong.audio import AudioData
>>> from shennong.features.processor.energy import EnergyProcessor
>>> audio = AudioData.load('./test/data/test.wav')

Computes energy on the audio signal:

>>> proc = EnergyProcessor(sample_rate=audio.sample_rate)
>>> energy1 = proc.process(audio)
>>> energy1.shape
(140, 1)

By default the energy is log-compressed, you can desactivate
compression available options for compression are 'log', 'sqrt' and
'off':

>>> proc.compression = 'off'
>>> energy2 = proc.process(audio)
>>> np.allclose(np.log(energy2.data), energy1.data, rtol=1)
True

The two energies above are not strictly identical because of
dithering.

You can also fix the framing and windowing parameters:

>>> proc.frame_shift = 0.02
>>> proc.frame_length = 0.05
>>> proc.window_type = 'hanning'
>>> energy3 = proc.process(audio)
>>> energy3.shape
(69, 1)

"""

import numpy as np
import kaldi.feat.window
import kaldi.matrix

from shennong.features import Features
from shennong.features.processor.base import FramesProcessor


class EnergyProcessor(FramesProcessor):
    def __init__(self, sample_rate=16000, frame_shift=0.01,
                 frame_length=0.025, dither=1.0, preemph_coeff=0.97,
                 remove_dc_offset=True, window_type='povey',
                 round_to_power_of_two=True, blackman_coeff=0.42,
                 snip_edges=True, raw_energy=True, compression='log'):
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

        self._compression_fun = {
            'off': lambda x: x,
            'log': np.log,
            'sqrt': np.sqrt}
        self.compression = compression
        self.raw_energy = raw_energy

    @property
    def ndims(self):
        return 1

    @property
    def compression(self):
        """Type of energy compression

        Must be 'off' (disable compression), 'log' (natural logarithm)
        or 'sqrt' (squared root).

        """
        return self._compression

    @compression.setter
    def compression(self, value):
        if value not in self._compression_fun.keys():
            raise ValueError(
                'compression must be in {}, it is {}'.format(
                    ', '.join(self._compression_fun.keys()), value))

        self._compression = value

    @property
    def raw_energy(self):
        """If true, compute energy before preemphasis and windowing"""
        return self._raw_energy

    @raw_energy.setter
    def raw_energy(self, value):
        self._raw_energy = value

    def process(self, signal):
        """Computes energy on the input `signal`

        Parameters
        ----------
        signal : :class:`~signal.audio.audioData`

        Returns
        -------
        energy : :class:`~shennong.features.features.Features`
            The computed - and compressed - energy

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

        if self.raw_energy:
            old_conf = self.get_params()
            self.preemph_coeff = 0
            self.window_type = 'rectangular'

        # number of frames in the framed signal
        nframes = kaldi.feat.window.num_frames(
            signal.nsamples, self._frame_options, flush=True)

        # a kaldi view of the numpy signal
        signal = kaldi.matrix.SubVector(signal.data)

        # windowing function to compute frames
        window = kaldi.feat.window.FeatureWindowFunction.from_options(
            self._frame_options)

        # compression function to compress energy
        compression = self._compression_fun[self._compression]

        # pre-allocate the resulting energy
        energy = np.zeros((nframes, 1))

        # pre-allocate a buffer for the frames, extract the frames and
        # compute the energy on them
        out_frame = kaldi.matrix.Vector(self._frame_options.window_size())
        for frame in range(nframes):
            kaldi.feat.window.extract_window(
                0, signal, frame, self._frame_options, window, out_frame)

            # avoid doing log on 0 (should be avoided already by
            # dithering, but who knows...)
            energy[frame] = compression(max(
                (out_frame.numpy() ** 2).sum(),
                np.finfo(np.float32).tiny))

        if self.raw_energy:
            self.set_params(**old_conf)

        return Features(energy, self.times(nframes), self.get_params())
