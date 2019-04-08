"""Compute Voice Activity Detection (VAD) on features log-energy

    :class:`~shennong.features.features.Features` -->
    VadPostProcessor -->
    :class:`~shennong.features.features.Features`

Compute voice-activity detection for speech features using the Kaldi
implementation see [kaldi-vad]_: The output is, for each input frame,
1 if we judge the frame as voiced, 0 otherwise. There are no
continuity constraints.

This method is a very simple energy-based method which only looks at
the first coefficient of the input features, which is assumed to be
**a log-energy or something similar**. A cutoff is set, we use a
formula of the general type: `cutoff = 5.0 + 0.5 * (average
log-energy)`, and for each frame the decision is based on the
proportion of frames in a context window around the current frame,
which are above this cutoff.

.. note::

   This code is geared toward speaker-id applications and is not
   suitable for automatic speech recognition (ASR) because it makes
   independent decisions for each frame without imposing any notion
   of continuity.

Examples
--------

>>> import numpy as np
>>> from shennong.audio import AudioData
>>> from shennong.features.processor.mfcc import MfccProcessor
>>> from shennong.features.postprocessor.vad import VadPostProcessor
>>> audio = AudioData.load('./test/data/test.wav')
>>> mfcc = MfccProcessor().process(audio)

Computes the voice activity detection on the extracted MFCCs:

>>> processor = VadPostProcessor()
>>> vad = processor.process(mfcc)

For each frames of the MFCCs, vad is 1 if detected as a voiced frame,
0 otherwise:

>>> nframes = mfcc.shape[0]
>>> vad.shape == (nframes, 1)
True
>>> nvoiced = sum(vad.data[vad.data == 1])
>>> print('{} voiced frames out of {}'.format(nvoiced, nframes))
119 voiced frames out of 140


References
----------

.. [kaldi-vad] https://kaldi-asr.org/doc/voice-activity-detection_8h.html

"""

import kaldi.matrix
import kaldi.ivector
import numpy as np

from shennong.features import Features
from shennong.features.postprocessor.base import FeaturesPostProcessor


class VadPostProcessor(FeaturesPostProcessor):
    """Computes VAD on speech features

    """
    def __init__(self, energy_threshold=5.0, energy_mean_scale=0.5,
                 frames_context=0, proportion_threshold=0.6):
        self._options = kaldi.ivector.VadEnergyOptions()
        self.energy_threshold = energy_threshold
        self.energy_mean_scale = energy_mean_scale
        self.frames_context = frames_context
        self.proportion_threshold = proportion_threshold

    @property
    def energy_threshold(self):
        """Constant term in energy threshold for MFCC0 for VAD

        See also :func:`energy_mean_scale`

        """
        return self._options.vad_energy_threshold

    @energy_threshold.setter
    def energy_threshold(self, value):
        self._options.vad_energy_threshold = value

    @property
    def energy_mean_scale(self):
        """Scale factor of the mean log-energy

        If this is set to `s`, to get the actual threshold we let `m`
        be the mean log-energy of the file, and use `s*m +`
        :func:`energy_threshold`. Must be greater or equal to 0.

        """
        return self._options.vad_energy_mean_scale

    @energy_mean_scale.setter
    def energy_mean_scale(self, value):
        if value < 0:
            raise ValueError(
                'Energy mean scale must be >= 0, it is {}'.format(value))

        self._options.vad_energy_mean_scale = value

    @property
    def frames_context(self):
        """Number of frames of context on each side of central frame

        The size of the window for which energy is monitored is
        `2 * frames_context + 1`. Must be greater or equal to 0.

        """
        return self._options.vad_frames_context

    @frames_context.setter
    def frames_context(self, value):
        if value < 0:
            raise ValueError(
                'frames_context must be >= 0, it is {}'.format(value))
        self._options.vad_frames_context = value

    @property
    def proportion_threshold(self):
        """Proportion of frames beyond the energy threshold

        Parameter controlling the proportion of frames within the
        window that need to have more energy than the threshold. Must
        be in ]0, 1[.

        """
        return self._options.vad_proportion_threshold

    @proportion_threshold.setter
    def proportion_threshold(self, value):
        if value <= 0 or value >= 1:
            raise ValueError(
                'proportion_threshold must be in ]0, 1[, it is {}'
                .format(value))
        self._options.vad_proportion_threshold = value

    @property
    def ndims(self):
        return 1

    def process(self, features):
        """Computes voice activity detection (VAD) on the input `features`

        Parameters
        ----------
        features : :class:`~shennong.features.features.Features`, shape = [n,m]
            The speech features on which to look for voiced
            frames. The first coefficient must be a log-energy (or
            equivalent). Works well with
            :class:`~shennong.features.processor.mfcc.MfccProcessor` and
            :class:`~shennong.features.processor.plp.PlpProcessor`.

        Returns
        -------
        vad : :class:`~shennong.features.features.Features`, shape = [n,1]
            The output vad features are of dtype uint8 and contain 1
            for voiced frames or 0 for unvoiced frames.

        """
        data = kaldi.matrix.SubVector(
            kaldi.ivector.compute_vad_energy(
                self._options, kaldi.matrix.SubMatrix(features.data))).numpy()

        return Features(
            np.atleast_2d(data.astype(np.uint8)).T,
            features.times, properties=self.get_params())
