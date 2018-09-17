"""Provides the OneHotProcessor to extract one hot features

One hot features are built from a speech signal and a time alignement
of the pronounced phonemes.

    *AudioData* + *Alignement* ---> OneHotProcessor ---> *Features*

"""

import numpy as np

from shennong.features.features import Features
from shennong.features.processor import FeaturesProcessor


class OneHotProcessor(FeaturesProcessor):
    def __init__(self, frame_shift=0.01, frame_length=0.025):
        self.frame_shift = frame_shift
        self.frame_length = frame_length

    def parameters(self):
        return {
            'frame_shitf': self.frame_shitf,
            'frame_length': self.frame_length}

    def times(self, nframes):
        return np.arange(nframes) * self.frame_shift + self.frame_length / 2.0

    def process(self, signal, alignement):
        pass
