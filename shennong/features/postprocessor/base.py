"""Abstract base class for all the features postprocessors"""

import abc

from shennong.features.processor.base import FeaturesProcessor


class FeaturesPostProcessor(FeaturesProcessor):
    @abc.abstractmethod
    def process(self, features):
        """Returns features post-processed from input `features`"""
        pass  # pragma: no cover
