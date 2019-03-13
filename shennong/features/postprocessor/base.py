"""A post-processor takes features as input and output new features:

    :class:`~shennong.features.features.Features` -->
    FeaturesPostProcessor -->
    :class:`~shennong.features.features.Features`

"""

import abc

from shennong.features.processor.base import FeaturesProcessor


class FeaturesPostProcessor(FeaturesProcessor):
    @abc.abstractmethod
    def process(self, features):
        """Returns features post-processed from input `features`"""
        pass  # pragma: no cover
