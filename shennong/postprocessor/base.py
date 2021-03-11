"""A post-processor takes features as input and output new features:

    :class:`~shennong.features.Features` -->
    FeaturesPostProcessor -->
    :class:`~shennong.features.Features`

"""

import abc
import copy

from shennong.processor.base import FeaturesProcessor


class FeaturesPostProcessor(FeaturesProcessor):
    @abc.abstractmethod
    def process(self, features):
        """Returns features post-processed from input `features`"""
        pass  # pragma: no cover

    def get_properties(self, features):
        properties = copy.deepcopy(features.properties)
        properties[self.name] = self.get_params()

        if 'pipeline' not in properties:
            properties['pipeline'] = []

        properties['pipeline'].append({
            'name': self.name,
            'columns': [0, self.ndims - 1]})

        return properties
