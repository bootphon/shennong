"""Builds, saves, loads and manipulate a collection of speech features"""

import collections
import numpy as np

from shennong import Features
from shennong.serializers import get_serializer


class FeaturesCollection(dict):
    # a tweak inspired by C++ metaprogramming to avoid import loops
    # with shennong.features.serializers
    _value_type = Features

    @classmethod
    def load(cls, filename, serializer=None, log_level='warning'):
        """Loads a FeaturesCollection from a `filename`

        Parameters
        ----------
        filename : str
            The file to load
        serializer : str, optional
            The file serializer to use for loading, if not specified
            guess the serializer from the `filename` extension
        log_level : str, optional
            The log level must be 'debug', 'info', 'warning' or 'error'.
            Default to 'warning'.


        Returns
        -------
        features : :class:`~shennong.features.FeaturesCollection`
            The features loaded from the `filename`

        Raises
        ------
        IOError
            If the `filename` cannot be read
        ValueError
            If the `serializer` or the file extension is not supported,
            if the features loading fails.

        """
        return get_serializer(cls, filename, log_level, serializer).load()

    def save(self, filename, serializer=None, log_level='warning', **kwargs):
        get_serializer(
            self.__class__, filename, log_level, serializer).save(
                self, **kwargs)

    def is_valid(self):
        """Returns True if all the features in the collection are valid"""
        for features in self.values():
            if not features.is_valid():
                return False
        return True

    def is_close(self, other, rtol=1e-5, atol=1e-8):
        """Returns True `self` is approximately equal to `other`

        Parameters
        ----------
        other : FeaturesCollection
            The collection of features to compare to the current one
        rtol : float, optional
            Relative tolerance
        atol : float, optional
            Absolute tolerance

        Returns
        -------
        equal : bool
            True if this collection is almost equal to the `other`

        See Also
        --------
        Features.is_close, numpy.allclose

        """
        if not self.keys() == other.keys():
            return False

        for k in self.keys():
            if not self[k].is_close(other[k], rtol=rtol, atol=atol):
                return False

        return True

    def partition(self, index):
        """Returns a partition of the collection as a dict of FeaturesCollection

        This method is usefull to create sub-collections from an
        existing one, for instance to make one sub-collection per
        speaker, or per gender, etc...

        Parameters
        ----------
        index : dict
            A mapping with, for each item in this collection, the
            sub-collection they belong to in the partition. We must
            have ``index.keys() == self.keys()``.

        Returns
        -------
        features : dict of FeaturesCollection
            A dictionnary of FeaturesCollection instances, one per
            speaker defined in `index`.

        Raises
        ------
        ValueError
            If one utterance in the collection is not mapped in
            `index`.

        """
        undefined_utts = set(self.keys()).difference(index.keys())
        if undefined_utts:
            raise ValueError(
                'following items are not defined in the partition index: {}'
                .format(', '.join(sorted(undefined_utts))))

        reverse_index = collections.defaultdict(list)
        for key, value in index.items():
            reverse_index[value].append(key)

        return {k: FeaturesCollection({item: self[item] for item in items})
                for k, items in reverse_index.items()}

    def trim(self, vad):
        """Returns a new instance of FeaturesCollection where each features
        has been trimmed with the corresponding VAD.

        Parameters
        ----------
        vad : dict of boolean ndarrays
            A dictionnary of arrays indicating which frame to keep.

        Returns
        -------
        features: FeaturesCollection
            A new FeaturesCollection trimmed with the input VAD

        Raises
        ------
        ValueError
            If the utterances are not the same. If the VAD arrays are
            not boolean arrays.
        """
        if vad.keys() != self.keys():
            raise ValueError('Vad keys are different from this keys.')

        for key in vad.keys():
            if vad[key].dtype != np.dtype('bool'):
                raise ValueError('Vad arrays must be arrays of bool.')
            if vad[key].shape[0] != self[key].nframes:
                raise ValueError(
                    'Vad arrays length must be equal to the number of frames.')

        return FeaturesCollection({
            k: Features(
                self[k].data[vad[k]],
                self[k].times[vad[k]],
                properties=self[k].properties) for k in self.keys()})
