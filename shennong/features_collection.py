"""Provides the `FeaturesCollection` class to manipulate speech features

- A `FeaturesCollection` is basically a dictionnary of
  :class:`~shennong.features.Features` indexed by names.

- A collection can be saved to and loaded from a file with the :func:`save` and
  :func:`load` methods.

- The following table details the supported file formats and compares the
  obtained file size, writing and reading times on MFCC features computed on
  the `Buckeye Corpus <https://buckeyecorpus.osu.edu`_
  (English, 40 speakers, about 38 hours of speech and 254 files):

  ===========  =========  =========  ============  ============
  File format  Extension  File size  Writing time  Reading time
  ===========  =========  =========  ============  ============
  pickle       .pkl       883.7 MB   0:00:07       0:00:05
  h5features   .h5f       873.0 MB   0:00:21       0:00:07
  numpy        .npz       869.1 MB   0:02:30       0:00:22
  matlab       .mat       721.1 MB   0:00:59       0:00:11
  kaldi        .ark       1.3 GB     0:00:06       0:00:07
  CSV          folder     4.8 GB     0:03:02       0:03:11
  ===========  =========  =========  ============  ============

- The documention for the *h5features* format is available at
  https://docs.cognitive-ml.fr/h5features.

- The CSV serializer writes into a folder: one CSV file per feature in the
  ``FeaturesCollection``, with an optional JSON file storing features
  properties.

Examples
--------

>>> import os
>>> import numpy as np
>>> from shennong import Features, FeaturesCollection

Create a collection of two random features

>>> fc = FeaturesCollection()
>>> fc['feat1'] = Features(np.random.random((5, 2)), np.linspace(0, 4, num=5))
>>> fc['feat2'] = Features(np.random.random((3, 2)), np.linspace(0, 2, num=3))
>>> fc.keys()
dict_keys(['feat1', 'feat2'])

Save the collection to a npz file

>>> fc.save('features.npz')

Load it back to a new collection

>>> fc2 = FeaturesCollection.load('features.npz')
>>> fc2.keys()
dict_keys(['feat1', 'feat2'])
>>> fc == fc2
True

>>> os.remove('features.npz')

"""

import collections
import numpy as np

from shennong import Features
from shennong.logger import get_logger
from shennong.serializers import get_serializer


class FeaturesCollection(dict):
    """Handles a collection of :class:`~shennong.Features` as a dictionary"""
    @classmethod
    def load(cls, filename, serializer=None,
             log=get_logger('serializer', 'warning')):
        """Loads a FeaturesCollection from a `filename`

        Parameters
        ----------
        filename : str
            The file to load
        serializer : str, optional
            The file serializer to use for loading, if not specified
            guess the serializer from the `filename` extension
        log : logging.Logger, optional
            Where to send log messages. Default to a logger named 'serializer'
            with a 'warning' level.

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
        return get_serializer(cls, filename, log, serializer).load()

    def save(self, filename, serializer=None, with_properties=True,
             log=get_logger('serializer', 'warning'), **kwargs):
        """Saves a FeaturesCollection to a `filename`

        Parameters
        ----------
        filename : str
            The file to write
        serializer : str, optional
            The file serializer to use for loading, if not specified
            guess the serializer from the `filename` extension
        with_properties : bool, optional
            When False do not save the features properties, default to True.
        log : logging.Logger, optional
            Where to send log messages. Default to a logger named 'serializer'
            with a 'warning' level.
        compress : bool_or_str_or_int, optional
            Only valid for numpy (.npz), matlab (.mat) and h5features (.h5f)
            serializers. When True compress the file. Default to True.
        scp : bool, optional
            Only valid for kaldi (.ark) serializer. When True writes a .scp
            file along with the .ark file. Default to False.

        Raises
        ------
        IOError
            If the file `filename` already exists
        ValueError
            If the `serializer` or the file extension is not supported,
            if the features saving fails.

        """
        get_serializer(self.__class__, filename, log, serializer).save(
            self, with_properties=with_properties, **kwargs)

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
