"""Builds, saves, loads and manipulate speech features"""


import collections
import copy
import logging
import numpy as np

from shennong.features.serializers import get_serializer
from shennong.utils import dict_equal


class Features:
    _log = logging.getLogger()

    def __init__(self, data, times, properties={}, validate=True):
        self._data = data
        self._times = times
        self._properties = properties

        # make sure the features are in a valid state
        if validate is True:
            self.validate()

    @property
    def data(self):
        """The underlying features data as a numpy matrix"""
        return self._data

    @property
    def times(self):
        """The frames timestamps on the vertical axis"""
        return self._times

    @property
    def dtype(self):
        """The type of the features data samples"""
        return self.data.dtype

    @property
    def shape(self):
        """The shape of the features data, as (nframes, ndims)"""
        return self.data.shape

    @property
    def ndims(self):
        """The number of dimensions of a features frame (feat.shape[1])"""
        return self.shape[1]

    @property
    def nframes(self):
        """The number of features frames (feat.shape[0])"""
        return self.shape[0]

    @property
    def properties(self):
        """A dictionnary of properties used to build the features

        Properties are references to the features extraction pipeline,
        parameters and source audio file used to generate the
        features.

        """
        return self._properties

    def _to_dict(self):
        """Returns the features as a dictionary

        Returns
        -------
        features : dict
            A dictionary with the following keys: 'data', 'times' and
            'properties'.

        """
        return {
            'data': self.data,
            'times': self.times,
            'properties': self.properties}

    @staticmethod
    def _from_dict(features, validate=True):
        """Return an instance of Features loaded from a dictionary

        Parameters
        ----------
        features : dict
            The dictionary to load the features from. Must have the
            following keys: 'data', 'times' and
            'properties'.

        validate : bool, optional
            When True, validate the features before returning. Default
            to True

        Returns
        -------
        An instance of ``Features``

        Raises
        ------
        ValueError
            If the ``features`` don't have the requested keys or if
            the underlying features data is not valid.

        """
        requested_keys = {'data', 'times', 'properties'}
        missing_keys = requested_keys - set(features.keys())
        if missing_keys:
            raise ValueError(
                'cannot read features from dict, missing keys: {}'
                .format(', '.join(missing_keys)))

        return Features(
            features['data'],
            features['times'],
            properties=features['properties'],
            validate=validate)

    def __eq__(self, other):
        """Returns True if `self` is equal `other`, False otherwise"""
        # object identity
        if self is other:
            return True

        # quick tests on attributes
        if self.shape != other.shape or self.dtype != other.dtype:
            return False

        # properties equality
        if not dict_equal(self.properties, other.properties):
            return False

        # timestamps equality
        if not np.array_equal(self.times, other.times):
            return False

        # features matrices equality
        if not np.array_equal(self.data, other.data):
            return False

        return True

    def is_close(self, other, rtol=1e-5, atol=1e-8):
        """Returns True if `self` is approximately equal to `other`

        Parameters
        ----------
        other : Features
            The Features instance to be compared to this one
        rtol : float, optional
            Relative tolerance
        atol : float, optional
            Absolute tolerance

        Returns
        -------
        equal : bool
            True if these features are almost equal to the `other`

        See Also
        --------
        FeaturesCollection.is_close, numpy.allclose


        """
        if self is other:
            return True

        if self.shape != other.shape:
            return False

        if not dict_equal(self.properties, other.properties):
            return False

        if not np.array_equal(self.times, other.times):
            return False

        if not np.allclose(self.data, other.data, atol=atol, rtol=rtol):
            return False

        return True

    def copy(self, dtype=None, subsample=None):
        """Returns a copy of the features

        Allocates new arrays for data, times and properties

        Parameters
        ----------
        dtype : type, optional
            When specified converts the data and times arrays to the
            requested `dtype`
        subsample : int, optional
            When specified subsample the features every `subsample` frames.
            When not specified do not do subsampling.

        Raises
        ------
        ValueError if `subsample` is defined but is not a strictly positive
        integer

        Returns
        -------
        features : Features
           A new instance of Features copied from this one.

        """
        # by default we do not subsample
        if subsample is None:
            subsample = 1
        else:
            if not isinstance(subsample, int) or subsample <= 0:
                raise ValueError(
                    f'subsample must be a strictly positive integer, '
                    f'it is: {subsample}')

        if dtype:
            return Features(
                self.data[0:self.nframes:subsample].astype(dtype),
                self.times[0:self.nframes:subsample].astype(dtype),
                properties=copy.deepcopy(self.properties),
                validate=False)

        return Features(
            self.data[0:self.nframes:subsample].copy(),
            self.times[0:self.nframes:subsample].copy(),
            properties=copy.deepcopy(self.properties),
            validate=False)

    def is_valid(self):
        """Returns True if the features are in a valid state

        Returns False otherwise. Consistency is checked for features's
        data, times and properties.

        See Also
        --------
        Features.validate

        """
        try:
            self.validate()
        except ValueError:
            return False
        return True

    def validate(self):
        """Raises a ValueError if the features are not in a valid state"""
        # accumulate detected errors and display them at the end
        errors = []

        # basic checks on types
        if not isinstance(self.data, np.ndarray):
            errors.append('data must be a numpy array')
        if not isinstance(self.times, np.ndarray):
            errors.append('times must be a numpy array')
        if not isinstance(self.properties, dict):
            errors.append('properties must be a dictionnary')

        if errors:
            raise ValueError(
                'invalid features data types: {}'.format(', '.join(errors)))

        # check arrays dimensions
        if not self.data.ndim == 2:
            errors.append(
                'data dimension must be 2 but is {}'.format(self.data.ndim))
        if self.times.ndim > 2:
            errors.append(
                'times dimension must be 1 or 2 but is {}'.format(
                    self.times.ndim))
        if self.times.ndim == 2 and self.times.shape[1] != 2:
            errors.append('times shape[1] must be 2, it is {}'.format(
                self.times.shape[1]))

        nframes1 = self.data.shape[0]
        nframes2 = self.times.shape[0]
        if not nframes1 == nframes2:
            errors.append(
                'mismatch in number of frames: {} for data but {} '
                'for times'.format(nframes1, nframes2))

        if errors:
            raise ValueError(
                'invalid features dimensions: {}'.format(', '.join(errors)))

        # check if time is increasing. This check comes from
        # h5features/labels.py
        index = (np.argsort(self.times) if self.times.ndim == 1
                 else np.lexsort(self.times.T))
        if not all(n == index[n] for n in range(self.nframes)):
            raise ValueError('times is not sorted in increasing order')

        # check all values in array are finit (not infinity nor nan)
        if not np.all(np.isfinite(self.data)):
            raise ValueError(
                'data contains non-finite numbers (nan of infinity)')

    def concatenate(self, other, tolerance=0):
        """Returns the concatenation of this features with `other`

        Build a new Features instance made of the concatenation of
        this instance with the other instance. Their `times` must be
        the equal.

        Parameters
        ----------
        other : Features, shape = [nframes +/- tolerance, ndim2]
            The other features to concatenate at the end of this one
        tolerance : int, optional
            If the number of frames of the two features is different,
            trim the longest one up to a frame difference of
            `tolerance`, otherwise raise a ValueError. This option is
            usefull when concatenating pitch with other 'standard'
            features because pitch processing includes a downsampling
            which can alter the resulting number of frames (the same
            tolerance is applied in Kaldi, e.g. in paste-feats).
            Default to 0.

        Returns
        -------
        features : Features, shape = [nframes +/- tolerance, ndim1 + ndim2]

        Raises
        ------
        ValueError
            If `other` cannot be concatenated because of
            inconsistencies: number of frames difference greater than
            tolerance, inequal times values.

        """
        # check the number of frames is within the tolerance
        need_trim = False
        diff = abs(self.nframes - other.nframes)
        if diff:
            if not tolerance:
                raise ValueError(
                    'features have a different number of frames')
            if tolerance and diff > tolerance:
                raise ValueError(
                    'features differs number of frames, and '
                    'greater than tolerance: |{} - {}| > {}'.format(
                        self.nframes, other.nframes, tolerance))

            self._log.warning(
                'features differs in number of frames, but '
                'within tolerance (|%s - %s| <= %s), trim the longest one',
                self.nframes, other.nframes, tolerance)
            need_trim = True

        # trim the longest features to the size of the shortest one
        data1 = self.data
        data2 = other.data
        times1 = self.times
        times2 = other.times
        if need_trim:
            if self.nframes > other.nframes:
                data1 = data1[:-diff]
                times1 = times1[:-diff]
            else:
                data2 = data2[:-diff]
                times2 = times2[:-diff]

        # ensures time axis is shared accross the two features
        if not np.allclose(times1, times2):
            raise ValueError('times are not equal')

        # merge properties of the two features
        properties = copy.deepcopy(self.properties)
        other_properties = copy.deepcopy(other.properties)
        properties.update(
            {k: v for k, v in other_properties.items() if k != 'pipeline'})
        if 'pipeline' not in properties:
            properties['pipeline'] = []
        if 'pipeline' in other_properties:
            for k in other_properties['pipeline']:
                properties['pipeline'].append(k)
                columns = properties['pipeline'][-1]['columns']
                properties['pipeline'][-1]['columns'] = [
                    columns[0] + self.ndims, columns[1] + self.ndims]

        return Features(
            np.hstack((data1, data2)), times1, properties=properties)


class FeaturesCollection(dict):
    # a tweak inspired by C++ metaprogramming to avoid import loops
    # with shennong.features.serializers
    _value_type = Features

    @classmethod
    def load(cls, filename, serializer=None):
        """Loads a FeaturesCollection from a `filename`

        Parameters
        ----------
        filename : str
            The file to load
        serializer : str, optional
            The file serializer to use for loading, if not specified
            guess the serializer from the `filename` extension

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
        return get_serializer(cls, filename, serializer).load()

    def save(self, filename, serializer=None, **kwargs):
        get_serializer(
            self.__class__, filename, serializer).save(self, **kwargs)

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
            sub-collection they belong to in the partitino. We must
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
