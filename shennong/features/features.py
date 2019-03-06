"""Builds, saves, loads and manipulate features data"""


import copy
import numpy as np

from shennong.features.handlers import get_handler
from shennong.utils import list2array, array2list, dict_equal


class Features:
    def __init__(self, data, times, properties={}, validate=True):
        # make sure the data type is correct
        if not isinstance(data, np.ndarray):
            raise ValueError('data must be a numpy array')
        self._data = data

        # make sure the times type is correct
        if not isinstance(data, np.ndarray):
            raise ValueError('times must be a numpy array')
        self._times = times

        # make sure the properties type is correct
        if not isinstance(properties, dict):
            raise ValueError('properties must be a dictionnary')
        self._properties = properties

        # make sure the features are in a valid state
        if validate is True:
            self.validate()

    @property
    def data(self):
        """The underlying features data as a numpy array"""
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

    def _to_dict(self, array_as_list=False):
        """Returns the features as a dictionary

        Parameters
        ----------
        array_as_list : bool, optional
            When True, converts numpy arrays to lists, default to
            False

        Returns
        -------
        features : dict
            A dictionary with the following keys: 'data', 'times' and
            'properties'.

        """
        def fun(x):
            return array2list(x) if array_as_list else x

        # we may have arrays in properties as well (when using CMVN)
        return {
            'data': fun(self.data),
            'times': fun(self.times),
            'properties': fun(self.properties)}

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
                .format(missing_keys))

        return Features(
            list2array(features['data']),
            list2array(features['times']),
            properties=list2array(features['properties']),
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
        if self.shape != other.shape:
            return False

        if not dict_equal(self.properties, other.properties):
            return False

        if not np.allclose(self.times, other.times, atol=atol, rtol=rtol):
            return False

        if not np.allclose(self.data, other.data, atol=atol, rtol=rtol):
            return False

        return True

    def copy(self, dtype=None):
        """Returns a copy of the features

        Allocates new arrays for data, times and properties

        Parameters
        ----------
        dtype : type, optional
            When specified converts the data and times arrays to the
            requested `dtype`

        Returns
        -------
        features : Features
           A new instance of Features copied from this one.

        """
        if dtype:
            return Features(
                self.data.astype(dtype),
                self.times.astype(dtype),
                properties=copy.deepcopy(self.properties),
                validate=False)

        return Features(
            self.data.copy(),
            self.times.copy(),
            properties=copy.deepcopy(self.properties),
            validate=False)

    def is_valid(self):
        """Returns True if the features are in a valid state

        Returns False otherwise. Consistency is checked for features's
        data, times and labels.

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
        errors = []

        # check data dimensions
        ndim = self.data.ndim
        if not ndim == 2:
            errors.append('data dimension must be 2 but is {}'.format(ndim))

        # check times dimensions
        ndim = self.times.ndim
        if not ndim == 1:
            errors.append('times dimension must be 1 but is {}'.format(ndim))

        nframes1 = self.data.shape[0]
        nframes2 = self.times.shape[0]
        if not nframes1 == nframes2:
            errors.append('mismath in number of frames: {} for data but {} '
                          'for times'.format(nframes1, nframes2))

        # TODO check properties

        if len(errors):
            raise ValueError('invalid features: {}'.format(', '.join(errors)))

    def concatenate(self, other):
        """Returns the concatenation of this features with `other`

        Build a new Features instance made of the concatenation of
        this instance with the other instance. Their `times` must be
        the equal.

        Parameters
        ----------
        other : Features, shape = [nframes, nlabels2]
            The other features to concatenate at the end of this one

        Returns
        -------
        features : Features, shape = [nframes, nlabels1 + nlabels2]

        Raises
        ------
        ValueError
            If `other` cannot be concatenated because of inconsistencies

        """
        # ensures time axis is shared accross the two features
        if not np.array_equal(self.times, other.times):
            raise ValueError('times are not equal')

        return Features(
            np.hstack((self.data, other.data)),
            np.hstack((self.labels, other.labels)),
            self.times,
            # TODO need a FeaturesParameter class: assign properties
            # per column
            self.properties.update(other.properties))


class FeaturesCollection(dict):
    _value_type = Features

    @classmethod
    def load(cls, filename, handler=None):
        """Loads a FeaturesCollection from a `filename`

        Parameters
        ----------
        filename : str
            The file to load
        handler : str, optional
            The file handler to use for loading, if not specified
            guess the handler from the `filename` extension

        Returns
        -------
        features : :class:`~shennong.features.FeaturesCollection`
            The features loaded from the `filename`

        Raises
        ------
        IOError
            If the `filename` cannot be read
        ValueError
            If the `handler` or the file extension is not supported,
            if the features loading fails.

        """
        return get_handler(cls, filename, handler).load()

    def save(self, filename, handler=None, **kwargs):
        get_handler(self.__class__, filename, handler).save(self, **kwargs)

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

    def by_speaker(self, spk2utt):
        """Returns a dict of :class:`FeaturesCollection` indexed by speakers

        Parameters
        ----------
        spk2utt : dict
            A mapping of speakers to their associated utterances
            (items in the ``FeaturesCollection``). We must have
            ``spk2utt.values() == self.keys()``.

        Returns
        -------
        features : dict of FeaturesCollection
            A list of FeaturesCollection instances, one per speaker
            defined in `utt2spk`

        Raises
        ------
        ValueError
            If one utterance in the collection is not mapped in
            `spk2utt`.

        """
        undefined_utts = set(self.keys()).difference(set(spk2utt.values()))
        if undefined_utts:
            raise ValueError(
                'following utterances are not defined in spk2utt: {}'
                .format(sorted(undefined_utts)))

        return {spk: FeaturesCollection({utt: self[utt] for utt in utts})
                for spk, utts in spk2utt}
