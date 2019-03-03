"""Builds, saves, loads and manipulate features data"""


import copy
import numpy as np

from shennong.features.handlers import get_handler


class Features:
    def __init__(self, data, times, properties={}, validate=True):
        self._data = data
        self._times = times
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
        """The type of the features data"""
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
            if array_as_list:
                if isinstance(x, dict):
                    return {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in x.items()}
                else:
                    return x.tolist()
            return x

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

        def fun(x):
            if isinstance(x, list):
                return np.asarray(x)
            elif isinstance(x, dict):
                return {k: fun(v) for k, v in x.items()}
            return x

        return Features(
            fun(features['data']), fun(features['times']),
            properties=fun(features['properties']),
            validate=validate)

    def __eq__(self, other):
        if self.shape != other.shape:
            return False

        if not np.array_equal(self.times, other.times):
            return False

        if not self.properties.keys() == other.properties.keys():
            return False

        for k, v in self.properties.items():
            w = other.properties[k]
            if not type(v) == type(w):
                return False
            if isinstance(v, np.ndarray):
                if not np.array_equal(v, w):
                    return False
            else:
                if not v == w:
                    return False
        if not np.array_equal(self.data, other.data):
            return False
        return True

    def copy(self):
        """Returns a copy of the features

        Allocates new arrays for both data and times

        """
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

    def save(self, filename, handler=None, ):
        get_handler(self.__class__, filename, handler).save(self)

    def is_valid(self):
        """Returns True if all the features in the collection are valid"""
        for features in self.values():
            if not features.is_valid():
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
