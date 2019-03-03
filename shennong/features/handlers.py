"""Saves and loads features to/from various file formats"""

import abc
import json
import os

import h5features
import numpy as np
import scipy


def supported_extensions():
    """Returns the list of file extensions to save/load features

    Returns
    -------
    handlers : dict
        File extensions mapped to their related handler class

    """
    return {
        '.npz': NumpyHandler,
        '.mat': MatlabHandler,
        '.json': JsonHandler,
        '.h5f': H5featuresHandler,
        '.ark': KaldiHandler}


def supported_handlers():
    """Returns the list of file format handlers to save/load features

    Returns
    -------
    handlers : dict
        Handlers names mapped to their related class

    """
    return {
        'numpy': NumpyHandler,
        'matlab': MatlabHandler,
        'json': JsonHandler,
        'h5features': H5featuresHandler,
        'kaldi': KaldiHandler}


def get_handler(cls, filename, handler=None):
    """Returns the file handler from filename extension or handler name

    Parameters
    ----------
    cls : class
        Must be :class:`shennong.features.FeaturesCollection`, this is
        a tweak to avoid circular imports
    filename : str
        The file to be handled (load or save)
    handler : str, optional
        If not None must be one of the :func:`supported_handlers`, if
        not specified, guess the handler from the `filename`
        extension using :func:`supported_extensions`.

    Returns
    -------
    handler : instance of :class:`FeaturesHandler`
        The guessed handler class, a child class of
        :class:`FeaturesHandler`.

    Raises
    ------
    ValueError
        If the handler class cannot be guessed, or if `cls` is not
        :class:`~shennong.features.FeaturesCollection`

    """
    if cls.__name__ != 'FeaturesCollection':
        raise ValueError(
            'The `cls` parameter must be shennong.features.FeaturesCollection')

    if handler is None:
        # guess handler from file extension
        ext = os.path.splitext(filename)[1]
        if not ext:
            raise ValueError('no extension nor handler name specified')

        try:
            handler = supported_extensions()[ext]
        except KeyError:
            raise ValueError(
                'invalid extension {}, must be in {}'.format(
                    ext, list(supported_extensions().keys())))
    else:
        try:
            handler = supported_handlers()[handler]
        except KeyError:
            raise ValueError(
                'invalid handler {}, must be in {}'.format(
                    handler, list(supported_handlers().keys())))

    return handler(cls, filename)


class FeaturesHandler(metaclass=abc.ABCMeta):
    """Base class of a features file handler

    This class must be specialized to handle a given file type.

    Parameters
    ----------
    cls : class
        Must be :class:`shennong.features.FeaturesCollection`, this is
        a tweak to avoid circular imports
    filename : str
        The file to save/load features to/from

    """
    def __init__(self, cls, filename):
        self._features_collection = cls
        self._features = self._features_collection._value_type
        self._filename = filename

    @property
    def filename(self):
        return self._filename

    @abc.abstractmethod
    def _save(self, features):
        pass

    @abc.abstractmethod
    def _load(self):
        pass

    def load(self):
        """Returns a collection of features from the `filename`

        Returns
        -------
        features : :class:`~shennong.features.FeaturesCollection`
            The features stored in the file

        Raises
        ------
        IOError
            If the input file does not exist or cannot be read

        ValueError
            If the features cannot be loaded from the file or are not
            in a valid state

        """
        if not os.path.isfile(self.filename):
            raise IOError('file not found: {}'.format(self.filename))
        if not os.access(self.filename, os.R_OK):
            raise IOError('file not readable: {}'.format(self.filename))

        features = self._load()

        if not features.is_valid():
            raise ValueError(
                'features not valid in file: {}'.format(self.filename))

        return features

    def save(self, features):
        """Saves a collection of `features` to a file

        Parameters
        ----------
        features : :class:`~shennong.features.FeaturesCollection`
            The features to store in the file

        Raises
        ------
        IOError
            If the output file already exists

        ValueError
            If the features cannot be saved to the file, are not in a
            valid state or are not an instance of
            :class:`~shennong.features.FeaturesCollection`

        """
        if os.path.isfile(self.filename):
            raise IOError('file already exists: {}'.format(self.filename))

        if not isinstance(features, self._features_collection):
            raise ValueError(
                'features must be {} but are {}'
                .format(
                    self._features_collection.__name__,
                    features.__class__.__name__))

        if not features.is_valid():
            raise ValueError('features are not valid')

        self._save(features)


class NumpyHandler(FeaturesHandler):
    """Saves and loads features to/from the numpy '.npz' format"""
    def _save(self, features, compress=True):
        # represent the features as dictionaries
        data = {k: v._to_dict() for k, v in features.items()}

        # save (and optionally compress) the features
        save = np.savez_compressed if compress is True else np.savez
        save(open(self.filename, 'wb'), features=data)

    def _load(self):
        data = np.load(open(self.filename, 'rb'))['features'].tolist()
        features = self._features_collection()
        for k, v in data.items():
            features[k] = self._features._from_dict(v, validate=False)
        return features


class MatlabHandler(FeaturesHandler):
    """Saves and loads features to/from the matlab '.mat' format"""
    def _save(self, features, compress=True):
        # represent the features as dictionaries
        data = {k: v._to_dict() for k, v in features.items()}

        # save (and optionally compress) the features
        scipy.io.savemat(
            self.filename, data,
            long_field_names=True,
            appendmat=False, do_compression=compress)

    def _load(self):
        data = scipy.io.loadmat(
            self.filename, appendmat=False, squeeze_me=True,
            mat_dtype=True, struct_as_record=False)

        features = self._features_collection()
        for k, v in data.items():
            if k not in ('__header__', '__version__', '__globals__'):
                features[k] = self._features(
                    v.data, v.times,
                    self._load_properties(v.properties),
                    validate=False)
        return features

    @staticmethod
    def _load_properties(properties):
        return {
            k: v for k, v in properties.__dict__.items()
            if k is not '_fieldnames'}


class JsonHandler(FeaturesHandler):
    """Saves and loads features to/from the JSON format"""
    def _save(self, features):
        data = json.dumps(
            {k: v._to_dict(array_as_list=True) for k, v in features.items()},
            indent=4)
        open(self.filename, 'wt').write(data)

    def _load(self):
        data = json.loads(open(self.filename, 'r').read())
        features = self._features_collection()
        for k, v in data.items():
            features[k] = self._features._from_dict(v, validate=False)
        return features


class H5featuresHandler(FeaturesHandler):
    """Saves and loads features to/from the h5features format"""
    def _save(self, features, groupname='features',
              compression='lzf', chunk_size='auto'):
        data = h5features.Data(
            list(features.keys()),
            [f.times for f in features.values()],
            [f.data for f in features.values()],
            properties=[f.properties for f in features.values()])

        h5features.Writer(
            self.filename,
            mode='w',
            chunk_size=chunk_size,
            compression=compression).write(data, groupname=groupname)

    def _load(self, groupname='features'):
        data = h5features.Reader(self.filename, groupname=groupname).read()

        features = self._features_collection()
        for n in range(len(data.items())):
            features[data.items()[n]] = self._features(
                data.features()[n],
                data.labels()[n],
                properties=data.properties()[n])
        return features


class KaldiHandler(FeaturesHandler):
    def _save(self, features):
        pass

    def _load(self):
        pass
