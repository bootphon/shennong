"""Saves and loads features to/from various file formats"""

import abc
import json
import os

import numpy as np
import scipy

from shennong.features import Features, FeaturesCollection


class FeaturesHandler(metaclass=abc.ABCMeta):
    """Base class of a features file handler

    This class must be specialized to handle a given file type.

    """
    def __init__(self, filename, append_ext=False):
        self._filename = filename
        if append_ext and not self._filename.endswith(self._extension()):
            self._filename += self._extension()

    @property
    def filename(self):
        return self._filename

    @abc.abstractstaticmethod
    def _extension():
        pass

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

        if not isinstance(features, FeaturesCollection):
            raise ValueError(
                'features must be FeaturesCollection but are {}'
                .format(type(features)))

        if not features.is_valid():
            raise ValueError('features are not valid')

        self._save(features)


class NumpyHandler(FeaturesHandler):
    """Saves and loads features to/from the numpy '.npz' format"""
    @staticmethod
    def _extension():
        return '.npz'

    def _save(self, features, compress=True):
        # represent the features as dictionaries
        data = {k: v._to_dict() for k, v in features.items()}

        # save (and optionally compress) the features
        save = np.savez_compressed if compress is True else np.savez
        save(open(self.filename, 'wb'), features=data)

    def _load(self):
        data = np.load(open(self.filename, 'rb'))['features'].tolist()
        features = FeaturesCollection()
        for k, v in data.items():
            features[k] = Features._from_dict(v, validate=False)
        return features


class MatlabHandler(FeaturesHandler):
    """Saves and loads features to/from the matlab '.mat' format"""
    @staticmethod
    def _extension():
        return '.mat'

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

        features = FeaturesCollection()
        for k, v in data.items():
            if k not in ('__header__', '__version__', '__globals__'):
                features[k] = Features(
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
    @staticmethod
    def _extension():
        return '.json'

    def _save(self, features):
        data = json.dumps(
            {k: v._to_dict(array_as_list=True) for k, v in features.items()},
            indent=4)
        open(self.filename, 'wt').write(data)

    def _load(self):
        data = json.loads(open(self.filename, 'r').read())
        features = FeaturesCollection()
        for k, v in data.items():
            features[k] = Features._from_dict(v, validate=False)
        return features


class H5featuresHandler(FeaturesHandler):
    @staticmethod
    def _extension():
        return '.h5f'

    def _save(self, features):
        pass

    def _load(self):
        pass


class KadliHandler(FeaturesHandler):
    @staticmethod
    def _extension():
        return '.ark'

    def _save(self, features):
        pass

    def _load(self):
        pass
