"""Saves and loads features collections to/from various file formats"""

import abc
import copy
import os
import pickle

import h5features
import json_tricks
import kaldi.matrix
import kaldi.util.table
import numpy as np
import scipy

from shennong.utils import get_logger


def supported_extensions():
    """Returns the list of file extensions to save/load features

    Returns
    -------
    serializers : dict
        File extensions mapped to their related serializer class

    """
    return {
        '.npz': NumpySerializer,
        '.mat': MatlabSerializer,
        '.json': JsonSerializer,
        '.pkl': PickleSerializer,
        '.h5f': H5featuresSerializer,
        '.ark': KaldiSerializer}


def supported_serializers():
    """Returns the list of file format serializers to save/load features

    Returns
    -------
    serializers : dict
        Serializers names mapped to their related class

    """
    return {
        'numpy': NumpySerializer,
        'matlab': MatlabSerializer,
        'json': JsonSerializer,
        'pickle': PickleSerializer,
        'h5features': H5featuresSerializer,
        'kaldi': KaldiSerializer}


def get_serializer(cls, filename, serializer=None):
    """Returns the file serializer from filename extension or serializer name

    Parameters
    ----------
    cls : class
        Must be :class:`shennong.features.FeaturesCollection`, this is
        a tweak to avoid circular imports
    filename : str
        The file to be handled (load or save)
    serializer : str, optional
        If not None must be one of the :func:`supported_serializers`, if
        not specified, guess the serializer from the `filename`
        extension using :func:`supported_extensions`.

    Returns
    -------
    serializer : instance of :class:`FeaturesSerializer`
        The guessed serializer class, a child class of
        :class:`FeaturesSerializer`.

    Raises
    ------
    ValueError
        If the serializer class cannot be guessed, or if `cls` is not
        :class:`~shennong.features.FeaturesCollection`

    """
    if cls.__name__ != 'FeaturesCollection':
        raise ValueError(
            'The `cls` parameter must be shennong.features.FeaturesCollection')

    if serializer is None:
        # guess serializer from file extension
        ext = os.path.splitext(filename)[1]
        if not ext:
            raise ValueError('no extension nor serializer name specified')

        try:
            serializer = supported_extensions()[ext]
        except KeyError:
            raise ValueError(
                'invalid extension {}, must be in {}'.format(
                    ext, list(supported_extensions().keys())))
    else:
        try:
            serializer = supported_serializers()[serializer]
        except KeyError:
            raise ValueError(
                'invalid serializer {}, must be in {}'.format(
                    serializer, list(supported_serializers().keys())))

    return serializer(cls, filename)


class FeaturesSerializer(metaclass=abc.ABCMeta):
    """Base class of a features file serializer

    This class must be specialized to handle a given file type.

    Parameters
    ----------
    cls : class
        Must be :class:`shennong.features.FeaturesCollection`, this is
        a tweak to avoid circular imports
    filename : str
        The file to save/load features to/from

    """
    _log = get_logger(__name__)

    def __init__(self, cls, filename):
        self._features_collection = cls
        self._features = self._features_collection._value_type
        self._filename = filename

    @property
    def filename(self):
        return self._filename

    @abc.abstractmethod
    def _save(self, features):  # pragma: nocover
        pass

    @abc.abstractmethod
    def _load(self):  # pragma: nocover
        pass

    def load(self, **kwargs):
        """Returns a collection of features from the `filename`

        Returns
        -------
        features : :class:`~shennong.features.FeaturesCollection`
            The features stored in the file.
        kwargs : optional
            Optional supplementary arguments, specific to each serializer.

        Raises
        ------
        IOError
            If the input file does not exist or cannot be read.

        ValueError
            If the features cannot be loaded from the file or are not
            in a valid state.

        """
        if not os.path.isfile(self.filename):
            raise IOError('file not found: {}'.format(self.filename))
        if not os.access(self.filename, os.R_OK):
            raise IOError('file not readable: {}'.format(self.filename))

        features = self._load(**kwargs)

        if not features.is_valid():
            raise ValueError(
                'features not valid in file: {}'.format(self.filename))

        return features

    def save(self, features, **kwargs):
        """Saves a collection of `features` to a file

        Parameters
        ----------
        features : :class:`~shennong.features.FeaturesCollection`
            The features to store in the file.
        kwargs : optional
            Optional supplementary arguments, specific to each serializer.

        Raises
        ------
        IOError
            If the output file already exists.

        ValueError
            If the features cannot be saved to the file, are not in a
            valid state or are not an instance of
            :class:`~shennong.features.FeaturesCollection`.

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

        self._save(features, **kwargs)


class NumpySerializer(FeaturesSerializer):
    """Saves and loads features to/from the numpy '.npz' format"""
    def _save(self, features, compress=True):
        self._log.info('writing %s', self.filename)

        # represent the features as dictionaries
        data = {k: v._to_dict() for k, v in features.items()}

        # save (and optionally compress) the features
        save = np.savez_compressed if compress is True else np.savez
        save(open(self.filename, 'wb'), features=data)

    def _load(self):
        self._log.info('loading %s', self.filename)

        data = np.load(open(self.filename, 'rb'))['features'].tolist()

        features = self._features_collection()
        for k, v in data.items():
            features[k] = self._features._from_dict(v, validate=False)
        return features


class MatlabSerializer(FeaturesSerializer):
    """Saves and loads features to/from the matlab '.mat' format"""
    def _save(self, features, compress=True):
        self._log.info('writing %s', self.filename)

        # represent the features as dictionaries
        data = {k: v._to_dict() for k, v in features.items()}

        # save (and optionally compress) the features
        scipy.io.savemat(
            self.filename, data,
            long_field_names=True,
            appendmat=False, do_compression=compress)

    def _load(self):
        self._log.info('loading %s', self.filename)

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


class JsonSerializer(FeaturesSerializer):
    """Saves and loads features to/from the JSON format"""
    def _save(self, features):
        self._log.info('writing %s', self.filename)
        open(self.filename, 'wt').write(json_tricks.dumps(features, indent=4))

    def _load(self):
        self._log.info('loading %s', self.filename)
        return self._features_collection(
            json_tricks.loads(open(self.filename, 'r').read()))


class PickleSerializer(FeaturesSerializer):
    """Saves and loads features to/from the Python pickle format"""
    def _save(self, features):
        self._log.info('writing %s', self.filename)
        with open(self.filename, 'wb') as fh:
            pickle.dump(features, fh)

    def _load(self):
        with open(self.filename, 'rb') as fh:
            return pickle.load(fh)


class H5featuresSerializer(FeaturesSerializer):
    """Saves and loads features to/from the h5features format"""
    def _save(self, features, groupname='features',
              compression='lzf', chunk_size='auto'):
        self._log.info('writing %s', self.filename)

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
        self._log.info('loading %s', self.filename)

        data = h5features.Reader(self.filename, groupname=groupname).read()

        features = self._features_collection()
        for n in range(len(data.items())):
            features[data.items()[n]] = self._features(
                data.features()[n],
                data.labels()[n],
                properties=data.properties()[n],
                validate=False)
        return features


class KaldiSerializer(FeaturesSerializer):
    def __init__(self, cls, filename):
        super().__init__(cls, filename)

        # make sure the filename extension is '.ark'
        filename_split = os.path.splitext(self.filename)
        if filename_split[1] != '.ark':
            raise ValueError(
                'when saving to Kaldi ark format, the file extension must be '
                '".ark", it is "{}"'.format(filename_split[1]))

        self._fileroot = filename_split[0]

    def _save(self, features, scp=False):
        # writing features
        ark = self._fileroot + '.ark'
        if scp:
            scp = self._fileroot + '.scp'
            self._log.info('writing %s and %s', ark, scp)
            wspecifier = 'ark,scp:' + ark + ',' + scp
        else:
            self._log.info('writing %s', ark)
            wspecifier = 'ark:' + ark
        with kaldi.util.table.DoubleMatrixWriter(wspecifier) as writer:
            for k, v in features.items():
                writer[k] = kaldi.matrix.DoubleSubMatrix(v.data)

        # writing times
        ark = self._fileroot + '.times.ark'
        if scp:
            scp = self._fileroot + '.times.scp'
            self._log.info('writing %s and %s', ark, scp)
            wspecifier = 'ark,scp:' + ark + ',' + scp
        else:
            self._log.info('writing %s', ark)
            wspecifier = 'ark:' + ark
        with kaldi.util.table.DoubleVectorWriter(wspecifier) as writer:
            for k, v in features.items():
                writer[k] = kaldi.matrix.DoubleSubVector(v.times)

        # writing properties. As we are writing double arrays, we need
        # to track the original dtype of features in the properties,
        # to ensure equality on load
        filename = self._fileroot + '.properties.json'
        self._log.info('writing %s', filename)
        data = {k: copy.deepcopy(v.properties) for k, v in features.items()}
        for k, v in data.items():
            data[k]['__dtype_data__'] = str(features[k].dtype)
            data[k]['__dtype_times__'] = str(features[k].times.dtype)
        open(filename, 'wt').write(json_tricks.dumps(data, indent=4))

    def _load(self):
        # loading properties
        filename = self._fileroot + '.properties.json'
        self._log.info('loading %s', filename)
        if not os.path.isfile(filename):
            raise IOError('file not found: {}'.format(filename))

        properties = json_tricks.loads(open(filename, 'r').read())

        # loading features
        ark = self._fileroot + '.ark'
        self._log.info('loading %s', ark)

        # rspecifier = 'ark,scp:' + ark + ',' + scp
        rspecifier = 'ark:' + ark
        with kaldi.util.table.SequentialDoubleMatrixReader(
                rspecifier) as reader:
            data = {k: v.numpy() for k, v in reader}

        if properties.keys() != data.keys():
            raise ValueError(
                'invalid features: items differ in data and properties')

        # loading times
        ark = self._fileroot + '.times.ark'
        self._log.info('loading %s', ark)
        if not os.path.isfile(ark):
            raise IOError('file not found: {}'.format(ark))

        rspecifier = 'ark:' + ark
        with kaldi.util.table.SequentialDoubleVectorReader(
                rspecifier) as reader:
            times = {k: v.numpy() for k, v in reader}

        if times.keys() != data.keys():
            raise ValueError(
                'invalid features: items differ in data and times')

        return self._features_collection(
            **{k: self._features(
                data[k].astype(properties[k]['__dtype_data__']),
                times[k].astype(properties[k]['__dtype_times__']),
                properties={
                    k: p for k, p in properties[k].items()
                    if '__dtype_' not in k},
                validate=False)
               for k in data.keys()})
