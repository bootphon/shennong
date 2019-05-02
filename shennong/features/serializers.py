"""Saves and loads features collections to/from various file formats

The following table shows the obtained file size, writing and reading
times on MFCC features computed on the `Zero Resource Speech
Challenge 2019 <https://zerospeech.com/2019>`_ train database
(English, about 26 hours of speech and 10k files):

===========  =========  =========  ============  ============
File format  Extension  File size  Writing time  Reading time
===========  =========  =========  ============  ============
h5features   .h5f       562.9 MB   0:00:20       0:00:08
pickle       .pkl       609.8 MB   0:00:08       0:00:06
numpy        .npz       582.8 MB   0:02:07       0:00:19
matlab       .mat       481.8 MB   0:00:58       0:00:13
kaldi        .ark       927.8 MB   0:00:10       0:00:15
JSON         .json      6.3 GB     0:11:34       1:04:25
===========  =========  =========  ============  ============


"""

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

from shennong.utils import get_logger, array2list


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
        save(open(self.filename, 'wb'), features=data, allow_pickle=True)

    def _load(self):
        self._log.info('loading %s', self.filename)

        data = np.load(
            open(self.filename, 'rb'), allow_pickle=True)['features'].tolist()

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
        # print(data['test']['properties'])

        # save (and optionally compress) the features
        scipy.io.savemat(
            self.filename, data,
            long_field_names=True,
            appendmat=False, do_compression=compress)

    def _load(self):
        self._log.info('loading %s', self.filename)

        data = self._check_keys(
            scipy.io.loadmat(
                self.filename, appendmat=False, squeeze_me=True,
                mat_dtype=True, struct_as_record=False))

        features = self._features_collection()
        for k, v in data.items():
            if k not in ('__header__', '__version__', '__globals__'):
                features[k] = self._features(
                    v['data'],
                    v['times'],
                    self._make_list(self._check_keys(v['properties'])),
                    validate=False)
        return features

    @staticmethod
    def _check_keys(d):
        """Checks if entries in dictionary are mat-objects.

        If yes todict is called to change them to nested dictionaries.

        From https://stackoverflow.com/a/8832212

        """
        for key in d:
            if isinstance(d[key], scipy.io.matlab.mio5_params.mat_struct):
                d[key] = MatlabSerializer._todict(d[key])
            elif isinstance(d[key], (list, np.ndarray)):
                d[key] = [MatlabSerializer._todict(dd) for dd in d[key]]
        return d

    @staticmethod
    def _todict(matobj):
        """Constructs from matobjects nested dictionaries

        From https://stackoverflow.com/a/8832212

        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
                d[strg] = MatlabSerializer._todict(elem)
            else:
                d[strg] = elem
        return d

    @staticmethod
    def _make_list(properties):
        if 'pipeline' in properties:
            # matlab format collapse a list of a single element into
            # that element, we need to rebuild that list here
            if isinstance(properties['pipeline'], list):
                properties['pipeline'] = [
                    array2list(p) for p in properties['pipeline']]
            else:
                properties['pipeline'] = [
                    array2list(properties['pipeline'])]
        return properties


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
        with kaldi.util.table.DoubleMatrixWriter(wspecifier) as writer:
            for k, v in features.items():
                # in case times are 1d, we force them to 2d so they
                # can be wrote as kaldi matrices (we do the reverse
                # 2d->1d on loading)
                writer[k] = kaldi.matrix.DoubleSubMatrix(
                    np.atleast_2d(v.times))

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

        # loading times
        ark = self._fileroot + '.times.ark'
        self._log.info('loading %s', ark)
        if not os.path.isfile(ark):
            raise IOError('file not found: {}'.format(ark))

        rspecifier = 'ark:' + ark
        with kaldi.util.table.SequentialDoubleMatrixReader(
                rspecifier) as reader:
            times = {k: v.numpy() for k, v in reader}

        # postprocess times: do 2d->1d if they are 1d vectors
        for k, v in times.items():
            if v.shape[0] == 1:
                times[k] = v.reshape((v.shape[1]))

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
