"""Saves and loads features collections to/from various file formats"""

import abc
import copy
import copyreg
import os
import pickle
import numpy as np
import scipy

import h5features
import json_tricks
import kaldi.matrix
import kaldi.util.table

from shennong import Features
from shennong.utils import array2list, list_files_with_extension


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
        '.pkl': PickleSerializer,
        '.h5f': H5featuresSerializer,
        '.ark': KaldiSerializer,
        '': CsvSerializer
    }


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
        'pickle': PickleSerializer,
        'h5features': H5featuresSerializer,
        'kaldi': KaldiSerializer,
        'csv': CsvSerializer
    }


def get_serializer(cls, filename, log, serializer=None):
    """Returns the file serializer from filename extension or serializer name

    Parameters
    ----------
    cls : class
        Must be :class:`shennong.features.FeaturesCollection`, this is
        a tweak to avoid circular imports
    filename : str
        The file to be handled (load or save)
    log : logging.Logger
        Where to send log messages
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

    return serializer(cls, filename, log)


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
    def __init__(self, cls, filename, log):
        self._features_collection = cls
        self._filename = filename
        self._log = log

        # disable the warning 'numpy serialization is experimental'
        json_tricks.NumpyEncoder.SHOW_SCALAR_WARNING = False

    @property
    def filename(self):
        """Name of the file to read or write"""
        return self._filename

    @abc.abstractmethod
    def _save(self, features, with_times, with_properties):  # pragma: nocover
        pass

    def _check_save(self):
        if os.path.isfile(self.filename):
            raise IOError(f'file already exists: {self.filename}')

    def save(self, features, with_properties=True, **kwargs):
        """Saves a collection of `features` to a file

        Parameters
        ----------
        features : :class:`~shennong.features.FeaturesCollection`
            The features to store in the file.
        with_properties : bool, optional
            When False do not save the features properties, default to True.
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
        self._check_save()

        if not isinstance(features, self._features_collection):
            raise ValueError(
                'features must be {} but are {}'
                .format(
                    self._features_collection.__name__,
                    features.__class__.__name__))

        if not features.is_valid():
            raise ValueError('features are not valid')

        self._save(features, with_properties, **kwargs)

    @abc.abstractmethod
    def _load(self):  # pragma: nocover
        pass

    def _check_load(self):
        if not os.path.isfile(self.filename):
            raise IOError(f'file not found: {self.filename}')
        if not os.access(self.filename, os.R_OK):
            raise IOError(f'file not readable: {self.filename}')

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
        self._check_load()

        features = self._load(**kwargs)

        if not features.is_valid():  # pragma: nocover
            raise ValueError(f'features not valid in "{self.filename}"')

        return features


class NumpySerializer(FeaturesSerializer):
    """Saves and loads features to/from the numpy '.npz' format"""
    def _save(self, features, with_properties, compress=True):
        self._log.info('writing %s', self.filename)

        # represent the features as dictionaries
        data = {
            k: v._to_dict(with_properties=with_properties)
            for k, v in features.items()}

        # save (and optionally compress) the features
        save = np.savez_compressed if compress is True else np.savez
        save(open(self.filename, 'wb'), features=data, allow_pickle=True)

    def _load(self):
        self._log.info('loading %s', self.filename)

        data = np.load(
            open(self.filename, 'rb'), allow_pickle=True)['features'].tolist()

        features = self._features_collection()
        for k, v in data.items():
            features[k] = Features._from_dict(v, validate=False)
        return features


class MatlabSerializer(FeaturesSerializer):
    """Saves and loads features to/from the matlab '.mat' format"""
    def _save(self, features, with_properties, compress=True):
        self._log.info('writing %s', self.filename)

        # represent the features as dictionaries
        data = {
            k: v._to_dict(with_properties=with_properties)
            for k, v in features.items()}

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
                if 'properties' in v:
                    features[k] = Features(
                        v['data'], v['times'],
                        self._make_list(self._check_keys(v['properties'])),
                        validate=False)
                else:
                    features[k] = Features(
                        v['data'], v['times'], validate=False)
        return features

    @classmethod
    def _check_keys(cls, data):
        """Checks if entries in the dictionary `data` are mat-objects.

        If yes todict is called to change them to nested dictionaries.

        From https://stackoverflow.com/a/8832212

        """
        for key in data:
            if isinstance(data[key], scipy.io.matlab.mio5_params.mat_struct):
                data[key] = cls._todict(data[key])
            elif isinstance(data[key], (list, np.ndarray)):
                data[key] = [cls._todict(dd) for dd in data[key]]
        return data

    @staticmethod
    def _todict(matobj):
        """Constructs from matobjects nested dictionaries

        From https://stackoverflow.com/a/8832212

        """
        data = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
                data[strg] = MatlabSerializer._todict(elem)
            else:
                data[strg] = elem
        return data

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


class _NoPropertiesPickler(pickle.Pickler):
    """Implements the with_properties=False for PickleSerializer"""
    dispatch_table = copyreg.dispatch_table.copy()
    dispatch_table[Features] = lambda obj: (
        obj.__class__, (obj.data, obj.times, None, False))


class PickleSerializer(FeaturesSerializer):
    """Saves and loads features to/from the Python pickle format"""
    def _save(self, features, with_properties):
        self._log.info('writing %s', self.filename)
        pickler = pickle.Pickler if with_properties else _NoPropertiesPickler
        with open(self.filename, 'wb') as stream:
            pickler(stream).dump(features)

    def _load(self):
        self._log.info('loading %s', self.filename)
        with open(self.filename, 'rb') as stream:
            return pickle.load(stream)


class H5featuresSerializer(FeaturesSerializer):
    """Saves and loads features to/from the h5features format"""
    def _save(self, features, with_properties, compress=True):
        self._log.info('writing %s', self.filename)

        # we safely use append mode as we are sure at this point the
        # file does not exist (from FeaturesSerializer.save)
        with h5features.Writer(
                self.filename, mode='a', chunk_size='auto',
                compression='lzf' if compress else None) as writer:
            # append the feature in the file one by one (this avoid to
            # duplicate the whole collection in memory, which can
            # cause MemoryError on big datasets).
            for k, v in features.items():
                if with_properties:
                    data = h5features.Data(
                        [k], [v.times], [v.data], properties=[v.properties])
                else:
                    data = h5features.Data([k], [v.times], [v.data])
                writer.write(data, groupname='features', append=True)

    def _load(self):
        self._log.info('loading %s', self.filename)

        data = h5features.Reader(self.filename, groupname='features').read()

        features = self._features_collection()
        for n in range(len(data.items())):
            features[data.items()[n]] = Features(
                data.features()[n],
                data.labels()[n],
                properties=(
                    data.properties()[n] if data.has_properties() else {}),
                validate=False)

        return features


class KaldiSerializer(FeaturesSerializer):
    """Saves and loads features to/from the Kaldi ark/scp format"""
    def __init__(self, cls, filename, log):
        super().__init__(cls, filename, log=log)

        # make sure the filename extension is '.ark'
        filename_split = os.path.splitext(self.filename)
        if filename_split[1] != '.ark':
            raise ValueError(
                'when saving to Kaldi ark format, the file extension must be '
                '".ark", it is "{}"'.format(filename_split[1]))

        self._fileroot = filename_split[0]

    def _save(self, features, with_properties, scp=False):
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
                # 2d->1d on loading). We are copying the array to
                # avoid a bug on macos.
                writer[k] = kaldi.matrix.DoubleSubMatrix(
                    np.atleast_2d(v.times).copy())

        # writing properties. As we are writing double arrays, we need
        # to track the original dtype of features in the properties,
        # to ensure equality on load
        filename = self._fileroot + '.properties.json'
        self._log.info('writing %s', filename)
        if with_properties:
            data = {
                k: copy.deepcopy(v.properties) for k, v in features.items()}
        else:
            data = {k: {} for k in features}

        for k in data:
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
        for key, value in times.items():
            if value.shape[0] == 1:
                times[key] = value.reshape((value.shape[1]))

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
            **{k: Features(
                data[k].astype(properties[k]['__dtype_data__']),
                times[k].astype(properties[k]['__dtype_times__']),
                properties={
                    k: p for k, p in properties[k].items()
                    if '__dtype_' not in k},
                validate=False)
               for k in data.keys()})


class CsvSerializer(FeaturesSerializer):
    """Saves and loads features to/from the CSV format"""
    def _check_load(self):
        if not os.path.isdir(self.filename):
            raise IOError(f'directory not found: {self.filename}')

    def _check_save(self):
        if os.path.exists(self.filename):
            raise IOError(f'already exists: {self.filename}')

    def _save(self, features, with_properties):
        # save one csv/json pair per features in a directory (csv for
        # data/times, json for properties)
        os.makedirs(self.filename)
        self._log.info('writing directory "%s"', self.filename)
        for name, feat in features.items():
            # save data and times into the csv. We need to append the features
            # dimension in the properties to rebuild the features on load,
            # because times can be 1d or 2d.
            csv_file = os.path.join(self.filename, name + '.csv')
            self._log.debug('writing %s', csv_file)
            np.savetxt(
                csv_file,
                np.hstack((
                    feat.times.reshape((feat.nframes, 1))
                    if feat.times.ndim == 1 else feat.times,
                    feat.data)),
                header=(
                    f'data_dtype = {feat.dtype}, '
                    f'times_dtype = {feat.times.dtype}, '
                    f'features_ndims = {feat.ndims}'),
                comments='# ')

            # if any, save properties into a json file
            if with_properties and feat.properties:
                json_file = os.path.join(self.filename, name + '.json')
                self._log.debug('writing %s', json_file)
                open(json_file, 'wt').write(
                    json_tricks.dumps(feat.properties, indent=4))

    @staticmethod
    def _parse_header(csv_file):
        header = open(csv_file, 'r').readline().strip()
        if header[0] != '#':
            raise ValueError(f'failed to parse header from {csv_file}')
        header = header.split(', ')

        try:
            data_dtype = np.dtype(header[0].split('= ')[1])
            times_dtype = np.dtype(header[1].split('= ')[1])
            ndims = int(header[2].split('= ')[1])
        except (IndexError, TypeError):
            raise ValueError(f'failed to parse header from {csv_file}')

        return data_dtype, times_dtype, ndims

    def _load(self):
        self._log.info('loading directory "%s"', self.filename)

        # list all the csv and json files
        csv_files = list_files_with_extension(
            self.filename, '.csv', recursive=False)
        json_files = list_files_with_extension(
            self.filename, '.json', recursive=False)

        features = self._features_collection()

        # load the features one by one
        for csv in csv_files:
            self._log.debug('loading %s', csv)

            data_dtype, times_dtype, ndims = self._parse_header(csv)

            # read times and features
            data = np.loadtxt(csv)
            times = data[:, :data.shape[1] - ndims].astype(times_dtype)
            if times.shape[1] == 1:
                times = times.flatten()
            data = data[:, data.shape[1] - ndims:].astype(data_dtype)

            # read properties
            properties = {}
            json = csv.replace('.csv', '.json')
            if json in json_files:
                self._log.debug('loading %s', json)
                properties = dict(json_tricks.loads(open(json, 'r').read()))

            # build the features
            name = os.path.basename(csv).replace('.csv', '')
            features[name] = Features(
                data, times, properties=properties, validate=False)

        return features
