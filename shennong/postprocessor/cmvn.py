"""Cepstral mean variance normalization (CMVN) on speech features

* The :class:`CmvnPostProcessor` class is used for accumulating CMVN
  statistics and applying CMVN on features using accumulated
  statistics. Uses the Kaldi implementation (see [kaldi-cmvn]_):

      :class:`Features` --> CmvnPostProcessor --> :class:`Features`

* The :class:`SlidingWindowCmvnPostProcessor` class is used to apply sliding
  window CMVN. With that class, each window is normalized independantly. Uses
  the Kaldi implementation:

      :class:`Features` --> SlidingWindowCmvnPostProcessor
      --> :class:`Features`

Examples
--------

Compute MFCC features:

>>> import numpy as np
>>> from shennong.audio import Audio
>>> from shennong.processor.mfcc import MfccProcessor
>>> from shennong.postprocessor.cmvn import CmvnPostProcessor
>>> audio = Audio.load('./test/data/test.wav')
>>> mfcc = MfccProcessor(sample_rate=audio.sample_rate).process(audio)

Accumulate CMVN statistics and normalize the features (in real life
you want to accumulate statistics over several features, for example
on all features belonging to one speaker, so as to obtain a
normalization per speaker):

>>> processor = CmvnPostProcessor(mfcc.ndims)
>>> processor.accumulate(mfcc)
>>> cmvn = processor.process(mfcc)

The normalized features have a zero mean and unitary variance:

>>> np.all(np.isclose(cmvn.data.mean(axis=0), np.zeros(cmvn.ndims), atol=1e-6))
True
>>> np.all(np.isclose(cmvn.data.var(axis=0), np.ones(cmvn.ndims), atol=1e-6))
True

This module also provides a high-level method for applying CMVN to a
whole :class:`~shennong.features_collection.FeaturesCollection` at once:

>>> from shennong import FeaturesCollection
>>> from shennong.postprocessor.cmvn import apply_cmvn
>>> feats = FeaturesCollection(utt1=mfcc)
>>> cmvns = apply_cmvn(feats)

As above, the features has zero mean and unitary variance

>>> cmvn = cmvns['utt1']
>>> np.all(np.isclose(cmvn.data.mean(axis=0), np.zeros(cmvn.ndims), atol=1e-6))
True
>>> np.all(np.isclose(cmvn.data.var(axis=0), np.ones(cmvn.ndims), atol=1e-6))
True

Apply sliding-window normalization to the features:

>>> from shennong.postprocessor.cmvn import SlidingWindowCmvnPostProcessor
>>> processor = SlidingWindowCmvnPostProcessor(normalize_variance=True)
>>> window_size = 40
>>> processor.cmn_window = window_size
>>> processor.min_window = window_size
>>> sliding_cmvn = processor.process(mfcc)

Each frame of the original features has been normalized with statistics
computed in the window:

>>> frame = 70
>>> window = mfcc.data[frame-window_size//2:frame+window_size//2, :]
>>> norm_mfcc = (mfcc.data[frame,:] - window.mean(axis=0)) / window.std(axis=0)
>>> np.all(np.isclose(sliding_cmvn.data[frame, :], norm_mfcc, atol=1e-6))
True

References
----------

.. [kaldi-cmvn] https://kaldi-asr.org/doc/transform.html#transform_cmvn

"""

import copy
import numpy as np
import kaldi.matrix
import kaldi.transform.cmvn
import kaldi.feat.functions

from shennong.postprocessor.base import FeaturesPostProcessor
from shennong import Features, FeaturesCollection


class CmvnPostProcessor(FeaturesPostProcessor):
    """Computes CMVN statistics on speech features

    Parameters
    ----------
    dim : int
        The features dimension, must be strictly positive

    stats : array, shape = [2, dim+1]
        Preaccumulated CMVN statistics (see :func:`CmvnPostProcessor:stats`)

    Raises
    ------
    ValueError
        If ``dim`` is not a strictly positive integer

    """

    def __init__(self, dim, stats=None):
        super().__init__()

        # init features dimension
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(
                'dimension must be a strictly positive integer, it is {}'
                .format(dim))
        self._dim = dim

        # init the pykaldi cmvn class
        self._cmvn = kaldi.transform.cmvn.Cmvn(dim=dim)

        # init the stats if specified
        if stats is not None:
            stats = np.asarray(stats)
            if stats.shape != (2, self.dim+1):
                raise ValueError(
                    'stats must be an array of shape {}, but is shaped as {}'
                    .format((2, self.dim+1), stats.shape))
            self._cmvn.stats = kaldi.matrix.SubMatrix(stats)

    @property
    def name(self):
        return 'cmvn'

    @property
    def dim(self):
        """The dimension of features on which to compute CMVN"""
        return self._dim

    @property
    def stats(self):
        """The accumulated CMVN statistics

        Array of shape `[2, dim+1]` with the following format:

        * ``stats[0, :]`` represents the sum of accumulated feature
          frames, used to estimate the accumulated mean.

        * ``stats[1, :]`` represents the sum of element-wise squares
          of accumulated feature frames, used to estimate the
          accumulated variance.

        * ``stats[0, -1]`` represents the weighted total count of
          accumulated feature frames.

        * ``stats[1, -1]`` is initialized to zero but otherwise is not
          used.

        """
        return self._cmvn.stats.numpy()

    @property
    def count(self):
        """The weighted total count of accumulated features frames"""
        return self.stats[0, -1]

    @property
    def ndims(self):
        return self.dim

    def get_properties(self, features):
        properties = super().get_properties(features)
        properties[self.name]['stats'] = self.stats
        return properties

    def accumulate(self, features, weights=None):
        """Accumulates CMVN statistics

        Computes the CMVN statistics for the given ``features`` and
        accumulates them for further processing.

        Parameters
        ----------
        features : :class:`~shennong.features.Features`
            The input features on which to accumulate statisitics.

        weights : array, shape = [``features.nframes``, 1], optional
            Weights to apply to each frame of the features (possibly
            zero to ignore silences or non-speech
            frames). Accumulation is non-weighted by default.

        Raises
        ------
        ValueError
            If ``weights`` have more than one dimension or if
            ``weights`` length does not fit ``features`` dimension.

        """
        # make sure weights have the expected dimension
        if weights is not None:
            if weights.ndim != 1:
                raise ValueError(
                    'weights must have a single dimension but have {}'
                    .format(weights.ndim))
            if weights.shape[0] != features.nframes:
                raise ValueError(
                    'there is {} weights but {} feature frames, must be equal'
                    .format(weights.shape[0], features.nframes))

            weights = kaldi.matrix.SubVector(weights)

        # delegate to pykaldi implementation
        self._cmvn.accumulate(
            kaldi.matrix.SubMatrix(features.data),
            weights=weights)

    def process(self, features, norm_vars=True, skip_dims=None, reverse=False):
        """Applies the accumulated CMVN statistics to the given ``features``

        Parameters
        ----------
        features : :class:`~shennong.features.features.Features`
            The input features on which to apply CMVN statisitics.

        norm_vars : bool, optional
            If False, do not apply variance normalization (only mean),
            default to True.

        skip_dims : list of positive integers, optional
            Dimensions for which to skip normalization. Default is to
            not skip any dimension.

        reverse : bool, optional
            Whether to apply CMVN in a reverse sense, so as to
            transform zero-mean, unit-variance features into features
            with the desired mean and variance.

        Returns
        -------
        cmvn_features : :class:`~shennong.features.features.Features`
            The normalized features

        Raises
        ------
        ValueError
            If no stats have been accumulated

        """
        # make sure we have accumulated some stats
        if self.count < 1.0:
            raise ValueError(
                'insufficient accumulation of stats for CMVN, '
                'must be >= 1.0 but is {}'.format(self.count))

        # skip dims in pykaldi is a destructive operation (alteration
        # of self.stats), so we work by copy here, to avoid modifying
        # statistics.
        if not skip_dims:
            cmvn = self._cmvn
        else:
            # make sure all skipped dims are valid dims
            dmin, dmax = min(skip_dims), max(skip_dims)
            if dmin < 0 or dmax >= features.ndims:
                raise ValueError(
                    'skipped dimensions must be in [0, {}[ but are in [{}, {}['
                    .format(features.ndims, dmin, dmax))

            # work by copy to not alter self.stats
            cmvn = kaldi.transform.cmvn.Cmvn(dim=self.dim)
            cmvn.stats = kaldi.matrix.DoubleMatrix(self.stats)
            cmvn.skip_dims(skip_dims)

        data = kaldi.matrix.SubMatrix(features.data)
        cmvn.apply(data, norm_vars=norm_vars, reverse=reverse)

        return Features(
            data.numpy(), features.times,
            properties=self.get_properties(features))


def apply_cmvn(feats_collection, by_collection=True, norm_vars=True,
               weights=None, skip_dims=None):
    """CMVN normalization of a collection of features

    This function is a simple wrapper on the class
    :class:`~shennong.features.CmvnPostProcessor` that allows to
    accumulate and apply CMVN statistics over a whole collections of
    features.

    Warnings
    --------
    The features in the collection must have the same
    dimensionality. It is assumed they are all extracted from the same
    processor. If this is not the case, a ValueError is raised.

    Parameters
    ----------
    feats_collection : :class:`~shennong.FeaturesCollection`
        The collection of features on wich to apply CMVN normlization.
        Each features in the collection is assumed to have consistent
        dimensions.

    by_collection : bool, optional
        When True, accumulate and apply CMVN over the entire
        collection. When False, do it independently for each features
        in the collection. Default to True.

    norm_vars : bool, optional
        If False, do not apply variance normalization (only mean),
        default to True.

    weights : dict of arrays, optional
        For each features in the collection, an array of weights to
        apply on the features frames, if specified we must have
        ``weights.keys() == feats_collections.keys()`` (see
        :func:`CmvnPostProcessor.accumulate`). Unweighted by default.

    skip_dims : list of integers
        The dimensions for which to skip the normalization (see
        :func:`CmvnPostProcessor.process`). Default is to normalize
        all dimensions.

    Returns
    -------
    cmvn_feats_collection : :class:`~shennong.features.FeaturesCollection`

    Raises
    ------
    ValueError
        If something goes wrong during CMVN processing.

    """
    # extract the features dimension
    dim = set(f.ndims for f in feats_collection.values())
    if not len(dim) == 1:
        raise ValueError(
            'features in the collection must have consistent dimensions '
            'but dimensions are: {}'.format(sorted(dim)))
    dim = list(dim)[0]

    # check weights
    if weights is not None and weights.keys() != feats_collection.keys():
        raise ValueError('keys differ for weights and features collection')

    # check skip_dims
    if skip_dims is not None:
        sdmin, sdmax = min(skip_dims), max(skip_dims)
        if sdmin < 0 or sdmax >= dim:
            raise ValueError(
                'out of bounds dimensions in skip_dims, must be in [0, {}] '
                'but are in [{}, {}]'.format(dim-1, sdmin, sdmax))

    if by_collection:
        # accumulate CMVN stats over the whole collection
        cmvn = CmvnPostProcessor(dim)
        for k, f in feats_collection.items():
            cmvn.accumulate(
                f, weights=weights[k] if weights is not None else None)

        # apply CMVN stats
        return FeaturesCollection(
            {k: cmvn.process(f, norm_vars=norm_vars, skip_dims=skip_dims)
             for k, f in feats_collection.items()})

    # independently for each features in the collection,
    # accumulate and apply CMNV stats
    cmvn_collection = FeaturesCollection()
    for k, f in feats_collection.items():
        cmvn = CmvnPostProcessor(f.ndims)
        cmvn.accumulate(
            f, weights=weights[k] if weights is not None else None)
        cmvn_collection[k] = cmvn.process(
            f, norm_vars=norm_vars, skip_dims=skip_dims)

    return cmvn_collection


class SlidingWindowCmvnPostProcessor(FeaturesPostProcessor):
    """Compute sliding-window normalization on speech features

    Parameters
    ----------
    center : bool, optional
        Whether to center the window on the current frame, default to True
    cmn_window : int, optional
        Window size for average CMN computation, default to 600
    min_window : int, optional
        Minimum CMN window used at start of decoding, default to 100
    max_warnings : int, optional
        Maximum warning to report per utterance, default to 5
    normalize_variance : bool, optional
        Whether to normalize variance to one, default to False

    """
    def __init__(self, center=True, cmn_window=600, min_window=100,
                 max_warnings=5, normalize_variance=False):
        super().__init__()

        self._options = kaldi.feat.functions.SlidingWindowCmnOptions()
        self.center = center
        self.cmn_window = cmn_window
        self.max_warnings = max_warnings
        self.min_window = min_window
        self.normalize_variance = normalize_variance

    @property
    def name(self):
        return 'sliding_window_cmvn'

    @property
    def ndims(self):
        raise ValueError('output dimension for sliding '
                         'window CMVN processor depends on input')

    @property
    def center(self):
        """Whether to center the window on the current frame"""
        return self._options.center

    @center.setter
    def center(self, value):
        self._options.center = value

    @property
    def cmn_window(self):
        """Window size for average CMN computation"""
        return self._options.cmn_window

    @cmn_window.setter
    def cmn_window(self, value):
        self._options.cmn_window = value

    @property
    def min_window(self):
        """Minimum CMN window used at start of decoding"""
        return self._options.min_window

    @min_window.setter
    def min_window(self, value):
        self._options.min_window = value

    @property
    def max_warnings(self):
        """Maximum warning to report per utterance"""
        return self._options.max_warnings

    @max_warnings.setter
    def max_warnings(self, value):
        self._options.max_warnings = value

    @property
    def normalize_variance(self):
        """Whether to normalize variance to one"""
        return self._options.normalize_variance

    @normalize_variance.setter
    def normalize_variance(self, value):
        self._options.normalize_variance = value

    def get_properties(self, features):
        properties = copy.deepcopy(features.properties)
        properties[self.name] = self.get_params()

        if 'pipeline' not in properties:
            properties['pipeline'] = []

        properties['pipeline'].append({
            'name': self.name,
            'columns': [0, features.ndims - 1]})

        return properties

    def process(self, features):
        """Applies sliding-window cepstral mean and/or variance normalization
        on `features` with the specified options

        Parameters
        ----------
        features : :class:`~shennong.features.Features`
            The input features.

        Returns
        -------
        slid_window_cmvn_feats : :class:`~shennong.features.Features`
            The normalized features.
        """
        data = kaldi.matrix.Matrix(*features.data.shape)
        kaldi.feat.functions.sliding_window_cmn(
            self._options, kaldi.matrix.SubMatrix(features.data), data)

        return Features(
            data.numpy(),
            features.times,
            self.get_properties(features))
