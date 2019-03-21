"""Computes cepstral mean variance normalization (CMVN)

The :class:`CmvnProcessor` class is used for accumulating CMVN
statistics and applying CMVN on features using accumulated
statistics. Uses the Kaldi implementation (see [kaldi-cmvn]_):

    :class:`Features` --> CmvnProcessor --> :class:`Features`

References
----------

.. [kaldi-cmvn] https://kaldi-asr.org/doc/transform.html#transform_cmvn

"""

import copy
import numpy as np
import kaldi.matrix
import kaldi.transform.cmvn

from shennong.features.postprocessor.base import FeaturesPostProcessor
from shennong.features import Features, FeaturesCollection


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

    def accumulate(self, features, weights=None):
        """Accumulates CMVN statistics

        Computes the CMVN statistics for the given ``features`` and
        accumulates them for further processing.

        Parameters
        ----------
        features : :class:`~shennong.features.features.Features`
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

        prop = copy.deepcopy(features.properties)
        prop['cmvn'] = self.stats
        return Features(data.numpy(), features.times, properties=prop)


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
            w = weights[k] if weights is not None else None
            cmvn.accumulate(f, weights=w)

        # apply CMVN stats
        return FeaturesCollection(
            {k: cmvn.process(f, norm_vars=norm_vars, skip_dims=skip_dims)
             for k, f in feats_collection.items()})
    else:
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
