"""Provides the DiagUbmProcessor class to train a Universal Background Model

- Gaussian Mixture Model (UBM-GMM) with diagonal covariances.
- Uses the kaldi implementation of GMM (see [kaldi-gmm]_).

The UBM is used as a preprocessing step by
:class:`~shennong.processor.vtln.VtlnProcessor`.

Examples
--------

>>> from shennong.processor.ubm import DiagUbmProcessor
>>> wav = './test/data/test.wav'
>>> utterances = [('utt1', wav, 'spk1', 0, 1), ('utt2', wav, 'spk1', 1, 1.5)]

Initialize the UBM-GMM with a given number of gaussians. Other options
can be specified at construction, or after:

>>> num_gauss = 4
>>> ubm = DiagUbmProcessor(num_gauss, num_iters_init=10)
>>> ubm.num_iters = 3

Process the utterances to update the model.

>>> ubm.process(utterances)

Each gaussian of the model has as many dimensions as the features.

>>> import kaldi.gmm
>>> isinstance(ubm.gmm, kaldi.gmm.DiagGmm)
True
>>> means = ubm.gmm.get_means()
>>> means.num_rows == num_gauss
True
>>> means.num_cols
39

References
----------

.. [kaldi-gmm]
     https://kaldi-asr.org/doc/model.html

"""

import copy
import os
import numpy as np
import kaldi.gmm
import kaldi.matrix
import kaldi.matrix.common
import kaldi.util.io

from shennong import FeaturesCollection
from shennong.base import BaseProcessor
from shennong.logger import null_logger
from shennong.pipeline import get_default_config, extract_features
from shennong.postprocessor.cmvn import SlidingWindowCmvnPostProcessor
from shennong.postprocessor.vad import VadPostProcessor


class DiagUbmProcessor(BaseProcessor):
    """Universal Background Model with Diagonal GMM"""
    def __init__(self, num_gauss,
                 num_iters=4, num_gselect=15, initial_gauss_proportion=0.5,
                 num_iters_init=20, num_frames=500000,
                 subsample=5, min_gaussian_weight=1e-4,
                 remove_low_count_gaussians=False, seed=0,
                 features=None, vad=None):
        super().__init__()

        self._options = kaldi.gmm.MleDiagGmmOptions()
        self._options.min_gaussian_weight = min_gaussian_weight
        self._options.remove_low_count_gaussians = remove_low_count_gaussians

        self.num_gauss = num_gauss
        self.num_iters = num_iters
        self.num_iters_init = num_iters_init
        self.num_gselect = num_gselect
        self.initial_gauss_proportion = initial_gauss_proportion
        self.num_frames = num_frames
        self.subsample = subsample
        self.seed = seed

        if vad is None:
            config = VadPostProcessor().get_params()
            config['energy_threshold'] = 5.5
            self.vad = config
        else:
            self.vad = vad

        if features in (None, 'default'):
            config = get_default_config(
                'mfcc', with_pitch=False, with_cmvn=False,
                with_sliding_window_cmvn=True)
            config['sliding_window_cmvn']['cmn_window'] = 300
            config['delta']['window'] = 3
            self.features = config
        else:
            self.features = features

        self.gmm = None
        self.selection = None

    @property
    def name(self):
        """Processor name"""
        return 'ubm'

    @property
    def num_gauss(self):
        """Number of Gaussians in the model"""
        return self._num_gauss

    @num_gauss.setter
    def num_gauss(self, value):
        if int(value) < 2:
            raise ValueError(
                'Number of gaussians must be at least 2, not {}'.format(value))
        self._num_gauss = int(value)

    @property
    def num_iters(self):
        """Number of iterations of training."""
        return self._num_iters

    @num_iters.setter
    def num_iters(self, value):
        self._num_iters = int(value)

    @property
    def num_iters_init(self):
        """ Number of E-M iterations for model initialization."""
        return self._num_iters_init

    @num_iters_init.setter
    def num_iters_init(self, value):
        self._num_iters_init = int(value)

    @property
    def num_gselect(self):
        """Number of Gaussians per frame to limit computation to, for speed."""
        return self._num_gselect

    @num_gselect.setter
    def num_gselect(self, value):
        self._num_gselect = int(value)

    @property
    def initial_gauss_proportion(self):
        """Proportion of Gaussians to start with in initialization phase
        (then split)"""
        return self._initial_gauss_proportion

    @initial_gauss_proportion.setter
    def initial_gauss_proportion(self, value):
        self._initial_gauss_proportion = float(value)

    @property
    def num_frames(self):
        """Maximum num-frames to keep in memory for model initialization."""
        return self._num_frames

    @num_frames.setter
    def num_frames(self, value):
        self._num_frames = int(value)

    @property
    def subsample(self):
        """In main E-M phase, use every n frames (a speedup)"""
        return self._subsample

    @subsample.setter
    def subsample(self, value):
        self._subsample = int(value)

    @property
    def min_gaussian_weight(self):
        """Minimum weight below which a Gaussian is not updated"""
        return np.float32(self._options.min_gaussian_weight)

    @min_gaussian_weight.setter
    def min_gaussian_weight(self, value):
        self._options.min_gaussian_weight = float(value)

    @property
    def remove_low_count_gaussians(self):
        """Remove Gaussians with a weight below `min_gaussian_weight`"""
        return self._options.remove_low_count_gaussians

    @remove_low_count_gaussians.setter
    def remove_low_count_gaussians(self, value):
        self._options.remove_low_count_gaussians = bool(value)

    @property
    def features(self):
        """Features extraction configuration"""
        return self._features

    @features.setter
    def features(self, value):
        if not isinstance(value, dict):
            raise TypeError('Features configuration must be a dict')
        if 'mfcc' not in value:
            raise ValueError('Need mfcc features to train UBM-GMM')
        self._features = copy.deepcopy(value)

    @property
    def vad(self):
        """VAD configuration for the UBM-GMM"""
        return self._vad

    @vad.setter
    def vad(self, value):
        if not isinstance(value, dict):
            raise TypeError('VAD configuration must be a dict')

        vad_keys = VadPostProcessor().get_params().keys()
        if not value.keys() <= vad_keys:
            raise ValueError('Unknown parameters given for VAD config')

        self._vad = copy.deepcopy(value)

    @property
    def seed(self):
        """Random seed for initialization from random frames"""
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = int(value)
        self._rng = np.random.RandomState(seed=self._seed)

    @classmethod
    def load(cls, path):
        """Load the GMM from a binary file"""
        if not os.path.isfile(path):
            raise OSError('{}: file not found'.format(path))

        gmm = kaldi.gmm.DiagGmm()
        kstream = kaldi.util.io.xopen(path, mode='rb')
        gmm.read(kstream.stream(), binary=True)
        ubm = DiagUbmProcessor(gmm.get_means().num_rows)
        ubm.gmm = gmm
        return ubm

    def save(self, path):
        """Save the GMM to a binary file"""
        if os.path.isfile(path):
            raise OSError('{}: file already exists'.format(path))

        if not isinstance(self.gmm, kaldi.gmm.DiagGmm):
            raise TypeError('GMM not initialized')

        try:
            self.gmm.gconsts()
        except RuntimeError:
            self.log.debug('Computing gconsts before saving GMM')
            self.gmm.compute_gconsts()

        kstream = kaldi.util.io.xopen(path, mode='wb')
        self.gmm.write(kstream.stream(), binary=True)

    def initialize_gmm(self, feats_collection, njobs=1):
        """Initializes a single diagonal GMM

        Also does multiple iterations of initial training. Adapted from
        [kaldi-init]_.

        Parameters
        ----------
        feats_collection : FeaturesCollection
            The collection of features to initialize the GMM with.
        njobs : int, optional
            Number of threads to use for computation, default to 1.

        Raises
        ------
        ValueError
            If the features have unconsistent dimensions.

        References
        ----------
        .. [kaldi-init]
             https://kaldi-asr.org/doc/gmm-global-init-from-feats_8cc.html

        """
        num_gauss_init = int(self.initial_gauss_proportion * self.num_gauss)
        self.log.info('Initializing model')
        self.log.debug(
            'Starting from %s gaussians, reaching %s in %s iterations',
            num_gauss_init, self.num_gauss, self.num_iters_init)

        self.log.debug('Reading features')
        num_read, dim = 0, 0
        feats = kaldi.matrix.Matrix()
        for utt in feats_collection.keys():
            this_feats = kaldi.matrix.SubMatrix(feats_collection[utt].data)
            for row in range(this_feats.num_rows):
                num_read += 1
                if dim == 0:
                    dim = this_feats.num_cols
                    feats.resize_(self.num_frames, dim)
                elif this_feats.num_cols != dim:
                    raise ValueError(
                        'Features have unconsistent dims '
                        f'{this_feats.num_cols} vs {dim}'
                        f'(current utt is {utt})')

                if num_read <= self.num_frames:
                    feats.row(num_read-1).copy_row_from_mat_(this_feats, row)
                else:
                    if self._rng.random_sample() <= self.num_frames / num_read:
                        feats.row(
                            self._rng.randint(0, self.num_frames + 1)
                        ).copy_row_from_mat_(this_feats, row)

        if num_read < self.num_frames:
            self.log.debug(
                'Number of frames read %s was less than'
                ' target number %s, using all we read',
                num_read, self.num_frames)
            feats.resize_(
                num_read, dim, kaldi.matrix.common.MatrixResizeType.COPY_DATA)
        else:
            self.log.debug(
                'Kept %s out of %s input frames = %s %%',
                self.num_frames, num_read, 100 * self.num_frames / num_read)

        num_gauss_init = int(self.initial_gauss_proportion * self.num_gauss)
        self.gmm = kaldi.gmm.DiagGmm(num_gauss_init, dim)
        self._init_from_random_frames(feats)

        cur_num_gauss = num_gauss_init
        gauss_inc = int((self.num_gauss - num_gauss_init) /
                        (self.num_iters_init / 2))
        if gauss_inc == 0:
            self.log.warning(
                'Number of gaussians %s is too low', self.num_gauss)
            gauss_inc = 1

        # Initial training
        for i in range(self.num_iters_init):
            self.log.debug('Iteration %s', i)
            frame_weights = kaldi.matrix.Vector(feats.num_rows)
            frame_weights.set_(1.0)
            gmm_accs = kaldi.gmm.AccumDiagGmm.new(
                self.gmm, kaldi.gmm.GmmUpdateFlags.ALL)
            tot_like = gmm_accs.accumulate_from_diag_multi_threaded(
                self.gmm, feats, frame_weights, njobs)

            self.log.debug(
                'Likelihood per frame: %s over %s frames',
                tot_like / feats.num_rows, feats.num_rows)

            obj_change, count, _, _, _ = kaldi.gmm.mle_diag_gmm_update(
                self._options,
                gmm_accs,
                kaldi.gmm.GmmUpdateFlags.ALL,
                self.gmm)

            self.log.debug(
                'Objective-function change: %s over %s frames',
                obj_change / count, count)

            next_num_gauss = min(
                self.num_gauss, cur_num_gauss + gauss_inc)
            if next_num_gauss > self.gmm.num_gauss():
                self.log.debug('Splitting to %s Gaussians', next_num_gauss)
                self.gmm.split(next_num_gauss, 0.1)
                cur_num_gauss = next_num_gauss

    def _init_from_random_frames(self, feats):
        """Initialize the GMM parameters by setting the variance to the global
        variance of the features, and the means to distinct randomly chosen
        frames.

        Auxiliary method to :func:`initialize_gmm`.

        Parameters
        ----------
        feats : kaldi.matrix.Matrix or kaldi.matrix.SubMatrix
            Features data from random frames.

        Raises
        ------
        ValueError
            If the features have too few frames to train on
            (less than 10*``num_gauss``). If the features do not
            have positive variance.
        """
        num_gauss = self.gmm.num_gauss()
        num_frames = feats.num_rows
        dim = feats.num_cols

        if num_frames < 10 * num_gauss:
            raise ValueError(
                f'Too few frames to train on ({num_frames} frames)')

        mean, var = kaldi.matrix.Vector(dim), kaldi.matrix.Vector(dim)
        for i in range(num_frames):
            mean.add_vec_(1.0/num_frames, feats.row(i))
            var.add_vec2_(1.0/num_frames, feats.row(i))
        var.add_vec2_(-1.0, mean)

        if var.max() <= 0:
            raise ValueError(
                f'Features do not have positive variance {var}')

        var.invert_elements_()  # Now inverse of variance
        random_frames = self._rng.choice(num_frames, num_gauss, replace=False)
        for gauss in range(num_gauss):
            self.gmm.set_component_weight(gauss, 1.0 / num_gauss)
            self.gmm.set_component_inv_var(gauss, var)
            self.gmm.set_component_mean(gauss, feats.row(random_frames[gauss]))
        self.gmm.compute_gconsts()

    def gaussian_selection(self, feats_collection):
        """Precompute Gaussian indices for pruning.
        For each frame, gives a list of the n best Gaussian indices
        sorted from best to worst.

        Adapted from [kaldi-gselect]_.

        Parameters
        ----------
        feats_collection : FeaturesCollection
            The collection of features to select the best Gaussians from.

        References
        ----------
        .. [kaldi-gselect]
             https://kaldi-asr.org/doc/gmm-gselect_8cc.html

        """
        if not isinstance(self.gmm, kaldi.gmm.DiagGmm):
            raise TypeError('GMM not initialized')

        already_selection = self.selection is not None
        if not already_selection:
            self.selection = {}

        if self.num_gselect > self.gmm.num_gauss():
            self.log.warning(
                'You asked for %s Gaussians but GMM only has %s,'
                ' returning this many. Note: this means the'
                ' Gaussian selection is pointless',
                self.num_gselect, self.gmm.num_gauss())
            self.num_gselect = self.gmm.num_gauss()

        tot_like, tot_t = 0., 0
        num_done = 0
        for utt in feats_collection.keys():
            tot_t_this_file, tot_like_this_file = 0, 0.
            mat = kaldi.matrix.SubMatrix(feats_collection[utt].data)
            tot_t_this_file += mat.num_rows
            if already_selection:
                if utt not in self.selection:
                    raise ValueError(
                        f'No gselect information for utterance {utt}')

                preselect = self.selection[utt]
                if len(preselect) != mat.num_rows:
                    raise ValueError(
                        f'Input gselect utterance {utt} has wrong size')

                for i in range(mat.num_rows):
                    tot_like_this_file_i, \
                        gselect_out = self.gmm.gaussian_selection_preselect(
                            mat.row(i), preselect[i], self.num_gselect)
                    tot_like_this_file += tot_like_this_file_i
                    self.selection[utt][i] = gselect_out
            else:
                tot_like_this_file, gselect_out = \
                    self.gmm.gaussian_selection_matrix(mat, self.num_gselect)
                self.selection[utt] = gselect_out

            tot_t += tot_t_this_file
            tot_like += tot_like_this_file
            if num_done % 10 == 0:
                self.log.debug(
                    'For %sth utterance, average UBM'
                    'likelihood over %s frame is %s',
                    num_done, tot_t_this_file,
                    tot_like_this_file / tot_t_this_file)

            num_done += 1

        self.log.debug(
            'Done %s utterances, mean UBM log-likelihood is %s over %s frames',
            num_done, tot_like / tot_t, tot_t)

    def gaussian_selection_to_post(
            self, feats_collection, min_post=None):
        """Get per-frames posteriors

        Given features and Gaussian-selection (gselect) information for
        a diagonal-covariance GMM, output per-frame posteriors for the selected
        indices.  Also supports pruning the posteriors if they are below
        a stated threshold (and renormalizing the rest to sum to one).

        Adapted from [kaldi-gselect-to-post]_

        Parameters
        ----------
        feats_collection : FeaturesCollection
            The collection of features to use to get the posteriors.
        min_post : int, optional
            Optional, posteriors below this threshold will be pruned away
            and the rest will be renormalized.

        Returns
        -------
        posteriors : dict[str, list[list[tuple[int, float]]]]
            For each utterance, the posteriors are a list of size the number
            of frames of the corresponding features. For each frame, we have
            a list of tuples corresponding to the gaussians in the gaussian
            selection for this frame and their log-likelihood (if the
            log-likelihood is positive).

        References
        ----------
        .. [kaldi-gselect-to-post]
              https://kaldi-asr.org/doc/gmm-global-gselect-to-post_8cc.html

        """
        if not isinstance(self.selection, dict):
            raise ValueError('Gaussian selection has not been done')

        posteriors = {}
        tot_posts, tot_loglike, tot_frames = 0, 0, 0
        for utt in feats_collection.keys():
            mat = kaldi.matrix.SubMatrix(feats_collection[utt].data)
            num_frames = mat.num_rows
            post = []
            if utt not in self.selection:
                raise ValueError(
                    f'No gselect information for utterance {utt}')
            if len(self.selection[utt]) != num_frames:
                raise ValueError(
                    f'Input gselect utterance {utt} has wrong size '
                    f'{len(self.selection[utt])} vs {num_frames}')

            this_tot_loglike = 0.0
            for i in range(num_frames):
                frame = kaldi.matrix.SubVector(mat.row(i))
                this_gselect = self.selection[utt][i]
                loglikes = self.gmm.log_likelihoods_preselect(
                    frame, this_gselect)
                this_tot_loglike += loglikes.apply_softmax_()
                post.append([])

                # now loglikes contains posteriors
                if min_post is not None:
                    _, max_index = loglikes.max_index()
                    for j in range(loglikes.dim):
                        if loglikes[j] < min_post:
                            loglikes[j] = 0
                        total = loglikes.sum()
                        if total == 0:
                            loglikes[max_index] = 1
                        else:
                            loglikes.scale_(1/total)
                for j in range(loglikes.dim):
                    if loglikes[j] != 0:
                        post[i].append((this_gselect[j], loglikes[j]))
                        tot_posts += 1
                assert len(post[i]) != 0

            self.log.debug(
                'Likelihood per frame for utt %s was'
                ' %s per frame over %s frames',
                utt, this_tot_loglike / num_frames, num_frames)

            posteriors[utt] = post
            tot_loglike += this_tot_loglike
            tot_frames += num_frames

        self.log.debug(
            'Overall likelihood per frame is %s with %s '
            'entries per frame over %s frames',
            tot_loglike / tot_frames, tot_posts / tot_frames, tot_frames)

        return posteriors

    def accumulate(self, feats_collection, weights_collection=None, njobs=1):
        """Accumulate stats for training a diagonal-covariance GMM.

        Adapted from [kaldi-acc]_

        Parameters
        ----------
        feats_collection : FeaturesCollection
            The collection of features to use to accumulate stats.
        weights_collection : dict[str, ndarrays], optional
            For each features in the collection, an array of weights to
            apply on the features frames, if specified we must have
            ``weights.keys() == feats_collections.keys()``.
            Unweighted by default.
        njobs : int, optional
            Number of threads to use for computation, default to 1.


        Returns
        -------
        gmm_accs : kaldi.gmm.AccumDiagGmm
            The accumulated stats.

        References
        ----------
        .. [kaldi-acc]
             https://kaldi-asr.org/doc/gmm-global-acc-stats_8cc.html

        """
        if not isinstance(self.gmm, kaldi.gmm.DiagGmm):
            raise TypeError('GMM not initialized')

        # check features
        dim = self.gmm.get_means().num_cols
        for utt, feats in feats_collection.items():
            if feats.ndims != dim:
                raise ValueError(
                    f'Features from utterance {utt} have wrong'
                    f' dims {feats.ndims}, instead of {dim}')

        # check weights
        if weights_collection is not None:
            if weights_collection.keys() != feats_collection.keys():
                raise ValueError(
                    'Keys differ between weights and features collections')
            for utt, weights in weights_collection.items():
                if weights.shape[0] != feats_collection[utt].nframes:
                    raise ValueError(
                        f'Wrong size for weights on utterance {utt}')

        update_flags = (
            kaldi.gmm.GmmUpdateFlags.MEANS +
            kaldi.gmm.GmmUpdateFlags.VARIANCES +
            kaldi.gmm.GmmUpdateFlags.WEIGHTS)
        gmm_accs = kaldi.gmm.AccumDiagGmm.new(self.gmm, update_flags)
        tot_like, tot_weight = 0., 0.
        for utt in feats_collection.keys():
            mat = kaldi.matrix.SubMatrix(feats_collection[utt].data)
            if weights_collection is None:
                weights = kaldi.matrix.Vector(mat.num_rows)
                weights.set_(1)
                file_weight = mat.num_rows
            else:
                weights = kaldi.matrix.SubVector(weights_collection[utt])
                file_weight = sum(weights_collection[utt])
            file_like = gmm_accs.accumulate_from_diag_multi_threaded(
                self.gmm, mat, weights, njobs)

            self.log.debug(
                'Utterance %s: average likelihood = %s over %s frames',
                utt, file_like / file_weight, file_weight)

            tot_like += file_like
            tot_weight += file_weight

        self.log.debug(
            'Overall likelihood per frame = %s over %s weighted frames',
            tot_like / tot_weight, tot_weight)
        return gmm_accs

    def estimate(self, gmm_accs, mixup=None, perturb_factor=0.01):
        """Estimate a diagonal-covariance GMM from the accumulated stats.

        Adapted from [kaldi-gmm-est]_

        Parameters
        ----------
        gmm_accs : kaldi.gmm.AccumDiagGmm
            Accumulated stats
        mixup : int, optional
            Increase number of mixture components to this overall target.
        perturb_factor : float, optional
            While mixing up, perturb means by standard deviation times
            this factor.

        References
        ----------
        .. [kaldi-gmm-est]
             https://kaldi-asr.org/doc/gmm-global-est_8cc.html

        """
        if not isinstance(self.gmm, kaldi.gmm.DiagGmm):
            raise TypeError('GMM not initialized')

        if mixup is not None and mixup <= self.num_gauss:
            raise ValueError(
                'Mixup parameter must be greater than the number of gaussians')

        update_flags = (
            kaldi.gmm.GmmUpdateFlags.MEANS +
            kaldi.gmm.GmmUpdateFlags.VARIANCES +
            kaldi.gmm.GmmUpdateFlags.WEIGHTS)

        objf_impr, count, _, _, _ = kaldi.gmm.mle_diag_gmm_update(
            self._options, gmm_accs, update_flags, self.gmm)

        self.log.debug(
            'Overall objective function improvement is '
            '%s per frame over %s frames', objf_impr / count, count)

        if mixup is not None:
            self.gmm.split(int(mixup), perturb_factor)

    def process(self, utterances, njobs=1):
        """Initialize the GMM, which sets the means to random data points and
        then does some iterations of EM. Train for a few iterations in parallel

        Parameters
        ----------
        utterances : list of tuples
            The utterances can be defined in one of the following format:
            * 1-uple (or str): `<wav-file>`
            * 2-uple: `<utterance-id> <wav-file>`
            * 3-uple: `<utterance-id> <wav-file> <speaker-id>`
            * 4-uple: `<utterance-id> <wav-file> <tstart> <tstop>`
            * 5-uple: `<utterance-id> <wav-file> <speaker-id> <tstart> <tstop>`
        njobs : int, optional
            Number of threads to use for computation, default to 1.

        Raises
        ------
        ValueError
            On errors

        """
        cmvn = self.features.pop('sliding_window_cmvn', None)
        self.log.info('Training UBM using %s jobs', njobs)
        raw_features = extract_features(
            self.features, utterances, njobs=njobs, log=null_logger())

        # Compute VAD decision
        vad = {}
        for utt, mfcc in raw_features.items():
            this_vad = VadPostProcessor(
                **self.vad).process(mfcc)
            vad[utt] = this_vad.data.reshape(
                (this_vad.shape[0],)).astype(bool)

        # Apply cmvn sliding
        features = FeaturesCollection()
        if cmvn is not None:
            proc = SlidingWindowCmvnPostProcessor(**cmvn)
            for utt, mfcc in raw_features.items():
                features[utt] = proc.process(mfcc)
            self.features['sliding_window_cmvn'] = cmvn
        else:
            features = raw_features

        # Select voiced frames
        features = features.trim(vad)

        self.initialize_gmm(features, njobs=njobs)
        self.log.info('Training for %s iterations', self.num_iters)
        features = FeaturesCollection(  # Subsample features collection
            {utt: feats.copy(subsample=self.subsample)
             for utt, feats in features.items()})

        remove_low_count_gaussians = self.remove_low_count_gaussians
        self.remove_low_count_gaussians = False

        for i in range(self.num_iters):
            self.log.debug('Training pass %s', i+1)
            gmm_accs = self.accumulate(features, njobs=njobs)
            if i == self.num_iters-1:
                self.remove_low_count_gaussians = remove_low_count_gaussians
            self.estimate(gmm_accs)
        self.log.info("Done training UBM.")
