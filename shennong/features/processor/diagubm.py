"""Provides the DiagUbmClass to train a Universal Background Model
- Gaussian Mixture Model (UBM-GMM) with diagonal covariances

Uses the kaldi implementation of GMM (see [kaldi_gmm]_)

Examples
--------

>>> from shennong.features.processor.diagubm import DiagUbmProcessor
>>> wav = './test/data/test.wav'
>>> utterances = [('utt1', wav, 'spk1', 0, 1), ('utt2', wav, 'spk1', 1, 1.5)]

Initialize the UBM-GMM with a given number of gaussians. Other options
can be specified at construction, or after:

>>> num_gauss = 32
>>> ubm = DiagUbmProcessor(num_gauss, num_iters_init=10)
>>> ubm.num_iters = 3

Process

>>> ubm.process(num_gauss)

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

.. [kaldi_gmm] https://kaldi-asr.org/doc/model.html
"""

import copy
import os
import kaldi.base.math
import kaldi.util.io
import kaldi.matrix
import kaldi.matrix.common
import kaldi.gmm
from kaldi.gmm import GmmUpdateFlags

from shennong.base import BaseProcessor
from shennong.features.pipeline import get_default_config, extract_features
from shennong.features.postprocessor.vad import VadPostProcessor
from shennong.features.postprocessor.cmvn import SlidingWindowCmvnPostProcessor
from shennong.features.features import FeaturesCollection

# -----------WILL BE REMOVED--------------
from shennong.utils import get_logger
from dataclasses import dataclass, field
import time
from typing import Callable, ClassVar, Dict, Optional
from contextlib import ContextDecorator


# https://realpython.com/python-timer
class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


@dataclass
class Timer(ContextDecorator):
    timers: ClassVar[Dict[str, float]] = dict()
    name: Optional[str] = None
    text: str = " {:0.4f} seconds"
    logger: Optional[Callable[[str], None]] = get_logger().info
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Add timer to dict of timers after initialization"""
        if self.name is not None:
            self.text = self.name + self.text
            self.timers.setdefault(self.name, 0)

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        self.stop()

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.logger:
            self.logger(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time

# ------------END OF REMOVING-----------------


class DiagUbmProcessor(BaseProcessor):
    """Universal Background Model with Diagonal GMM
    """

    def __init__(self, num_gauss,
                 num_iters=4, num_gselect=15, initial_gauss_proportion=0.5,
                 num_iters_init=20, njobs=1, num_frames=500000,
                 subsample=5, min_gaussian_weight=0.0001,
                 remove_low_count_gaussians=False, seed=0,
                 extract_config=None, vad_config=None):
        self._options = kaldi.gmm.MleDiagGmmOptions()
        self._options.min_gaussian_weight = min_gaussian_weight
        self._options.remove_low_count_gaussians = remove_low_count_gaussians
        self._state = kaldi.base.math.RandomState()
        self._state.seed = seed

        self.num_gauss = num_gauss
        self.num_iters = num_iters
        self.num_iters_init = num_iters_init
        self.num_gselect = num_gselect
        self.initial_gauss_proportion = initial_gauss_proportion
        self.njobs = njobs
        self.num_frames = num_frames
        self.subsample = subsample

        if vad_config is None:
            config = VadPostProcessor().get_params()
            config['energy_threshold'] = 5.5
            self.vad_config = config
        else:
            self.vad_config = vad_config

        if extract_config is None:
            config = get_default_config(
                'mfcc', with_pitch=False, with_cmvn=False,
                with_sliding_window_cmvn=True)
            config['sliding_window_cmvn']['cmn_window'] = 300
            config['delta']['window'] = 3
            self.extract_config = config
        else:
            self.extract_config = extract_config

        self.gmm = None
        self.selection = None

    @property
    def name(self):  # pragma: nocover
        return 'diag-ubm'

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
    def njobs(self):
        """Number of threads to use in initialization phase."""
        return self._njobs

    @njobs.setter
    def njobs(self, value):
        self._njobs = int(value)

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
        return self._options.min_gaussian_weight

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
    def seed(self):
        """Random seed"""
        return self._state.seed

    @seed.setter
    def seed(self, value):
        self._state.seed = int(value)

    @property
    def extract_config(self):
        """Features extraction configuration"""
        return self._extract_config

    @extract_config.setter
    def extract_config(self, value):
        if not isinstance(value, dict):
            raise TypeError('Features configuration must be a dict')
        if 'mfcc' not in value:
            raise ValueError('Need mfcc features to train UBM-GMM')
        self._extract_config = copy.deepcopy(value)

    @property
    def vad_config(self):
        """VAD configuration for the UBM-GMM"""
        return self._vad_config

    @vad_config.setter
    def vad_config(self, value):
        if not isinstance(value, dict):
            raise TypeError('VAD configuration must be a dict')
        vad_keys = VadPostProcessor().get_params().keys()
        if not value.keys() <= vad_keys:
            raise ValueError('Unknown parameters given for VAD config')
        self._vad_config = copy.deepcopy(value)

    @classmethod
    def load(cls, path):
        """Load the GMM from a binary file"""
        if not os.path.isfile(path):
            raise OSError('{}: file not found'.format(path))

        gmm = kaldi.gmm.DiagGmm()
        ki = kaldi.util.io.xopen(path, mode='rb')
        gmm.read(ki.stream(), binary=True)
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
            self._log.debug('Computing gconsts before saving GMM')
            self.gmm.compute_gconsts()

        ki = kaldi.util.io.xopen(path, mode='wb')
        self.gmm.write(ki.stream(), binary=True)

    @Timer(name='Global init')
    def initialize_gmm(self, feats_collection):
        """Initializes a single diagonal GMM and does multiple iterations of
        training.

        Adapted from [kaldi_init]_

        Parameters
        ----------
        feats_collection : FeaturesCollection
            The collection of features to initialize the GMM with.

        Raises
        ------
        ValueError
            If the features have unconsistent dimensions.

        References
        ----------
        .. [kaldi_init] https://kaldi-asr.org/doc/gmm-global-init-from-feats_8cc.html
        """
        num_gauss_init = int(self.initial_gauss_proportion*self.num_gauss)
        self._log.info(
            f'Initializing model from E-M in memory. Starting from'
            f' {num_gauss_init}, reaching {self.num_gauss} in'
            f' {self.num_iters_init} iterations, using at most'
            f' {self.num_frames} frames of data')

        self._log.debug(
            f'Reading features (will keep {self.num_frames} frames)')
        num_read, dim = 0, 0
        feats = kaldi.matrix.Matrix()
        for utt in feats_collection.keys():
            this_feats = kaldi.matrix.SubMatrix(feats_collection[utt].data)
            for t in range(this_feats.num_rows):
                num_read += 1
                if dim == 0:
                    dim = this_feats.num_cols
                    feats.resize_(self.num_frames, dim)
                elif this_feats.num_cols != dim:
                    raise ValueError('Features have unconsistent dims '
                                     f'{this_feats.num_cols} vs {dim}'
                                     f'(current utt is {utt})')
                if num_read <= self.num_frames:
                    feats.row(num_read-1).copy_row_from_mat_(this_feats, t)
                else:
                    keep_prob = self.num_frames / num_read
                    if kaldi.base.math.with_prob(keep_prob):
                        feats.row(kaldi.base.math.rand_int(
                            0, self.num_frames-1, self._state)
                        ).copy_row_from_mat_(this_feats, t)
        if num_read < self.num_frames:
            self._log.debug(f'Number of frames read {num_read} was less than'
                            f' target number {self.num_frames}, using all we'
                            f' read.')
            feats.resize_(
                num_read, dim, kaldi.matrix.common.MatrixResizeType.COPY_DATA)
        else:
            percent = self.num_frames*100/num_read
            self._log.debug(f'Kept {self.num_frames} out of {num_read} input'
                            f' frames = {percent} %')

        num_gauss_init = int(self.initial_gauss_proportion*self.num_gauss)
        self.gmm = kaldi.gmm.DiagGmm(num_gauss_init, dim)
        self._init_from_random_frames(feats)

        cur_num_gauss = num_gauss_init
        gauss_inc = int((self.num_gauss - num_gauss_init) /
                        (self.num_iters_init / 2))
        if gauss_inc == 0:
            self._log.warning(
                f'Number of gaussians {self.num_gauss} is too low')
            gauss_inc = 1

        # Initial training
        for i in range(self.num_iters_init):
            self._log.debug(f'Iteration {i}')
            frame_weights = kaldi.matrix.Vector(feats.num_rows)
            frame_weights.set_(1.0)
            gmm_accs = kaldi.gmm.AccumDiagGmm.new(self.gmm, GmmUpdateFlags.ALL)
            tot_like = gmm_accs.accumulate_from_diag_multi_threaded(
                self.gmm, feats, frame_weights, self.njobs)
            self._log.debug(f'Likelihood per frame: {tot_like/feats.num_rows}'
                            f' over {feats.num_rows} frames')
            obj_change, count, _, _, _ = kaldi.gmm.mle_diag_gmm_update(
                self._options, gmm_accs, GmmUpdateFlags.ALL, self.gmm)
            self._log.debug(
                f'Objective-function change: {obj_change/count} over {count}'
                f'frames')

            next_num_gauss = min(
                self.num_gauss, cur_num_gauss + gauss_inc)
            if next_num_gauss > self.gmm.num_gauss():
                self._log.debug(f'Splitting to {next_num_gauss} Gaussians')
                self.gmm.split(next_num_gauss, 0.1)
                cur_num_gauss = next_num_gauss

    @Timer(name='Init gmm from random frames')
    def _init_from_random_frames(self, feats):
        """GMM initialization.

        Parameters
        ----------
        feats : Matrix or SubMatrix
            Features data from random frames.

        Raises
        ------
        ValueError
            If the features have too few frames to train on
            (less than 10*`num_gauss`). If the features do not
            have positive variance.
        """
        num_gauss = self.gmm.num_gauss()
        num_frames = feats.num_rows
        dim = feats.num_cols
        if num_frames < 10*num_gauss:
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
        used_frames = set()
        var.invert_elements_()  # Now inverse of variance
        for g in range(num_gauss):
            random_frame = kaldi.base.math.rand_int(
                0, num_frames-1, self._state)
            while random_frame in used_frames:
                random_frame = kaldi.base.math.rand_int(
                    0, num_frames-1, self._state)
            used_frames.add(random_frame)
            self.gmm.set_component_weight(g, 1.0/num_gauss)
            self.gmm.set_component_inv_var(g, var)
            self.gmm.set_component_mean(g, feats.row(random_frame))
        self.gmm.compute_gconsts()

    @Timer(name='Gselect')
    def gaussian_selection(self, feats_collection):
        """Precompute Gaussian indices for pruning
        For each frame, gives a list of the n best Gaussian indices
        sorted from best to worst.

        Adapted from [kaldi_gselect]_.

        Parameters
        ----------
        feats_collection : FeaturesCollection
            The collection of features to select the best Gaussians from.

        References
        ----------
        .. [kaldi_gselect] https://kaldi-asr.org/doc/gmm-gselect_8cc.html
        """
        if not isinstance(self.gmm, kaldi.gmm.DiagGmm):
            raise TypeError('GMM not initialized')
        already_selection = self.selection is not None
        if not already_selection:
            self.selection = {}
        if self.num_gselect > self.gmm.num_gauss():
            self._log.warning(f'You asked for {self.num_gselect} Gaussians'
                              f' but GMM only has {self.gmm.num_gauss()},'
                              f' returning this many. Note: this means the'
                              f' Gaussian selection is pointless')
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
                self._log.debug(f'For {num_done}th utterance, average UBM'
                                f'likelihood over {tot_t_this_file} frame is'
                                f' {tot_like_this_file/tot_t_this_file}')
            num_done += 1
        self._log.debug(f'Done {num_done} utterances, average UBM log-'
                        f'likelihood is {tot_like/tot_t} over {tot_t} frames')

    @Timer(name="Global gselect to post")
    def gaussian_selection_to_post(self,
                                   feats_collection,
                                   min_post=None):
        """Given features and Gaussian-selection (gselect) information for
        a diagonal-covariance GMM, output per-frame posteriors for the selected
        indices.  Also supports pruning the posteriors if they are below
        a stated threshold (and renormalizing the rest to sum to one)

        Parameters
        ----------
        feats_collection : FeaturesCollection
            [description]
        min_post : int, optional
            Optional, posteriors below this threshold will be pruned away
            and the rest will be renormalized

        Returns
        -------
        posteriors : List[List[Tuple[int, float]]]
            [description]
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
            posteriors[utt] = post
            self._log.debug(f'Likelihood per frame for utt {utt} was'
                            f' {this_tot_loglike/num_frames} per frame'
                            f' over {num_frames} frames')
            tot_loglike += this_tot_loglike
            tot_frames += num_frames
        self._log.debug(f' Overall likelihood per frame is'
                        f' {tot_loglike/tot_frames} with'
                        f' {tot_posts/tot_frames}'
                        f' entries per frame over {tot_frames} frames')
        return posteriors

    @Timer(name='Global acc stats')
    def accumulate(self, feats_collection, weights_collection=None):
        """Accumulate stats for training a diagonal-covariance GMM.

        Adapted from [kaldi_acc]_

        Parameters
        ----------
        feats_collection : FeaturesCollection
            The collection of features to use to accumulate stats.
        weights_collection : dict of arrays, optional
            For each features in the collection, an array of weights to
            apply on the features frames, if specified we must have
            ``weights.keys() == feats_collections.keys()``.
            Unweighted by default.

        Returns
        -------
        gmm_accs : AccumDiagGmm
            accumulated stats

        References
        ----------
        .. [kaldi_acc] https://kaldi-asr.org/doc/gmm-global-acc-stats_8cc.html
        """
        if not isinstance(self.gmm, kaldi.gmm.DiagGmm):
            raise TypeError('GMM not initialized')
        # check features
        dim = self.gmm.get_means().num_cols
        for utt, feats in feats_collection.items():
            if feats.ndims != dim:
                raise ValueError(f'Features from utterance {utt} have wrong'
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

        update_flags = GmmUpdateFlags.MEANS + \
            GmmUpdateFlags.VARIANCES + \
            GmmUpdateFlags.WEIGHTS
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
                self.gmm, mat, weights, self.njobs)
            self._log.debug(
                f'Utterance {utt}: average likelihood ='
                f'{file_like/file_weight} over {file_weight} frames')
            tot_like += file_like
            tot_weight += file_weight
        self._log.debug(
            f'Overall likelihood per frame = {tot_like/tot_weight} over'
            f' {tot_weight} weighted frames')
        return gmm_accs

    @Timer(name='Global est')
    def estimate(self, gmm_accs, mixup=None, perturb_factor=0.01):
        """Estimate a diagonal-covariance GMM from the accumulated stats.

        Adapted from [kaldi_gmm_est]_

        Parameters
        ----------
        gmm_accs : AccumDiagGmm
            Accumulated stats
        mixup : int, optional
            Increase number of mixture components to this overall target.
        perturb_factor : float, optional
            While mixing up, perturb means by standard deviation times
            this factor.

        References
        ----------
        .. [kaldi_gmm_est] https://kaldi-asr.org/doc/gmm-global-est_8cc.html
        """
        if not isinstance(self.gmm, kaldi.gmm.DiagGmm):
            raise TypeError('GMM not initialized')
        if mixup is not None:
            pass  # TODO: vérifier tout ca
        update_flags = GmmUpdateFlags.MEANS + \
            GmmUpdateFlags.VARIANCES + \
            GmmUpdateFlags.WEIGHTS

        objf_impr, count, _, _, _ = kaldi.gmm.mle_diag_gmm_update(
            self._options, gmm_accs, update_flags, self.gmm)
        self._log.debug(f'Overall objective function improvement is'
                        f' {objf_impr/count} per frame over {count} frames')
        if mixup is not None:
            self.gmm.split(mixup, perturb_factor)

    @Timer(name='Fit')
    def process(self, utterances):
        """Initialize the GMM, which sets the means to random data points and
        then does some iterations of EM. Train for a few iterations in parallel

        Parameters
        ----------
        utterances : list of tuples
            The utterances can be defined in one of the following format:
            * 1-uple (or str): <wav-file>`
            * 2-uple: `<utterance-id> <wav-file>`
            * 3-uple: `<utterance-id> <wav-file> <speaker-id>`
            * 4-uple: `<utterance-id> <wav-file> <tstart> <tstop>`
            * 5-uple: `<utterance-id> <wav-file> <speaker-id> <tstart> <tstop>`
        """
        cmvn_config = self.extract_config.pop('sliding_window_cmvn', None)
        raw_mfcc = extract_features(self.extract_config, utterances)
        # Compute VAD decision
        vad = {}
        for utt, mfcc in raw_mfcc.items():
            this_vad = VadPostProcessor(
                **self.vad_config).process(mfcc)
            vad[utt] = this_vad.data.reshape(
                (this_vad.shape[0],)).astype(bool)
        # Apply cmvn sliding
        features = FeaturesCollection()
        if cmvn_config is not None:
            proc = SlidingWindowCmvnPostProcessor(**cmvn_config)
            for utt, mfcc in raw_mfcc.items():
                features[utt] = proc.process(mfcc)
            self.extract_config['sliding_window_cmvn'] = cmvn_config
        else:
            features = raw_mfcc
        # Select voiced frames
        features = features.trim(vad)

        self.initialize_gmm(features)
        self._log.info(f'Will train for {self.num_iters} iterations')
        features = FeaturesCollection(  # Subsample features collection
            {utt: feats.copy(n=self.subsample)
             for utt, feats in features.items()})
        for i in range(self.num_iters):
            self._log.info(f'Training pass {i+1}')
            gmm_accs = self.accumulate(features)
            if i == self.num_iters-1:
                self.remove_low_count_gaussians = True
            self.estimate(gmm_accs)
        self.remove_low_count_gaussians = False
        self._log.info("Done training UBM.")
