"""Extraction of VTLN warp factors from utterances.

Uses the Kaldi implmentation of Linear Vocal Tract Length Normalization
(see [kaldi_lvtln]_).

Examples
--------

>>> from shennong.features.processor.vtln import VtlnProcessor
>>> wav = './test/data/test.wav'
>>> utterances = [('utt1', wav, 'spk1', 0, 1), ('utt2', wav, 'spk1', 1, 1.5)]

Initialize the VTLN model. Other options can be specified at construction,
or after:

>>> vtln = VtlnProcessor(min_warp=0.95, max_warp=1.05)
>>> vtln.num_iters = 10

Returns the computed warps for each utterance. If the `utt2speak` argument
is given, the warps have been computed for each speaker, and each utterance
from the same speaker is mapped to the same warp factor.

>>> warps = vtln.process(utterances)

Those warps can be passed individually in the `process` method of
`MfccProcessor`, `FilterbankProcessor`, `PlpProcessor` and
`SpectrogramProcessor` to warp the corresponding feature.

The features can also be warped directly via the pipeline.

>>> from shennong.features.pipeline import get_default_config, extract_features
>>> config = get_default_config('mfcc', with_vtln=True)
>>> warped_features = extract_features(config, utterances)

References
----------
.. [kaldi_lvtln] https://kaldi-asr.org/doc/transform.html#transform_lvtln

"""

from math import sqrt
import numpy as np
import copy
import os
import yaml
import kaldi.matrix
import kaldi.matrix.common
import kaldi.matrix.functions
import kaldi.util.io
import kaldi.transform

from shennong.base import BaseProcessor
from shennong.features.features import FeaturesCollection, Features
from shennong.features.pipeline import extract_features, get_default_config, _extract_features_warp
from shennong.features.processor.diagubm import DiagUbmProcessor
from shennong.features.postprocessor.vad import VadPostProcessor
from shennong.features.postprocessor.cmvn import SlidingWindowCmvnPostProcessor
from shennong.utils import Timer


class VtlnProcessor(BaseProcessor):
    """VTLN model
    """

    def __init__(self, num_iters=15, min_warp=0.85,
                 max_warp=1.25, warp_step=0.01,
                 logdet_scale=0.0, norm_type='offset', njobs=1,
                 subsample=5, extract_config=None,
                 ubm_config=None, num_gauss=64, by_speaker=True):
        self.num_iters = num_iters
        self.min_warp = min_warp
        self.max_warp = max_warp
        self.warp_step = warp_step
        self.logdet_scale = logdet_scale
        self.norm_type = norm_type
        self.subsample = subsample
        self.njobs = njobs
        self.by_speaker = by_speaker

        if extract_config is None:
            config = get_default_config(
                'mfcc', with_pitch=False, with_cmvn=False,
                with_sliding_window_cmvn=True, with_delta=True)
            config['sliding_window_cmvn']['cmn_window'] = 300
            config['delta']['window'] = 3
            self.extract_config = config
        else:
            self.extract_config = extract_config

        if ubm_config is None:
            self.ubm_config = DiagUbmProcessor(num_gauss).get_params()
        else:
            self.ubm_config = ubm_config

        self.lvtln = None
        self.transforms = None
        self.warps = None

    @property
    def name(self):  # pragma: nocover
        return 'vtln'

    @property
    def num_iters(self):
        """Number of iterations of training"""
        return self._num_iters

    @num_iters.setter
    def num_iters(self, value):
        self._num_iters = int(value)

    @property
    def min_warp(self):
        """Minimum warp considered"""
        return self._min_warp

    @min_warp.setter
    def min_warp(self, value):
        self._min_warp = float(value)

    @property
    def max_warp(self):
        """Maximum warp considered"""
        return self._max_warp

    @max_warp.setter
    def max_warp(self, value):
        self._max_warp = float(value)

    @property
    def warp_step(self):
        """Warp step"""
        return self._warp_step

    @warp_step.setter
    def warp_step(self, value):  # TODO CHECK WITH MAX WARP AND MIN WARP
        self._warp_step = float(value)

    @property
    def logdet_scale(self):
        """Scale on log-determinant term in auxiliary function"""
        return self._logdet_scale

    @logdet_scale.setter
    def logdet_scale(self, value):
        self._logdet_scale = float(value)

    @property
    def norm_type(self):
        """Type of fMLLR applied (`offset`, `none`, `diag`)"""
        return self._norm_type

    @norm_type.setter
    def norm_type(self, value):
        if value not in ['offset', 'none', 'diag']:
            raise ValueError('Invalid norm type {}'.format(value))
        self._norm_type = value

    @property
    def subsample(self):
        return self._subsample

    @subsample.setter
    def subsample(self, value):
        self._subsample = int(value)

    @property
    def njobs(self):
        return self._njobs

    @njobs.setter
    def njobs(self, value):
        self._njobs = int(value)

    @property
    def by_speaker(self):
        return self._by_speaker

    @by_speaker.setter
    def by_speaker(self, value):
        self._by_speaker = bool(value)

    @property
    def extract_config(self):
        return self._extract_config

    @extract_config.setter
    def extract_config(self, value):
        if not isinstance(value, dict):
            raise TypeError('Features extraction configuration must be a dict')
        if 'mfcc' not in value:
            raise ValueError('Need mfcc features to train VTLN model')
        self._extract_config = copy.deepcopy(value)

    @property
    def ubm_config(self):
        return self._ubm_config

    @ubm_config.setter
    def ubm_config(self, value):
        if not isinstance(value, dict):
            raise TypeError('UBM configuration must be a dict')
        ubm_keys = DiagUbmProcessor(2).get_params().keys()
        if not value.keys() <= ubm_keys:
            raise ValueError('Unknown parameters given for UBM config')
        self._ubm_config = copy.deepcopy(value)

    @classmethod
    def load(cls, path):
        """Load the LVTLN from a binary file"""
        if not os.path.isfile(path):
            raise OSError('{}: file not found'.format(path))

        vtln = VtlnProcessor()
        ki = kaldi.util.io.xopen(path, mode='rb')
        vtln.lvtln = kaldi.transform.lvtln.LinearVtln.new(0, 1, 0)
        vtln.lvtln.read(ki.stream(), binary=True)
        return vtln

    @classmethod
    def load_warps(cls, path):
        """Load precomputed warps"""
        if not os.path.isfile(path):
            raise OSError('{}: file not found'.format(path))
        try:
            with open(path) as f:
                warps = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as err:  # pragma: nocover
            raise ValueError(
                'Error in VTLN warps file when loading: {}'.format(err))
        return warps

    def save(self, path):
        """Save the LVTLN to a binary file"""
        if os.path.isfile(path):
            raise OSError('{}: file already exists'.format(path))
        if not isinstance(self.lvtln, kaldi.transform.lvtln.LinearVtln):
            raise TypeError('VTLN not initialized')

        ki = kaldi.util.io.xopen(path, mode='wb')
        self.lvtln.write(ki.stream(), binary=True)

    def save_warps(self, path):
        """Save the computed warps"""
        if os.path.isfile(path):
            raise OSError('{}: file already exists'.format(path))
        if not isinstance(self.warps, dict):
            raise TypeError('Warps not computed')
        try:
            with open(path, 'w') as f:
                yaml.dump(self.warps, f)
        except yaml.YAMLError as err:  # pragma: nocover
            raise ValueError(
                'Error in VTLN warps file when saving: {}'.format(err))

    @Timer(name='Train lvtln special')
    def compute_mapping_transform(self, feats_untransformed,
                                  feats_transformed,
                                  class_idx, warp,
                                  weights=None):
        """"Set one of the transforms in lvtln to the minimum-squared-error solution
        to mapping feats_untransformed to feats_transformed; posteriors may
        optionally be used to downweight/remove silence.

        Adapted from [kaldi_train_lvtln_special]_

        Parameters
        ----------
        feats_untransformed : FeaturesCollection
            Collection of original features.
        feats_transformed :
            Collection of warped features.
        class_idx : int
            Rank of warp considered.
        warp : float, optional
            Warp considered.
        weights : dict of arrays, optional
            For each features in the collection, an array of weights to
            apply on the features frames. Unweighted by default.

        Raises
        ------
        ValueError
            If the features have unconsistent dimensions. If the size of the
            posteriors does not correspond to the size of the features.

        References
        ----------
        .. [kaldi_train_lvtln_special] https://kaldi-asr.org/doc/gmm-train-lvtln-special_8cc.html
        """
        # Normalize diagonal of variance to be the
        # same before and after transform.
        # We are not normalizing the full covariance
        if not isinstance(self.lvtln, kaldi.transform.lvtln.LinearVtln):
            raise TypeError('VTLN not initialized')
        dim = self.lvtln.dim()
        Q = kaldi.matrix.packed.SpMatrix(dim+1)
        l = kaldi.matrix.Matrix(dim, dim+1)
        c = kaldi.matrix.Vector(dim)
        beta = 0.0
        sum_xplus, sumsq_x, sumsq_diff = kaldi.matrix.Vector(
            dim+1), kaldi.matrix.Vector(dim), kaldi.matrix.Vector(dim)
        for utt in feats_untransformed:
            if utt not in feats_transformed:
                raise ValueError(f'No transformed features for key {utt}')
            x_feats = kaldi.matrix.SubMatrix(feats_untransformed[utt].data)
            y_feats = kaldi.matrix.SubMatrix(feats_transformed[utt].data)
            if x_feats.num_rows != y_feats.num_rows or \
                    x_feats.num_cols != y_feats.num_cols or \
                    x_feats.num_cols != dim:
                raise ValueError('Number of rows and/or columns differs: '
                                 f'{x_feats.num_rows} vs {y_feats.num_rows} '
                                 f'rows, {x_feats.num_cols} vs '
                                 f'{y_feats.num_cols} columns, {dim} dim')

            this_weights = kaldi.matrix.Vector(x_feats.num_rows)
            if weights is not None:
                if utt not in weights:
                    raise ValueError(f'No weights for utterance {utt}')
                this_weights.copy_(kaldi.matrix.SubVector(weights[utt]))
            else:
                this_weights.add_(1)
            for i in range(x_feats.num_rows):
                weight = this_weights[i]
                x_row = kaldi.matrix.SubVector(x_feats.row(i))
                y_row = kaldi.matrix.SubVector(y_feats.row(i))
                xplus_row_dbl = kaldi.matrix.Vector(x_row)
                xplus_row_dbl.resize_(
                    dim+1, kaldi.matrix.common.MatrixResizeType.COPY_DATA)
                xplus_row_dbl[dim] = 1.0
                y_row_dbl = kaldi.matrix.Vector(y_row)
                Q.add_vec2_(weight, xplus_row_dbl)
                l.add_vec_vec_(weight, y_row_dbl, xplus_row_dbl)
                beta += weight

                sum_xplus.add_vec_(weight, xplus_row_dbl)
                sumsq_x.add_vec2_(weight, x_row)
                sumsq_diff.add_vec2_(weight, x_row)
                sumsq_diff.add_vec2_(weight, y_row)
                sumsq_diff.add_vec_vec_(-2*weight, x_row, y_row, 1)
                c.add_vec2_(weight, y_row)
        A = kaldi.matrix.Matrix(dim, dim)
        Qinv = kaldi.matrix.packed.SpMatrix(Q.num_rows)
        Qinv.copy_from_sp_(Q)
        Qinv.invert_()
        for i in range(dim):
            w_i = kaldi.matrix.Vector(dim+1)
            l_i = kaldi.matrix.SubVector(l.row(i))
            w_i.add_mat_vec_(
                1.0, Qinv,
                kaldi.matrix.common.MatrixTransposeType.NO_TRANS,
                l_i, 0.0)
            a_i = kaldi.matrix.SubVector(w_i, 0, dim)
            A.row(i).copy_(a_i)
            error = (kaldi.matrix.functions.vec_vec(
                w_i, kaldi.matrix.Vector(dim+1).add_mat_vec_(
                    1.0, Q,
                    kaldi.matrix.common.MatrixTransposeType.NO_TRANS,
                    w_i, 0.0))
                - 2*kaldi.matrix.functions.vec_vec(w_i, l_i) + c[i])/beta
            sqdiff = sumsq_diff[i]/beta
            scatter = sumsq_x[i]/beta
            self._log.debug(
                f'For dimension {i} sum-squared error in linear'
                f' approximation is {error}, versus feature-difference'
                f' {sqdiff}, orig-sumsq is {scatter}')
            # Normalize variance
            x_var = scatter - (sum_xplus[i]/beta)**2
            y_var = kaldi.matrix.functions.vec_vec(
                w_i, kaldi.matrix.Vector(dim+1).add_mat_vec_(
                    1.0, Q,
                    kaldi.matrix.common.MatrixTransposeType.NO_TRANS,
                    w_i, 0.0))/beta \
                - (kaldi.matrix.functions.vec_vec(w_i, sum_xplus)/beta)**2
            scale = sqrt(x_var/y_var)
            A.row(i).scale_(scale)
        self.lvtln.set_transform(class_idx, A)
        self.lvtln.set_warp(class_idx, warp)

    @Timer('Global est lvtln trans')
    def estimate(self, ubm,
                 feats_collection,
                 posteriors, utt2speak=None):
        """Estimate linear-VTLN transforms, either per utterance or for
        the supplied set of speakers (utt2speak option).
        Reads posteriors indicating Gaussian indexes in the UBM.

        Adapted from [kaldi_global_est_lvtln_trans]_

        Parameters
        ----------
        ubm : DiagUbmProcessor
            The Universal Background Model.
        feats_collection : FeaturesCollection
            The untransformed features.
        posteriors : dict of List[List[Tuple[int, float]]]
            The posteriors indicating Gaussian indexes in the UBM.
        utt2speak : dict of str, optional
            If provided, map each utterance to a speaker.

        References
        ----------
        .. [kaldi_global_est_lvtln_trans] https://kaldi-asr.org/doc/gmm-global-est-lvtln-trans_8cc.html
        """
        if not isinstance(self.lvtln, kaldi.transform.lvtln.LinearVtln):
            raise TypeError('VTLN not initialized')
        transforms = {}
        warps = {}
        tot_lvtln_impr, tot_t = 0.0, 0.0
        class_counts = kaldi.matrix.Vector(self.lvtln.num_classes())
        class_counts.set_zero_()

        if utt2speak is not None:  # per speaker adaptation
            spk2utt2feats = feats_collection.partition(utt2speak)
            for spk in spk2utt2feats:
                spk_stats = kaldi.transform.mllr.FmllrDiagGmmAccs.from_dim(
                    self.lvtln.dim())
                # Accumulate stats over all utterances of the current speaker
                for utt in spk2utt2feats[spk]:
                    if utt not in posteriors:
                        raise ValueError(f'No posterior for utterance {utt}')
                    feats = kaldi.matrix.SubMatrix(
                        spk2utt2feats[spk][utt].data)
                    post = posteriors[utt]
                    if len(post) != feats.num_rows:
                        raise ValueError(
                            f'Posterior has wrong size {len(post)}'
                            f' vs {feats.num_rows}')
                    # Accumulate for utterance
                    for i in range(len(post)):
                        gselect = []
                        this_post = kaldi.matrix.Vector(len(post[i]))
                        for j in range(len(post[i])):
                            gselect.append(post[i][j][0])
                            this_post[j] = post[i][j][1]
                        spk_stats.accumulate_from_posteriors_preselect(
                            ubm.gmm, gselect, feats.row(i), this_post)
                # Compute the transform
                transform = kaldi.matrix.Matrix(
                    self.lvtln.dim(), self.lvtln.dim()+1)
                class_idx, logdet_out, objf_impr, count = \
                    self.lvtln.compute_transform(spk_stats,
                                                 self.norm_type,
                                                 self.logdet_scale,
                                                 transform)
                class_counts[class_idx] += 1
                transforms[spk] = transform
                warps[spk] = self.lvtln.get_warp(class_idx)
                self._log.debug(f'For speaker {spk}, auxf-impr from LVTLN is'
                                f' {objf_impr/count}, over {count} frames')
                tot_lvtln_impr += objf_impr
                tot_t += count

        else:  # per utterance adaptation
            for utt in feats_collection:
                if utt not in posteriors:
                    raise ValueError(f'No posterior for utterance {utt}')
                feats = kaldi.matrix.Matrix(feats_collection[utt].data)
                post = posteriors[utt]
                if len(post) != feats.num_rows:
                    raise ValueError(f'Posterior has wrong size {len(post)}'
                                     f' vs {feats.num_rows}')
                spk_stats = kaldi.transform.mllr.FmllrDiagGmmAccs.from_dim(
                    self.lvtln.dim())
                # Accumulate for utterance
                for i in range(len(post)):
                    gselect = []
                    this_post = kaldi.matrix.Vector(len(post[i]))
                    for j in range(len(post[i])):
                        gselect.append(post[i][j][0])
                        this_post[j] = post[i][j][1]
                    spk_stats.accumulate_from_posteriors_preselect(
                        ubm.gmm, gselect, feats.row(i), this_post)
                # Compute the transform
                transform = kaldi.matrix.Matrix(
                    self.lvtln.dim(), self.lvtln.dim()+1)
                class_idx, logdet_out, objf_impr, count = \
                    self.lvtln.compute_transform(spk_stats,
                                                 self.norm_type,
                                                 self.logdet_scale,
                                                 transform)
                class_counts[class_idx] += 1
                transforms[utt] = transform
                warps[utt] = self.lvtln.get_warp(class_idx)
                self._log.debug(f'For utterance {utt}, auxf-impr from LVTLN is'
                                f' {objf_impr/count}, over {count} frames')
                tot_lvtln_impr += objf_impr
                tot_t += count

        message = 'Distribution of classes is'
        for count in class_counts:
            message += " "+str(count)
        message += f'\n Overall LVTLN auxfimpr per' \
            f' frame is {tot_lvtln_impr/tot_t}  over {tot_t} frames'
        self._log.debug(message)
        return transforms, warps

    @Timer('Fit')
    def process(self, utterances, utt2speak=None, ubm=None):
        """Compute the VTLN warp factors for the given utterances.

        Parameters
        ----------
        utts_index : list of tuples
            The utterances can be defined in one of the following format:
            * 1-uple (or str): ``<wav-file>``
            * 2-uple: ``<utterance-id> <wav-file>``
            * 3-uple: ``<utterance-id> <wav-file> <speaker-id>``
            * 4-uple: ``<utterance-id> <wav-file> <tstart> <tstop>``
            * 5-uple: ``<utterance-id> <wav-file> <speaker-id> <tstart> <tstop>``
        ubm : DiagUbmProcessor
            If provided, uses this UBM instead of computing a new one.

        Returns
        -------
        warps : dict of float
            Warps computed for each speaker or each utterance.
            If by speaker: same warp for all utterances of this spk.
        """
        if not self.by_speaker:
            utt2speak = None

        # UBM-GMM
        if ubm is None:
            ubm = DiagUbmProcessor(**self.ubm_config)
            ubm.process(utterances)
        else:
            if ubm.gmm is None:
                raise ValueError('Given UBM-GMM has not been trained')
            self.ubm_config = ubm.get_params()

        self._log.info('Initializing base LVTLN transforms')
        dim = ubm.gmm.dim()
        num_classes = int(1.5 + (self.max_warp-self.min_warp)/self.warp_step)
        default_class = int(0.5 + (1-self.min_warp)/self.warp_step)
        self.lvtln = kaldi.transform.lvtln.LinearVtln.new(
            dim, num_classes, default_class)

        cmvn_config = self.extract_config.pop('sliding_window_cmvn', None)
        raw_mfcc = extract_features(self.extract_config, utterances)
        # Compute VAD decision
        vad = {}
        for utt, mfcc in raw_mfcc.items():
            this_vad = VadPostProcessor(
                **ubm.vad_config).process(mfcc)
            vad[utt] = this_vad.data.reshape(
                (this_vad.shape[0],)).astype(bool)
        # Apply cmvn sliding
        orig_features = FeaturesCollection()
        if cmvn_config is not None:
            proc = SlidingWindowCmvnPostProcessor(**cmvn_config)
            for utt, mfcc in raw_mfcc.items():
                orig_features[utt] = proc.process(mfcc)
        else:
            orig_features = raw_mfcc
        # Select voiced frames
        orig_features = orig_features.trim(vad)
        orig_features = FeaturesCollection(  # Subsample
            {utt: feats.copy(n=self.subsample)
             for utt, feats in orig_features.items()})

        # Computing base transforms
        featsub_unwarped = extract_features(
            self.extract_config, utterances, njobs=self.njobs).trim(vad)
        featsub_unwarped = FeaturesCollection(
            {utt: feats.copy(n=self.subsample)
             for utt, feats in featsub_unwarped.items()})
        for c in range(num_classes):
            this_warp = self.min_warp + c*self.warp_step
            featsub_warped = _extract_features_warp(
                self.extract_config, utterances, this_warp,
                njobs=self.njobs).trim(vad)
            featsub_warped = FeaturesCollection(
                {utt: feats.copy(n=self.subsample)
                 for utt, feats in featsub_warped.items()})
            self.compute_mapping_transform(
                featsub_unwarped, featsub_warped, c, this_warp)
        del featsub_warped, featsub_unwarped, vad
        if cmvn_config is not None:
            self.extract_config['sliding_window_cmvn'] = cmvn_config

        self._log.info('Computing Gaussian selection info')
        ubm.gaussian_selection(orig_features)

        self._log.info('Computing initial LVTLN transforms')
        posteriors = ubm.gaussian_selection_to_post(orig_features)
        self.transforms, self.warps = self.estimate(
            ubm, orig_features, posteriors, utt2speak)

        for i in range(self.num_iters):
            # Transform the features
            features = FeaturesCollection()
            for utt, feats in orig_features.items():
                ind = utt if utt2speak is None else utt2speak[utt]
                linear_part = self.transforms[ind][:, : feats.ndims]
                offset = self.transforms[ind][:, feats.ndims]
                data = np.dot(feats.data, linear_part.numpy().T) + \
                    offset.numpy()
                features[utt] = Features(data, feats.times, feats.properties)

            # Update the model
            self._log.info(f'Updating model on pass {i+1}')
            gmm_accs = ubm.accumulate(features)
            ubm.estimate(gmm_accs)

            # Now update the LVTLN transforms (and warps)
            self._log.info(f'Re-estimating LVTLN transforms on pass {i+1}')
            posteriors = ubm.gaussian_selection_to_post(features)
            self.transforms, self.warps = self.estimate(
                ubm, orig_features, posteriors, utt2speak)

        if utt2speak is not None:
            self.transforms = {utt: self.transforms[spk]
                               for utt, spk in utt2speak.items()}
            self.warps = {utt: self.warps[spk]
                          for utt, spk in utt2speak.items()}

        self._log.info("Done training LVTLN model.")
        return self.warps
