"""Extraction of VTLN warp factors from speech signals.

Examples
--------

>>> from shennong.features.processor.vtln import VtlnProcessor
"""
import os
import copy
import kaldi.matrix
import kaldi.matrix.common
import kaldi.matrix.functions
import kaldi.util.io
import kaldi.transform
from math import sqrt

from shennong.features.processor.diagubm import DiagUbmProcessor, Timer
from shennong.features.postprocessor.vad import VadPostProcessor
from shennong.base import BaseProcessor
from shennong.features.features import FeaturesCollection, Features
from shennong.pipeline import extract_features, get_default_config, _extract_features_warp, _Utterance


@Timer('Transform feats')
def _transform_feats(feats_collection, transforms,
                     utt2speak=None):
    transformed_feats = FeaturesCollection()
    if utt2speak is None:
        utt2speak = {utt: utt for utt in feats_collection.keys()}
    for utt in feats_collection:
        transform_rows = transforms[utt2speak[utt]].num_rows
        transform_cols = transforms[utt2speak[utt]].num_cols
        feat_dim = feats_collection[utt].ndims
        feat_out = kaldi.matrix.Matrix(
            feats_collection[utt].nframes, transform_rows)
        if transform_cols == feat_dim:
            feat_out.add_mat_mat_(
                kaldi.matrix.SubMatrix(
                    feats_collection[utt].data), transforms[utt2speak[utt]],
                transA=kaldi.matrix.common.MatrixTransposeType.NO_TRANS,
                transB=kaldi.matrix.common.MatrixTransposeType.TRANS,
                alpha=1.0,
                beta=0.0)
        elif transform_cols == feat_dim+1:
            linear_part = kaldi.matrix.SubMatrix(
                transforms[utt2speak[utt]], 0, transform_rows, 0, feat_dim)
            feat_out.add_mat_mat_(
                kaldi.matrix.SubMatrix(
                    feats_collection[utt].data), linear_part,
                transA=kaldi.matrix.common.MatrixTransposeType.NO_TRANS,
                transB=kaldi.matrix.common.MatrixTransposeType.TRANS,
                alpha=1.0,
                beta=0.0)
            offset = kaldi.matrix.Vector(transform_rows)
            offset.copy_col_from_mat_(transforms[utt2speak[utt]], feat_dim)
            feat_out.add_vec_to_rows_(1.0, offset)
        else:
            raise ValueError(
                f'Transform matrix for utterance {utt} has wrong number of'
                f' cols {transform_cols} versus feat dim {feat_dim}')
        transformed_feats[utt] = Features(
            feat_out.numpy(), feats_collection[utt].times,
            feats_collection[utt].properties)
    return transformed_feats


def _check_utterances(raw_utterances, by_speaker):
    utt2speak = {} if by_speaker else None
    if isinstance(raw_utterances, dict):
        utterances = []
        entries = next(iter(raw_utterances.items()))[1]
        provided = list(map(lambda x: x is None, entries))
        for index, utt in raw_utterances.items():
            if not isinstance(index, str) or not isinstance(utt, _Utterance):
                raise TypeError('Invalid dict of utterances')
            if list(map(lambda x: x is None, utt)) != provided:
                raise ValueError('Unconsistent utterances')
            if by_speaker:
                if utt.speaker is None:
                    raise ValueError(
                        'Requested speaker-adapted VTLN, but speaker'
                        ' information is missing ')
                utt2speak[index] = utt.speaker
            utterances.append(
                (index,)+tuple(info for info in utt if info is not None))
    else:
        if not isinstance(raw_utterances, list):
            raise TypeError('Invalid utterances format')
        utterances = raw_utterances
        if by_speaker:
            utts = list((u,) if isinstance(u, str)
                        else u for u in raw_utterances)
            index_format = set(len(u) for u in utts)
            if not len(index_format) == 1:
                raise ValueError(
                    'the wavs index is not homogeneous, entries'
                    'have different lengths: {}'.format(
                        ', '.join(str(t) for t in index_format)))
            index_format = list(index_format)[0]
            if index_format in [1, 2, 4]:
                raise ValueError(
                    'Requested speaker-adapted VTLN, but speaker'
                    'information is missing ')
            utt2speak = {utt[0]: utt[2] for utt in raw_utterances}
    return utterances, utt2speak


class VtlnProcessor(BaseProcessor):
    """VTLN model
    """

    def __init__(self, by_speaker=True, num_iters=15,
                 min_warp=0.85, max_warp=1.25, warp_step=0.01,
                 logdet_scale=0.0, norm_type='offset', njobs=1,
                 subsample=5, extract_config=None,
                 ubm_config=None, num_gauss=64):
        self.by_speaker = by_speaker
        self.num_iters = num_iters
        self.min_warp = min_warp
        self.max_warp = max_warp
        self.warp_step = warp_step
        self.logdet_scale = logdet_scale
        self.norm_type = norm_type
        self.subsample = subsample
        self.njobs = njobs

        if extract_config is None:
            config = get_default_config(
                'mfcc', with_pitch=False, with_cmvn=False,
                with_sliding_window_cmvn=True)
            config['sliding_window_cmvn']['cmn_window'] = 300
            config['delta']['window'] = 3
            self.extract_config = config
        else:
            self.extract_config = extract_config

        if ubm_config is None:
            self.ubm_config = DiagUbmProcessor(num_gauss).get_params()
        else:
            self.ubm_config = ubm_config

        self._lvtln = None

    @property
    def name(self):  # pragma: nocover
        return 'vtln'

    @property
    def by_speaker(self):
        """Compute the warps for each speaker, or each utterance"""
        return self._by_speaker

    @by_speaker.setter
    def by_speaker(self, value):
        self._by_speaker = bool(value)

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
        ubm_keys = DiagUbmProcessor(1).get_params().keys()
        if not value.keys() <= ubm_keys:
            raise ValueError('Unknown parameters given for UBM config')
        self._ubm_config = copy.deepcopy(value)

    @classmethod
    def load(cls, path):
        """Load the LVTLN from a binary file"""
        if not os.path.isfile(path):
            raise IOError('{}: file not found'.format(path))

        vtln = VtlnProcessor()
        ki = kaldi.util.io.xopen(path, mode='rb')
        vtln._lvtln = kaldi.transform.lvtln.LinearVtln.new(0, 1, 0)
        vtln._lvtln.read(ki.stream(), binary=True)
        return vtln

    def save(self, path):
        """Save the LVTLN to a binary file"""
        if os.path.isfile(path):
            raise IOError('{}: file already exists'.format(path))

        if not isinstance(self._lvtln, kaldi.transform.lvtln.LinearVtln):
            raise TypeError('VTLN not initialized')
        ki = kaldi.util.io.xopen(path, mode='wb')
        self._lvtln.write(ki.stream(), binary=True)

    @Timer(name='Train lvtln special')
    def _train_lvtln_special(self, feats_untransformed,
                             feats_transformed,
                             class_idx, warp,
                             weights_collection=None, posteriors=None):
        """"Set one of the transforms in lvtln to the minimum-squared-error solution
        to mapping feats_untransformed to feats_transformed; posteriors may
        optionally be used to downweight/remove silence.

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
        weights_collection : dict of arrays, optional
            For each features in the collection, an array of weights to
            apply on the features frames. Unweighted by default.
        posteriors : dict of arrays, optional
            Posteriors may "optionally be used to downweight/remove silence

        Raises
        ------
        ValueError
            If the features have unconsistent dimensions. If the size of the
            posteriors does not correspond to the size of the features.

        """
        # Normalize diagonal of variance to be the
        # same before and after transform.
        # We are not normalizing the full covariance
        dim = self._lvtln.dim()
        Q = kaldi.matrix.packed.SpMatrix(dim+1)
        l = kaldi.matrix.Matrix(dim, dim+1)
        c = kaldi.matrix.Vector(dim)
        beta = 0.0
        sum_xplus, sumsq_x, sumsq_diff = kaldi.matrix.Vector(
            dim+1), kaldi.matrix.Vector(dim), kaldi.matrix.Vector(dim)
        for utt in feats_untransformed:
            if utt not in feats_transformed:
                self._log.warning(f'No transformed features for key {utt}')
                continue
            x_feats = kaldi.matrix.SubMatrix(feats_untransformed[utt].data)
            y_feats = kaldi.matrix.SubMatrix(feats_transformed[utt].data)
            if x_feats.num_rows != y_feats.num_rows or \
                    x_feats.num_cols != y_feats.num_cols or \
                    x_feats.num_cols != dim:
                raise ValueError('Number of rows and/or columns differs: '
                                 f'{x_feats.num_rows} vs {y_feats.num_rows} '
                                 f'rows, {x_feats.num_cols} vs '
                                 f'{y_feats.num_cols} columns, {dim} dim')

            weights = kaldi.matrix.Vector(x_feats.num_rows)
            if weights_collection is None and posteriors is not None:
                if utt not in posteriors:
                    self._log.warning(f'No posteriors for utterance {utt}')
                    continue
                post = posteriors[utt]
                if len(post) != x_feats.num_rows:
                    raise ValueError('Mismatch in size of posterior')
                for i in range(len(post)):
                    for j in range(len(post[i])):
                        weights[i] += post[i][j][1]
            elif weights_collection is not None:
                if utt not in weights_collection:
                    self._log.warning(f'No weights for utterance {utt}')
                    continue
                weights.copy_(weights_collection[utt])
            else:
                weights.add_(1)
            for i in range(x_feats.num_rows):
                weight = weights[i]
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
        self._lvtln.set_transform(class_idx, A)
        self._lvtln.set_warp(class_idx, warp)

    @Timer(name="Global gselect to post")
    def gaussian_selection_to_post(self, ubm,
                                   feats_collection,
                                   min_post=None):
        """Given features and Gaussian-selection (gselect) information for
        a diagonal-covariance GMM, output per-frame posteriors for the selected
        indices.  Also supports pruning the posteriors if they are below
        a stated threshold (and renormalizing the rest to sum to one)

        Parameters
        ----------
        ubm : DiagUbm
            [description]
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
        posteriors = {}
        tot_posts, tot_loglike, tot_frames = 0, 0, 0
        num_done, num_err = 0, 0
        for utt in feats_collection.keys():
            mat = kaldi.matrix.SubMatrix(feats_collection[utt].data)
            num_frames = mat.num_rows
            post = []
            if utt not in ubm.selection:
                self._log.warning(
                    f'No gselect information for utterance {utt}')
                num_err += 1
                continue
            if len(ubm.selection[utt]) != num_frames:
                self._log.warning(
                    f'gselect information for utterance {utt} has'
                    f' wrong size {len(ubm.selection[utt])} vs {num_frames}')
                continue
            this_tot_loglike = 0.0
            utt_ok = True
            for i in range(num_frames):
                frame = kaldi.matrix.SubVector(mat.row(i))
                this_gselect = ubm.selection[utt][i]
                loglikes = ubm.gmm.log_likelihoods_preselect(
                    frame, this_gselect)
                this_tot_loglike += loglikes.apply_softmax_()
                post.append([])
                # now loglikes contains posteriors
                if abs(loglikes.sum()-1) > 0.01:
                    utt_ok = False
                else:
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
            if not utt_ok:
                self._log.warning(
                    f'Skipping utterance {utt} because bad'
                    f' posterior-sum encountered (Nan ?)')
            else:
                posteriors[utt] = post
                num_done += 1
                self._log.debug(f'Likelihood per frame for utt {utt} was'
                                f' {this_tot_loglike/num_frames} per frame'
                                f' over {num_frames} frames')
                tot_loglike += this_tot_loglike
                tot_frames += num_frames
        self._log.debug(f'Done {num_done} files, {num_err} had errors.'
                        f' Overall likelihood per frame is'
                        f' {tot_loglike/tot_frames} with'
                        f' {tot_posts/tot_frames}'
                        f' entries per frame over {tot_frames} frames')
        return posteriors

    @Timer('Global est lvtln trans')
    def estimate(self, ubm,
                 feats_collection,
                 posteriors, utt2speak=None):
        """Estimate linear-VTLN transforms, either per utterance or for
        the supplied set of speakers (utt2speak option).
        Reads posteriors indicating Gaussian indexes in the UBM.

        Parameters
        ----------
        ubm : DiagUbmProcessor
            [description]
        feats_collection : FeaturesCollection
            [description]
        posteriors : List[List[Tuple[int, float]]]
            [description]
        utt2speak : dict of str, optional
            [description]
        """
        transforms = {}
        warps = {}
        tot_lvtln_impr, tot_t = 0.0, 0.0
        class_counts = kaldi.matrix.Vector(self._lvtln.num_classes())
        class_counts.set_zero_()
        num_done, num_no_post, num_other_error = 0, 0, 0

        if utt2speak is not None:  # per speaker adaptation
            spk2utt2feats = feats_collection.partition(utt2speak)
            for spk in spk2utt2feats:
                spk_stats = kaldi.transform.mllr.FmllrDiagGmmAccs.from_dim(
                    self._lvtln.dim())
                # Accumulate stats over all utterances of the current speaker
                for utt in spk2utt2feats[spk]:
                    if utt not in posteriors:
                        self._log.warning(
                            f'Did not find posterior for utterance {utt}')
                        num_no_post += 1
                        continue
                    feats = kaldi.matrix.SubMatrix(
                        spk2utt2feats[spk][utt].data)
                    post = posteriors[utt]
                    if len(post) != feats.num_rows:
                        self._log.warning(
                            f'Posterior has wrong size {len(post)}'
                            f' vs {feats.num_rows}')
                        num_other_error += 1
                        continue
                    # Accumulate for utterance
                    for i in range(len(post)):
                        gselect = []
                        this_post = kaldi.matrix.Vector(len(post[i]))
                        for j in range(len(post[i])):
                            gselect.append(post[i][j][0])
                            this_post[j] = post[i][j][1]
                        spk_stats.accumulate_from_posteriors_preselect(
                            ubm.gmm, gselect, feats.row(i), this_post)
                    num_done += 1
                # Compute the transform
                transform = kaldi.matrix.Matrix(
                    self._lvtln.dim(), self._lvtln.dim()+1)
                class_idx, logdet_out, objf_impr, count = \
                    self._lvtln.compute_transform(spk_stats,
                                                  self.norm_type,
                                                  self.logdet_scale,
                                                  transform)
                class_counts[class_idx] += 1
                transforms[spk] = transform
                warps[spk] = self._lvtln.get_warp(class_idx)
                self._log.debug(f'For speaker {spk}, auxf-impr from LVTLN is'
                                f' {objf_impr/count}, over {count} frames')
                tot_lvtln_impr += objf_impr
                tot_t += count

        else:  # per utterance adaptation
            for utt in feats_collection:
                if utt not in posteriors:
                    self._log.warning(
                        f'Did not find posterior for utterance {utt}')
                    num_no_post += 1
                    continue
                feats = kaldi.matrix.Matrix(feats_collection[utt].data)
                post = posteriors[utt]
                if len(post) != feats.num_rows:
                    self._log.warning(f'Posterior has wrong size {len(post)}'
                                      f' vs {feats.num_rows}')
                    num_other_error += 1
                    continue
                num_done += 1
                spk_stats = kaldi.transform.mllr.FmllrDiagGmmAccs.from_dim(
                    self._lvtln.dim())
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
                    self._lvtln.dim(), self._lvtln.dim()+1)
                class_idx, logdet_out, objf_impr, count = \
                    self._lvtln.compute_transform(spk_stats,
                                                  self.norm_type,
                                                  self.logdet_scale,
                                                  transform)
                class_counts[class_idx] += 1
                transforms[utt] = transform
                warps[utt] = self._lvtln.get_warp(class_idx)
                self._log.debug(f'For utterance {utt}, auxf-impr from LVTLN is'
                                f' {objf_impr/count}, over {count} frames')
                tot_lvtln_impr += objf_impr
                tot_t += count

        message = 'Distribution of classes is'
        for count in class_counts:
            message += " "+str(count)
        message += f'\n Done {num_done} files, {num_no_post} with no ' \
            f'posteriors, {num_other_error} with other errors. ' \
            f'Overall LVTLN auxfimpr per frame is {tot_lvtln_impr/tot_t} ' \
            f'over {tot_t} frames'
        self._log.debug(message)
        return transforms, warps

    @Timer('Fit')
    def process(self, raw_utterances, ubm=None):
        """[summary]

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
            [description]

        Returns
        -------
        warps : dict of float
            Warps computed for each speaker or each utterance.
            If by speaker: same warp for all utterances of this spk.
        """
        utterances, utt2speak = _check_utterances(
            raw_utterances, self.by_speaker)
        if ubm is None:
            ubm = DiagUbmProcessor(**self.ubm_config)
            ubm.process(utterances)
        else:
            self.ubm_config = ubm.get_params()
        self._log.info('Initializing base LVTLN transforms')
        dim = ubm.gmm.dim()
        num_classes = int(1.5 + (self.max_warp-self.min_warp)/self.warp_step)
        default_class = int(0.5 + (1-self.min_warp)/self.warp_step)
        self._lvtln = kaldi.transform.lvtln.LinearVtln.new(
            dim, num_classes, default_class)

        orig_features = extract_features(
            self.extract_config, utterances, njobs=self.njobs)
        # Compute VAD decision
        vad = {}
        for utt in orig_features.keys():
            this_vad = VadPostProcessor(
                **ubm.vad_config).process(orig_features[utt])
            vad[utt] = this_vad.data.reshape(
                (this_vad.shape[0],)).astype(bool)
        orig_features = orig_features.trim(vad)
        orig_features = FeaturesCollection(  # Subsample
            {utt: feats.copy(n=self.subsample)
             for utt, feats in orig_features.items()})

        cmvn_config = self.extract_config.pop('sliding_window_cmvn', None)
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
            self._train_lvtln_special(
                featsub_unwarped, featsub_warped, c, this_warp)
        del featsub_warped, featsub_unwarped, vad
        if cmvn_config is not None:
            self.extract_config['sliding_window_cmvn'] = cmvn_config

        self._log.info('Computing Gaussian selection info')
        ubm.gaussian_selection(orig_features)

        self._log.info('Computing initial LVTLN transforms')
        posteriors = self.gaussian_selection_to_post(ubm, orig_features)
        transforms, warps = self.estimate(
            ubm, orig_features, posteriors, utt2speak)
        for i in range(self.num_iters):
            features = _transform_feats(orig_features, transforms, utt2speak)
            # First update the model
            self._log.info(f'Updating model on pass {i+1}')
            gmm_accs = ubm.accumulate(features)
            ubm.estimate(gmm_accs)

            # Now update the LVTLN transforms (and warps)
            self._log.info(f'Re-estimating LVTLN transforms on pass {i+1}')
            posteriors = self.gaussian_selection_to_post(ubm, features)
            transforms, warps = self.estimate(
                ubm, orig_features, posteriors, utt2speak)

        if utt2speak is not None:
            transforms = {utt: transforms[spk]
                          for utt, spk in utt2speak.items()}
            warps = {utt: warps[spk] for utt, spk in utt2speak.items()}
        self._log.info("Done training LVTLN model.")
        return warps, transforms
