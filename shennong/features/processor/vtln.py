import os
import kaldi.matrix
import kaldi.matrix.common
import kaldi.util.io
import kaldi.transform
from math import sqrt

from shennong.features.processor.diagubm import DiagUbmProcessor, subsample_feats, extract_features_sliding_warp, Timer
from shennong.base import BaseProcessor
from shennong.features.features import FeaturesCollection, Features
from shennong.utils import get_logger


def _transform_feats(feats_collection, transforms, log=get_logger()):
    num_err, num_done = 0, 0
    transformed_feats = FeaturesCollection()
    for utt in feats_collection:
        transform_rows = transforms[utt].num_rows
        transform_cols = transforms[utt].num_cols
        feat_dim = feats_collection[utt].num_cols
        feat_out = kaldi.matrix.Matrix(
            feats_collection[utt].num_rows, transform_rows)
        if transform_cols == feat_dim:
            feat_out.add_mat_mat_(
                kaldi.matrix.SubMatrix(feats_collection[utt]), transforms[utt],
                transA=kaldi.matrix.common.MatrixTransposeType.NO_TRANS,
                transB=kaldi.matrix.common.MatrixTransposeType.TRANS,
                alpha=1.0,
                beta=0.0)
        elif transform_cols == feat_dim+1:
            linear_part = kaldi.matrix.SubMatrix(
                transforms[utt], 0, transform_rows, 0, feat_dim)
            feat_out.add_mat_mat_(
                kaldi.matrix.SubMatrix(feats_collection[utt]), linear_part,
                transA=kaldi.matrix.common.MatrixTransposeType.NO_TRANS,
                transB=kaldi.matrix.common.MatrixTransposeType.TRANS,
                alpha=1.0,
                beta=0.0)
            offset = kaldi.matrix.Vector(transform_rows)
            offset.copy_col_from_mat_(transforms[utt], feat_dim)
            feat_out.add_vec_to_rows_(1.0, offset)
        else:
            log.warning(
                f'Transform matrix for utterance {utt} has wrong number of'
                f' cols {transform_cols} versus feat dim {feat_dim}')
            num_err += 1
            continue
        num_done += 1
        transformed_feats[utt] = Features(
            feat_out.numpy(), feats_collection[utt].times,
            feats_collection[utt].properties)
    return transformed_feats


class VtlnProcessor(BaseProcessor):
    """VTLN model
    """

    def __init__(self, by_speaker=True, num_iters=15,
                 min_warp=0.85, max_warp=1.25, warp_step=0.01,
                 logdet_scale=0.0, norm_type='offset', subsample=5):
        self._by_speaker = by_speaker
        self._num_iters = num_iters
        self._min_warp = min_warp
        self._max_warp = max_warp
        self._warp_step = warp_step
        self._logdet_scale = logdet_scale
        self._norm_type = norm_type
        self._subsample = subsample

        self._lvtln = None

    @property
    def name(self):
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
    def warp_step(self, value):
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
    def subsample(self):  # TODO: get rid of subsampling afterwards ?
        return self._subsample

    @subsample.setter
    def subsample(self, value):
        self._subsample = int(value)

    def load(self, path):
        """Load the LVTLN from a binary file"""
        if not os.path.isfile(path):
            raise IOError('{}: file not found'.format(path))

        ki = kaldi.util.io.xopen(path, mode='rb')
        self._lvtln = kaldi.transform.lvtln.LinearVtln.new(0, 1, 0)
        self._lvtln.read(ki.stream(), binary=True)

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
                    w_i, 0.0))/beta
            - (kaldi.matrix.functions.vec_vec(w_i, sum_xplus)/beta)**2
            scale = sqrt(x_var/y_var)
            A.row(i).scale_(scale)
        self._lvtln.set_transform(class_idx, A)
        self._lvtln.set_warp(class_idx, warp)

    @Timer(name="Global gselect to post")
    def _global_gselect_to_post(self, ubm,
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
            mat = kaldi.matrix.Matrix(feats_collection[utt].data)
            num_frames = mat.num_rows
            post = []
            if utt not in ubm.select:
                self._log.warning(
                    f'No gselect information for utterance {utt}')
                num_err += 1
                continue
            if len(ubm.select[utt]) != num_frames:
                self._log.warning(
                    f'gselect information for utterance {utt} has'
                    f' wrong size {len(ubm.select[utt])} vs {num_frames}')
                continue
            this_tot_loglike = 0.0
            utt_ok = True
            for i in range(num_frames):
                frame = kaldi.matrix.SubVector(mat.row(i))
                this_gselect = ubm.select[utt][i]
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
    def _global_est_lvtln_trans(self, ubm,
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
                    feats = kaldi.matrix.Matrix(spk2utt2feats[spk][utt].data)
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
    def process(self, utterances, ubm=None, **kwargs):
        """[summary]

        Parameters
        ----------
        feats_collection : FeaturesCollection
            [description]
        ubm : DiagUbmProcessor
            [description]

        Returns
        -------
        warps : dict of float
            Warps computed for each speaker or each utterance.
        """
        if ubm is None:
            ubm = DiagUbmProcessor(**kwargs)
            ubm.process(utterances)

        utt2speak = {u.file: u.speaker for u in utterances.values()
                     } if self.by_speaker else {}

        self._log.info('Initiliazing base LVTLN transforms')
        dim = ubm.gmm.dim()
        num_classes = int(1.5 + (self.max_warp-self.min_warp)/self.warp_step)
        default_class = int(0.5 + (1-self.min_warp)/self.warp_step)
        self._lvtln = kaldi.transform.lvtln.LinearVtln.new(
            dim, num_classes, default_class)

        featsub_unwarped = extract_features_sliding_warp(
            utterances, apply_cmn=False)
        featsub_unwarped = subsample_feats(featsub_unwarped, n=self.subsample)
        for c in range(num_classes):
            this_warp = self.min_warp + c*self.warp_step
            featsub_warped = extract_features_sliding_warp(
                utterances, apply_cmn=False, warp=this_warp)
            featsub_warped = subsample_feats(featsub_warped, n=self.subsample)
            self._train_lvtln_special(
                featsub_unwarped, featsub_warped, c, warp=this_warp)
            del featsub_warped
        del featsub_unwarped

        orig_features = extract_features_sliding_warp(utterances, **ubm.config)
        orig_features = subsample_feats(orig_features, n=self.subsample)

        self._log.info('Computing Gaussian selection info')
        ubm.gselect(orig_features)

        self._log.info('Computing initial LVTLN transforms')
        posteriors = self._global_gselect_to_post(ubm, orig_features)
        transforms, warps = self._global_est_lvtln_trans(
            ubm, orig_features, posteriors, utt2speak)
        for i in range(self.num_iters):
            features = _transform_feats(orig_features, transforms)
            # First update the model
            self._log.info(f'Updating model on pass {i+1}')
            gmm_accs = ubm._global_acc_stats(features)
            ubm._global_est(gmm_accs)

            # Now update the LVTLN transforms (and wrap)
            self._log.info(f'Re-estimating LVTLN transforms on pass {i+1}')
            posteriors = self._global_gselect_to_post(ubm, features)
            transforms, warps = self._global_est_lvtln_trans(
                ubm, orig_features, posteriors, utt2speak)

        self._log.info("Done training LVTLN model.")
        return warps
