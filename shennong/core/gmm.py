###############################################################################
#                                                                             #
#  copyright (C) 2017 by Anna Silnova, Pavel Matejka, Oldrich Plchot,         #
#                        Frantisek Grezl                                      #
#                                                                             #
#                        Brno Universioty of Technology                       #
#                        Faculty of information technology                    #
#                        Department of Computer Graphics and Multimedia       #
#  email: {isilnova,matejkap,iplchot,grezl}@vut.cz                            #
#                                                                             #
###############################################################################
#                                                                             #
#  This software and provided models can be used freely for research          #
#  and educational purposes. For any other use, please contact BUT            #
#  and / or LDC representatives.                                              #
#                                                                             #
###############################################################################
#                                                                             #
# Licensed under the Apache License, Version 2.0 (the "License");             #
# you may not use this file except in compliance with the License.            #
# You may obtain a copy of the License at                                     #
#                                                                             #
#     http://www.apache.org/licenses/LICENSE-2.0                              #
#                                                                             #
# Unless required by applicable law or agreed to in writing, software         #
# distributed under the License is distributed on an "AS IS" BASIS,           #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
#                                                                             #
###############################################################################
"""Gaussian Mixture Model"""

import numpy as np
import scipy.linalg as spl
import scipy as sp


def uppertri_indices(dim, isdiag=False):
    """ [utr utc]=uppertri_indices(D, isdiag) returns row and column indices
    into upper triangular part of DxD matrices. Indices go in zigzag feshinon
    starting by diagonal. For convenient encoding of diagonal matrices, 1:D
    ranges are returned for both outputs utr and utc when ISDIAG is true.
    """
    if isdiag:
        utr = np.arange(dim)
        utc = np.arange(dim)
    else:
        utr = np.hstack([np.arange(ii) for ii in range(dim, 0, -1)])
        utc = np.hstack([np.arange(ii, dim) for ii in range(dim)])
    return utr, utc


# def uppertri_from_sym(covs_full, utr, utc):
#     """ covs_ut2d = uppertri_from_sym(covs_full) reformat full
#     symmetric matrices stored in 3rd dimension of 3D matrix into
#     vectorized upper triangual matrices efficiently stored in
#     columns of 2D matrix """
#     dim, dim, n_mix = covs_full.shape
#
#     covs_ut2d = np.zeros(((dim**2+dim)/2, n_mix), dtype=covs_full.dtype)
#     for ii in range(n_mix):
#         covs_ut2d[:, ii] = covs[:, :, ii][(utr, utc)]
#     return covs_ut2d


def uppertri_to_sym(covs_ut2d, utr, utc):
    """ covs = uppertri_to_sym(covs_ut2d) reformat vectorized upper triangual
    matrices efficiently stored in columns of 2D matrix into full symmetric
    matrices stored in 3rd dimension of 3D matrix
    """

    (ut_dim, n_mix) = covs_ut2d.shape
    dim = (np.sqrt(1 + 8 * ut_dim) - 1) / 2

    covs_full = np.zeros((dim, dim, n_mix), dtype=covs_ut2d.dtype)
    for ii in range(n_mix):
        covs_full[:, :, ii][(utr, utc)] = covs_ut2d[:, ii]
        covs_full[:, :, ii][(utc, utr)] = covs_ut2d[:, ii]
    return covs_full


def uppertri1d_from_sym(cov_full, utr, utc):
    return cov_full[(utr, utc)]


def uppertri1d_to_sym(covs_ut1d, utr, utc):
    return uppertri_to_sym(np.array(covs_ut1d)[:, None], utr, utc)[:, :, 0]


def inv_posdef_and_logdet(M):
    U = np.linalg.cholesky(M)
    logdet = 2*np.sum(np.log(np.diagonal(U)))
    invM = spl.solve(M, np.identity(M.shape[0], M.dtype), sym_pos=True)
    return invM, logdet


def gmm_eval_prep(weights, means, covs):
    n_mix, dim = means.shape
    GMM = dict()
    is_full_cov = covs.shape[1] != dim
    GMM['utr'], GMM['utc'] = uppertri_indices(dim, not is_full_cov)

    if is_full_cov:
        GMM['gconsts'] = np.zeros(n_mix)
        GMM['gconsts2'] = np.zeros(n_mix)
        GMM['invCovs'] = np.zeros_like(covs)
        GMM['invCovMeans'] = np.zeros_like(means)

        for ii in range(n_mix):
            uppertri1d_to_sym(covs[ii], GMM['utr'], GMM['utc'])

            invC, logdetC = inv_posdef_and_logdet(
                uppertri1d_to_sym(covs[ii], GMM['utr'], GMM['utc']))

            # log of Gauss. dist. normalizer + log weight + mu' invCovs mu
            invCovMean = invC.dot(means[ii])
            GMM['gconsts'][ii] = np.log(weights[ii]) - 0.5 * (
                logdetC + means[ii].dot(invCovMean) + dim * np.log(2.0*np.pi))
            GMM['gconsts2'][ii] = - 0.5 * (
                logdetC + means[ii].dot(invCovMean) + dim * np.log(2.0*np.pi))
            GMM['invCovMeans'][ii] = invCovMean

            # Inverse covariance matrices are stored in columns of 2D
            # matrix as vectorized upper triangual parts ...
            GMM['invCovs'][ii] = uppertri1d_from_sym(
                invC, GMM['utr'], GMM['utc'])

        # ... with elements above the diagonal multiply by 2
        GMM['invCovs'][:, dim:] *= 2.0
    else:  # for diagonal
        GMM['invCovs'] = 1 / covs
        GMM['gconsts'] = np.log(weights) - 0.5 * (
            np.sum(np.log(covs) + means**2 * GMM['invCovs'],
                   axis=1) + dim * np.log(2 * np.pi))
        GMM['gconsts2'] = -0.5 * (
            np.sum(np.log(covs) + means**2 * GMM['invCovs'],
                   axis=1) + dim * np.log(2 * np.pi))
        GMM['invCovMeans'] = GMM['invCovs'] * means

    # for weight = 0, prepare GMM for uninitialized model with single
    # gaussian
    if len(weights) == 1 and weights[0] == 0:
        GMM['invCovs'] = np.zeros_like(GMM['invCovs'])
        GMM['invCovMeans'] = np.zeros_like(GMM['invCovMeans'])
        GMM['gconsts'] = np.ones(1)
    return GMM


def gmm_llhs(data, GMM):
    """llh = GMM_EVAL(d,GMM) returns vector of log-likelihoods evaluated
    for each frame of dimXn_samples data matrix using GMM object. GMM
    object must be initialized with GMM_EVAL_PREP function.

    [llh N F] = GMM_EVAL(d,GMM,1) also returns accumulators with zero,
    first order statistic.

    [llh N F S] = GMM_EVAL(d,GMM,2) also returns accumulators with
    second order statistic.  For full covariance model second order
    statiscics, only the vectorized upper triangual parts are stored
    in columns of 2D matrix (similarly to GMM.invCovs).

    """
    # quadratic expansion of data
    data_sqr = data[:, GMM['utr']] * data[:, GMM['utc']]

    # computate of log-likelihoods for each frame and all Gaussian
    # components
    gamma = -0.5 * data_sqr.dot(GMM['invCovs'].T) + data.dot(
        GMM['invCovMeans'].T) + GMM['gconsts']

    return gamma


def gmm_eval(data, GMM, return_accums=0):
    """llh = GMM_EVAL(d,GMM) returns vector of log-likelihoods evaluated
    for each frame of dimXn_samples data matrix using GMM object. GMM
    object must be initialized with GMM_EVAL_PREP function.

    [llh N F] = GMM_EVAL(d,GMM,1) also returns accumulators with zero,
    first order statistic.

    [llh N F S] = GMM_EVAL(d,GMM,2) also returns accumulators with
    second order statistic.  For full covariance model second order
    statiscics, only the vectorized upper triangual parts are stored
    in columns of 2D matrix (similarly to GMM.invCovs).

    """
    # quadratic expansion of data
    data_sqr = data[:, GMM['utr']] * data[:, GMM['utc']]

    # computate of log-likelihoods for each frame and all Gaussian components
    gamma = -0.5 * data_sqr.dot(GMM['invCovs'].T) + data.dot(
        GMM['invCovMeans'].T) + GMM['gconsts']
    llh = logsumexp(gamma, axis=1)

    if return_accums == 0:
        return llh

    gamma = sp.exp(gamma.T - llh)
    N = gamma.sum(axis=1)
    F = gamma.dot(data)

    if return_accums == 1:
        return llh, N, F

    S = gamma.dot(data_sqr)
    return llh, N, F, S


def logsumexp(x, axis=0):
    xmax = x.max(axis)
    ex = sp.exp(x - np.expand_dims(xmax, axis))
    x = xmax + sp.log(sp.sum(ex, axis))
    not_finite = ~np.isfinite(xmax)
    x[not_finite] = xmax[not_finite]
    return x


def gmm_update(N, F, S):
    """weights means covs = gmm_update(N,F,S) return GMM parameters,
    which are updated from accumulators

    """
    dim = F.shape[1]
    is_diag_cov = S.shape[1] == dim
    utr, utc = uppertri_indices(dim, is_diag_cov)
    sumN = N.sum()
    weights = N / sumN
    means = F / N[:, np.newaxis]
    covs = S / N[:, np.newaxis] - means[:, utr] * means[:, utc]
    return weights, means, covs
