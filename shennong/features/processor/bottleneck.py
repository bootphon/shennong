# coding: utf-8

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
#                                                                             #
# Adaptation for shennong made under GPL3 licence by Mathieu Bernard          #
# <mathieu.a.bernard@inria.fr>. Original code available at                    #
# speech.fit.vutbr.cz/software/but-phonexia-bottleneck-feature-extractor.     #
# Updated code (git repo, python3 compatibility, improvements) available at   #
# https://gitlab.coml.lscp.ens.fr/mbernard/bottleneckfeatureextractor         #
#                                                                             #
###############################################################################
"""Extraction of bottleneck features from a speech signal

    :class:`AudioData` --> BottleneckProcessor --> :class:`Features`

This module provides the class BottleneckProcessor which computes
stacked bottleneck features from audio signals (see [Silnova2018]_ and
[Fer2017]_). This is an adpatation of the original code released on
[bottleneck-site]_. Features are extracted from one of the three
provided pre-trained neural networks:

* *FisherMono*: Trained on Fisher English (parts 1 and 2 datasets,
  about 2000 hours of clean telephone speech) with 120 phoneme states
  as output classes (40 phonemes, 3 state for each phoneme).

* *FisherTri*: Trained on the same datasets as *FisherMono*, with 2423
  triphones as output classes.

* *BabelMulti*: Trained on 17 languages from the IARPA
  [BABEL-project]_, with 3096 output classes (3 phoneme states per
  each language stacked together).

Examples
--------

Compute bottleneck features on some speech using the multilingual
network (*BabelMulti*):

>>> from shennong.audio import AudioData
>>> from shennong.features.processor.bottleneck import BottleneckProcessor
>>> audio = AudioData.load('./test/data/test.wav')
>>> processor = BottleneckProcessor(weights='BabelMulti')
>>> features = processor.process(audio)
>>> features.shape
(140, 80)

References
----------

.. [bottleneck-site]
     https://speech.fit.vutbr.cz/software/but-phonexia-bottleneck-feature-extractor

.. [BABEL-project]
     https://www.iarpa.gov/index.php/research-programs/babel

.. [Silnova2018] Anna Silnova, Pavel Matejka, Ondrej Glembek, Oldrich
     Plchot, Ondrej Novotny, Frantisek Grezl, Petr Schwarz, Lukas
     Burget, Jan “Honza” Cernocky, "BUT/Phonexia Bottleneck Feature
     Extractor", Submitted to Odyssey: The Speaker and Language
     Recognition Workshop 2018

.. [Fer2017] Fér Radek, Matějka Pavel, Grézl František, Plchot
     Oldřich, Veselý Karel and Černocký Jan. Multilingually Trained
     Bottleneck Features in Spoken Language Recognition. Computer
     Speech and Language. Amsterdam: Elsevier Science, 2017,
     vol. 2017, no. 46, pp. 252-267.

"""

import os
import pkg_resources
import warnings

import numpy as np
import scipy as sp
import scipy.linalg as spl
import scipy.fftpack

from shennong.utils import get_logger
from shennong.features import Features
from shennong.features.processor.base import FeaturesProcessor


def _add_dither(signal, level):
    return signal + level * (np.random.rand(*signal.shape) * 2 - 1)


def _mel_inv(x):
    return (np.exp(x/1127.)-1.)*700.


def _mel(x):
    return 1127.*np.log(1. + x/700.)


def _framing(a, window, shift=1):
    shape = (int((a.shape[0] - window) / shift + 1), window) + a.shape[1:]
    strides = (a.strides[0] * shift, a.strides[0]) + a.strides[1:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def _mel_fbank_mx(winlen_nfft, fs, numchans=20, lofreq=0.0, hifreq=None):
    """Returns mel filterbank as an array shaped (nfft/2+1, numchans)

    Parameters
    ----------
    winlen_nfft : int
        Typically the window length as used in mfcc_htk() call. It is
        used to determine number of samples for FFT computation
        (NFFT). If positive, the value (window lenght) is rounded up
        to the next higher power of two to obtain HTK-compatible NFFT.
        If negative, NFFT is set to -winlen_nfft. In such case, the
        parameter nfft in mfcc_htk() call should be set likewise.
    fs : int
        sampling frequency (in Hz)
    numchans : int, optional
        number of filter bank bands (default to 20)
    lofreq : float, optional
        frequency (Hz) where the first filter strats (default to 0.0)
    hifreq : float, optional
        frequency (Hz) where the last filter ends (default fs/2)

    """
    if not hifreq:
        hifreq = 0.5 * fs

    nfft = (2**int(np.ceil(np.log2(winlen_nfft)))
            if winlen_nfft > 0 else -int(winlen_nfft))
    fbin_mel = _mel(np.arange(nfft / 2 + 1, dtype=float) * fs / nfft)
    cbin_mel = np.linspace(_mel(lofreq), _mel(hifreq), numchans + 2)
    cind = np.floor(_mel_inv(cbin_mel) / fs * nfft).astype(int) + 1
    mfb = np.zeros((len(fbin_mel), numchans))

    for i in range(numchans):
        mfb[cind[i]:cind[i+1], i] = (
            cbin_mel[i] - fbin_mel[cind[i]:cind[i+1]]) / (
                cbin_mel[i] - cbin_mel[i+1])

        mfb[cind[i+1]:cind[i+2], i] = (
            cbin_mel[i+2] - fbin_mel[cind[i+1]:cind[i+2]]) / (
                cbin_mel[i+2] - cbin_mel[i+1])

        if lofreq > 0.0 and float(lofreq) / fs*nfft+0.5 > cind[0]:
            mfb[cind[0], :] = 0.0  # Just to be HTK compatible

    return mfb


def _fbank_htk(x, window, noverlap, fbank_mx):
    """Mel log Mel-filter bank channel outputs

    Returns numchans-by-M matrix of log Mel-filter bank outputs extracted from
    signal x, where M is the number of extracted frames, which can be computed
    as floor((length(x)-noverlap)/(window-noverlap)).

    Parameters
    ----------
    x : array
        input signal
    window : int
        frame window lentgth (in samples,
        i.e. window_size/source_rate) or vector of widow weights
    noverlap : int
        overlapping between frames (in samples, i.e window -
        target_rate/source_rate)
    fbank_mx : array
        array with (Mel) filter bank (as returned by function
        :func:`mel_fbank_mx`)

    """
    if np.isscalar(window):
        window = np.hamming(window)
    nfft = 2 ** int(np.ceil(np.log2(window.size)))
    x = _framing(x.astype("float"), window.size, window.size-noverlap).copy()
    x *= window

    # inhibit a FutureWarning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FutureWarning)
        x = np.fft.rfft(x, nfft)

    x = x.real**2 + x.imag**2
    x = np.log(np.maximum(1.0, np.dot(x, fbank_mx)))
    return x


def _uppertri_indices(dim, isdiag=False):
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


def _uppertri_to_sym(covs_ut2d, utr, utc):
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


def _uppertri1d_from_sym(cov_full, utr, utc):
    return cov_full[(utr, utc)]


def _uppertri1d_to_sym(covs_ut1d, utr, utc):
    return _uppertri_to_sym(np.array(covs_ut1d)[:, None], utr, utc)[:, :, 0]


def _inv_posdef_and_logdet(M):
    U = np.linalg.cholesky(M)
    logdet = 2*np.sum(np.log(np.diagonal(U)))
    invM = spl.solve(M, np.identity(M.shape[0], M.dtype), sym_pos=True)
    return invM, logdet


def _gmm_eval_prep(weights, means, covs):
    n_mix, dim = means.shape
    GMM = dict()
    is_full_cov = covs.shape[1] != dim
    GMM['utr'], GMM['utc'] = _uppertri_indices(dim, not is_full_cov)

    if is_full_cov:
        GMM['gconsts'] = np.zeros(n_mix)
        GMM['gconsts2'] = np.zeros(n_mix)
        GMM['invCovs'] = np.zeros_like(covs)
        GMM['invCovMeans'] = np.zeros_like(means)

        for ii in range(n_mix):
            _uppertri1d_to_sym(covs[ii], GMM['utr'], GMM['utc'])

            invC, logdetC = _inv_posdef_and_logdet(
                _uppertri1d_to_sym(covs[ii], GMM['utr'], GMM['utc']))

            # log of Gauss. dist. normalizer + log weight + mu' invCovs mu
            invCovMean = invC.dot(means[ii])
            GMM['gconsts'][ii] = np.log(weights[ii]) - 0.5 * (
                logdetC + means[ii].dot(invCovMean) + dim * np.log(2.0*np.pi))
            GMM['gconsts2'][ii] = - 0.5 * (
                logdetC + means[ii].dot(invCovMean) + dim * np.log(2.0*np.pi))
            GMM['invCovMeans'][ii] = invCovMean

            # Inverse covariance matrices are stored in columns of 2D
            # matrix as vectorized upper triangual parts ...
            GMM['invCovs'][ii] = _uppertri1d_from_sym(
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


def _gmm_llhs(data, GMM):
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


def _gmm_eval(data, GMM, return_accums=0):
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
    llh = _logsumexp(gamma, axis=1)

    if return_accums == 0:
        return llh

    gamma = sp.exp(gamma.T - llh)
    N = gamma.sum(axis=1)
    F = gamma.dot(data)

    if return_accums == 1:
        return llh, N, F

    S = gamma.dot(data_sqr)
    return llh, N, F, S


def _logsumexp(x, axis=0):
    xmax = x.max(axis)
    ex = sp.exp(x - np.expand_dims(xmax, axis))
    x = xmax + sp.log(sp.sum(ex, axis))
    not_finite = ~np.isfinite(xmax)
    x[not_finite] = xmax[not_finite]
    return x


def _gmm_update(N, F, S):
    """weights means covs = gmm_update(N,F,S) return GMM parameters,
    which are updated from accumulators

    """
    dim = F.shape[1]
    is_diag_cov = S.shape[1] == dim
    utr, utc = _uppertri_indices(dim, is_diag_cov)
    sumN = N.sum()
    weights = N / sumN
    means = F / N[:, np.newaxis]
    covs = S / N[:, np.newaxis] - means[:, utr] * means[:, utc]
    return weights, means, covs


def _compute_vad(s, log, win_length=200, win_overlap=120,
                 n_realignment=5, threshold=0.3, bugfix=False):
    warnings.filterwarnings('error')

    # power signal for energy computation
    if bugfix is False:
        s = s ** 2  # yields to negative squares because s are int16
    else:
        s = s.astype(np.float64) ** 2

    # frame signal with overlap
    F = _framing(s, win_length, win_length - win_overlap)
    # sum frames to get energy
    E = F.sum(axis=1).astype(np.float64)
    # E = np.sqrt(E)
    # E = np.log(E)

    # normalize the energy
    try:
        E -= E.mean()
        E /= E.std()
        # initialization
        mm = np.array((-1.00, 0.00, 1.00))[:, np.newaxis]
        ee = np.array((1.00, 1.00, 1.00))[:, np.newaxis]
        ww = np.array((0.33, 0.33, 0.33))

        GMM = _gmm_eval_prep(ww, mm, ee)

        E = E[:, np.newaxis]

        for i in range(n_realignment):
            # collect GMM statistics
            llh, N, F, S = _gmm_eval(E, GMM, return_accums=2)

            # update model
            ww, mm, ee = _gmm_update(N, F, S)
            # wrap model
            GMM = _gmm_eval_prep(ww, mm, ee)

        # evaluate the gmm llhs
        llhs = _gmm_llhs(E, GMM)
        llh = _logsumexp(llhs, axis=1)[:, np.newaxis]
        llhs = np.exp(llhs - llh)

        out = np.zeros(llhs.shape[0], dtype=np.bool)
        out[llhs[:, 0] < threshold] = True
    except RuntimeWarning:
        log.warning("signal contains only silence")
        out = np.zeros(E.shape[0], dtype=np.bool)

    return out


def _dct_basis(nbasis, length):
    # the same DCT as in matlab
    return scipy.fftpack.idct(np.eye(nbasis, length), norm='ortho')


def _sigmoid_fun(x):
    return 1 / (1 + np.exp(-x))


def _preprocess_nn_input(X, left_ctx=5, right_ctx=5):
    X = _framing(X, left_ctx+1+right_ctx).transpose(0, 2, 1)
    dct_basis = 6
    dct_xform = _dct_basis(dct_basis, left_ctx+right_ctx+1)
    dct_xform[0] = np.sqrt(2./(left_ctx+right_ctx+1))
    hamming_dct = (dct_xform*np.hamming(left_ctx+right_ctx+1)).T

    return np.dot(
        X.reshape(-1, hamming_dct.shape[0]),
        hamming_dct).reshape(X.shape[0], -1)


def _create_nn_extract_st_BN(X, param_dict, bn_position):
    mean = param_dict['input_mean']
    std = param_dict['input_std']
    Y = (X + mean) * std
    num_of_layers = int((len(param_dict.keys()) - 5) / 2)

    # n_hidden_before_BN --> sigmoid
    # BN activation --> linear
    for ii, f in enumerate(
            [lambda x: _sigmoid_fun(x)]*bn_position+[lambda x:x]):
        W = param_dict['W'+str(ii+1)]
        b = param_dict['b'+str(ii+1)]
        Y = f(Y.dot(W) + b)

    Y1 = np.hstack([Y[0:-20], Y[5:-15], Y[10:-10], Y[15:-5], Y[20:]])
    bn_mean = param_dict['bn_mean']
    bn_std = param_dict['bn_std']
    Y1 = (Y1+bn_mean) * bn_std
    for ii, f in enumerate(
            [lambda x: _sigmoid_fun(x)]*(
                num_of_layers - bn_position-2) + [lambda x:x]):
        W = param_dict['W'+str(ii+bn_position+3)]
        b = param_dict['b'+str(ii+bn_position+3)]
        Y1 = f(Y1.dot(W) + b)
    return Y1, Y


class BottleneckProcessor(FeaturesProcessor):
    """Bottleneck features from a pre-trained neural network

    Parameters
    ----------
    weights : 'BabelMulti', 'FisherMono' or 'FisherMulti'
        The pretrained weights to use for features extraction

    Raises
    ------
    ValueError
        If the `weights` are invalid

    RuntimeError
        If the weights file cannot be found (meaning shennong is not
        correctly installed on your system)

    """
    _log = get_logger(__name__)

    def __init__(self, weights='BabelMulti'):
        self.weights = weights

    @property
    def weights(self):
        """The name of the pretrained weights used to extract the features

        Must be 'BabelMulti', 'FisherMono' or 'FisherTri'.

        """
        return self._weights

    @weights.setter
    def weights(self, value):
        _available_weights = self.available_weights()
        try:
            weights_file = _available_weights[value]
            self._log.debug('loading %s', os.path.basename(weights_file))
            self._weights_data = np.load(_available_weights[value])
            self._weights = value
        except KeyError:
            raise ValueError(
                'invalid weights "{}", choose in "{}"'.format(
                    value, ', '.join(sorted(_available_weights.keys()))))

    @property
    def ndims(self):
        return 80

    @classmethod
    def available_weights(cls):
        """Return the pretrained weights files as a dict (name -> file)

        Returns
        -------
        weight_files : dict
            A mapping 'weights name' -> 'weights files', where the
            files are absolutes paths to compressed numpy array (.npz
            format). The 'weights name' is either *BabelMulti*,
            *FisherMono* or *FisherTri*.

        Raises
        ------
        RuntimeError
            If the directory `shennong/share/bottleneck` is not found,
            or if all the weights files are missing in it.

        """
        # locate the directory shennong/share/bottleneck, raise if it
        # cannot be found
        directory = pkg_resources.resource_filename(
            pkg_resources.Requirement.parse('shennong'),
            'shennong/share/bottleneck')
        if not os.path.isdir(directory):  # pragma: nocover
            raise RuntimeError('directory not found: {}'.format(directory))

        # retrieve the weights files
        expected_files = {
            f[0]: os.path.join(directory, f[1] + '.npz') for f in
            [('BabelMulti', 'Babel-ML17_FBANK_HL1500_SBN80_PhnStates3096'),
             ('FisherMono', 'FisherEnglish_FBANK_HL500_SBN80_PhnStates120'),
             ('FisherTri', 'FisherEnglish_FBANK_HL500_SBN80_triphones2423')]}

        # make sure all the files are here, raise a RuntimeError if
        # all files are missing, log a warning is only one or two
        # files are missing
        files = {k: v for k, v in expected_files.items() if os.path.isfile(v)}
        if not files:  # pragma: nocover
            raise RuntimeError('no weights file found in {}'.format(directory))
        for k in expected_files.keys():
            if k not in files:  # pragma: nocover
                cls.log.warning('weights file for "%s" is unavailable', k)

        return files

    def process(self, signal):
        """Computes bottleneck features on an audio `signal`

        Use a pre-trained neural network to extract bottleneck
        features. Features have a frame shift of 10 ms and frame
        length of 25 ms.

        Parameters
        ----------
        signal : AudioData, shape = [nsamples, 1]
            The input audio signal to compute the features on, must be
            mono. The signal is up/down-sampled at 8 kHz during
            processing.

        Returns
        -------
        features : Features, shape = [nframes, 80]
            The computes bottleneck features will have as many rows as
            there are frames (depends on the `signal` duration, expect
            about 100 frames per second), each frame with 80
            dimensions.

        Raises
        ------
        RuntimeError
            If no speech is detected on the `signal` during the voice
            activity detection preprocessing step.

        """
        # force resampling to 8 kHz and 16 bits integers
        need_resample = (
            signal.sample_rate != 8000 or
            signal.dtype is not np.dtype(np.int16))

        if need_resample:
            self._log.info(
                'resampling audio from %dHz@%db to %dHz@%db',
                signal.sample_rate, signal.dtype.itemsize * 8, 8000, 16)
            signal = signal.resample(8000).astype(np.int16)

        signal = signal.data

        # define parameters to extract mel filterbanks. Those
        # parameters cannot be tuned because the networks are trained
        # with them... frame_noverlap is the number of samples to
        # overlap in each frame, so the frame_shift is 200 - 120 = 80
        frame_length = 200
        frame_noverlap = 120
        frame_shift = frame_length - frame_noverlap

        # voice activity detection TODO implement user-provided VAD
        # (vad input format could be an instance of Alignment, or
        # simply an array of bool).
        vad = _compute_vad(
            signal, self._log,
            win_length=frame_length, win_overlap=frame_noverlap)

        # ensure we have some voiced frames in the signal
        voiced_frames = sum(vad)
        if not voiced_frames:
            raise RuntimeError(
                'no voice detected in signal, failed to extract features')
        self._log.info('%d frames of speech detected (on %d total frames)',
                       voiced_frames, len(vad))

        # from audio signal to mel filterbank
        signal = _add_dither(signal, 0.1)
        window = np.hamming(frame_length)
        fbank_mx = _mel_fbank_mx(
            window.size, 8000, numchans=24, lofreq=64.0, hifreq=3800.0)
        fea = _fbank_htk(signal, window, frame_noverlap, fbank_mx)

        # center the mel features from voiced frames mean
        fea -= np.mean(fea[vad], axis=0)

        # add a global context to the mel features
        left_ctx = right_ctx = 15
        fea = np.r_[np.repeat(fea[[0]], left_ctx, axis=0),
                    fea,
                    np.repeat(fea[[-1]], right_ctx, axis=0)]

        # compute the network output from mel features
        left_ctx_bn1 = right_ctx_bn1 = self._weights_data['context']
        nn_input = _preprocess_nn_input(fea, left_ctx_bn1, right_ctx_bn1)
        nn_output = np.vstack(_create_nn_extract_st_BN(
            nn_input, self._weights_data, 2)[0])

        # compute the timestamps for each output frame
        times = (1.0 / 8000) * np.vstack((
            np.arange(nn_output.shape[0]) * frame_shift,
            np.arange(nn_output.shape[0]) * frame_shift + frame_length)).T

        # return the final bottleneck features
        return Features(nn_output, times, self.get_params())
