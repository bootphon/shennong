"""Extraction of RASTA-PLP features from a speech signal

    :class:`~shennong.audio.Audio` --> RastaPlpProcessor
    --> :class:`~shennong.features.Features`

Implementation of the RASTA-PLP features extraction algorithm (see
[labrosa]_ and [rastapy]_ for implementations and [Herm94]_ for the
paper).

Examples
--------

Compute RASTA-PLP features on some speech signal:

>>> from shennong.audio import Audio
>>> from shennong.processor.rastaplp import RastaPlpProcessor
>>> audio = Audio.load('./test/data/test.wav')
>>> processor = RastaPlpProcessor(order=8)
>>> features = processor.process(audio)
>>> features.shape
(140, 9)

The output dimension depends on the PLP ``order`` parameter:

>>> processor.order = 10
>>> features = processor.process(audio)
>>> features.shape
(140, 11)


References
----------

.. [labrosa]
     https://labrosa.ee.columbia.edu/matlab/rastamat/

.. [rastapy]
     https://github.com/mystlee/rasta_py

.. [Herm94]
     H. Hermansky and N. Morgan, "RASTA processing of speech", IEEE
     Trans. on Speech and Audio Proc., vol. 2, no. 4, pp. 578-589,
     Oct. 1994.

"""

import kaldi.feat.window
import kaldi.matrix
import numpy as np
import scipy.fftpack
import scipy.signal

from shennong import Features
from shennong.processor.base import FramesProcessor


def _audspec(p_spectrum, fs=16000, nfilts=0, fbtype='bark',
             min_freq=0, max_freq=0, sumpower=True, bandwidth=1):
    if max_freq == 0:
        max_freq = fs / 2

    nfreqs = p_spectrum.shape[0]
    nfft = (int(nfreqs) - 1) * 2

    if fbtype == 'bark':
        wts = _fft2barkmx(nfft, fs, nfilts, bandwidth, min_freq, max_freq)
    else:  # pragma: nocover
        if fbtype == 'mel':
            htk = False
            constamp = False
        elif fbtype == 'htkmel':
            htk = True
            constamp = True
        elif fbtype == 'fcmel':
            htk = True
            constamp = False
        else:
            raise ValueError('unknown fbtype')

        wts = _fft2melmx(
            nfft, fs, nfilts, bandwidth, min_freq, max_freq,
            htk=htk, constamp=constamp)

    wts = wts[:, 0:nfreqs]

    if sumpower:
        aspectrum = np.matmul(wts, p_spectrum)
    else:  # pragma: nocover
        aspectrum = np.matmul(wts, np.sqrt(p_spectrum)) ** 2
    return aspectrum


def _hz2bark(f):
    return 6 * np.arcsinh(f / 600)


def _bark2hz(z):
    return 600 * np.sinh(z / 6)


def _fft2barkmx(fft_length, fs, nfilts, band_width, min_freq, max_freq):
    min_bark = _hz2bark(min_freq)
    nyqbark = _hz2bark(max_freq) - min_bark

    if nfilts == 0:
        nfilts = np.ceil(nyqbark) + 1

    wts = np.zeros((int(nfilts), int(fft_length)))
    step_barks = nyqbark / (nfilts - 1)
    binbarks = _hz2bark(np.arange(0, fft_length / 2 + 1) * fs / fft_length)

    for i in range(int(nfilts)):
        f_bark_mid = min_bark + i * step_barks
        lof = binbarks - f_bark_mid - 0.5
        hif = binbarks - f_bark_mid + 0.5
        minimum = np.minimum(0, np.minimum(hif, -2.5 * lof) / band_width)
        wts[i, 0:int(fft_length / 2) + 1] = np.power(10, minimum)
    return wts


def _fft2melmx(fft_length, fs, nfilts=0, band_width=1, min_freq=0, max_freq=0,
               htk=False, constamp=False):  # pragma: nocover
    if nfilts == 0:
        nfilts = np.ceil(_hz2mel(max_freq, htk) / 2)
    if max_freq == 0:
        max_freq = fs / 2

    wts = np.zeros((int(nfilts), int(fft_length)))
    fftfrqs = (np.arange(0, fft_length / 2 + 1) / fft_length) * fs

    min_mel = _hz2mel(min_freq, htk)
    max_mel = _hz2mel(max_freq, htk)
    binfrqs = _mel2hz(np.add(min_mel, np.multiply(
        np.arange(0, nfilts + 2), (max_mel - min_mel) / (nfilts + 1))), htk)

    for i in range(int(nfilts)):
        fs_tmp = binfrqs[np.arange(0, 3) + i]
        fs_tmp = fs_tmp[1] + band_width * (fs_tmp - fs_tmp[1])
        loslope = (fftfrqs - fs_tmp[0]) / (fs_tmp[1] - fs_tmp[0])
        hislope = (fs_tmp[2] - fftfrqs) / (fs_tmp[2] - fs_tmp[1])
        wts[i, 0:int(fft_length / 2) + 1] = np.maximum(
            0, np.minimum(loslope, hislope))

    if constamp is False:
        sub = binfrqs[2:int(nfilts) + 2] - binfrqs[0:int(nfilts)]
        wts = np.matmul(np.diag(2 / sub), wts)

    return wts


def _hz2mel(f, htk=False):  # pragma: nocover
    if htk:
        z = 2595 * np.log10(1 + f / 700)
    else:
        f_0 = 0.0
        f_sp = 200 / 3
        brkfrq = 1000
        brkpt = (brkfrq - f_0) / f_sp
        logstep = np.exp(np.log(6.4) / 27.0)

        f = np.array(f, ndmin=1)
        z = np.zeros((f.shape[0], ))

        for i in range(f.shape[0]):
            if f[i] < brkpt:
                z[i] = (f[i] - f_0) / f_sp
            else:
                z[i] = brkpt + (np.log(f[i] / brkfrq) / np.log(logstep))
    return z


def _mel2hz(z, htk=False):  # pragma: nocover
    if htk:
        f = 700 * (np.power(10, z / 2595) - 1)
    else:
        f_0 = 0
        f_sp = 200/3
        brkfrq = 1000
        brkpt = (brkfrq - f_0) / f_sp
        logstep = np.exp(np.log(6.4) / 27.0)

        z = np.array(z, ndmin=1)
        f = np.zeros((z.shape[0], ))

        for i in range(z.shape[0]):
            if z[i] < brkpt:
                f[i] = f_0 + f_sp * z[i]
            else:
                f[i] = brkfrq * np.exp(np.log(logstep) * (z[i] - brkpt))
    return f


def _rastafilt(x):
    numer = np.arange(-2, 3)
    numer = -numer / np.sum(numer ** 2)
    denom = np.array([1, -0.94])

    zi = scipy.signal.lfilter_zi(numer, 1)
    y = np.zeros((x.shape))
    for i in range(x.shape[0]):
        y1, zi = scipy.signal.lfilter(
            numer, 1, x[i, 0:4], axis=0, zi=zi * x[i, 0])
        y1 = y1*0
        y2, _ = scipy.signal.lfilter(
            numer, denom, x[i, 4:x.shape[1]], axis=0, zi=zi)
        y[i, :] = np.append(y1, y2)
    return y


def _postaud(x, fmax, fbtype='bark', broaden=False):
    nbands, nframes = x.shape
    nfpts = int(nbands + 2 * broaden)

    if fbtype == 'bark':
        bandcfhz = _bark2hz(
            np.linspace(0, _hz2bark(fmax), nfpts))
    elif fbtype == 'mel':  # pragma: nocover
        bandcfhz = _mel2hz(
            np.linspace(0, _hz2mel(fmax), nfpts))
    elif fbtype == 'htkmel' or fbtype == 'fcmel':  # pragma: nocover
        bandcfhz = _mel2hz(
            np.linspace(0, _hz2mel(fmax, htk=True), nfpts), htk=True)

    bandcfhz = bandcfhz[broaden:(nfpts - broaden)]

    fsq = np.power(bandcfhz, 2)
    ftmp = fsq + 1.6e5
    eql = np.power(fsq / ftmp, 2) * (fsq + 1.44e6) / (fsq + 9.61e6)

    z = np.multiply(np.tile(eql, (nframes, 1)).T, x)
    z = np.power(z, 0.33)

    if broaden:  # pragma: nocover
        y = np.zeros((z.shape[0] + 2, z.shape[1]))
        y[0, :] = z[0, :]
        y[1:nbands + 1, :] = z
        y[nbands + 1, :] = z[z.shape[0] - 1, :]
    else:
        y = np.zeros((z.shape[0], z.shape[1]))
        y[0, :] = z[1, :]
        y[1:nbands - 1, :] = z[1:z.shape[0] - 1, :]
        y[nbands - 1, :] = z[z.shape[0] - 2, :]

    return y, eql


def _dolpc(x, modelorder=8):
    nbands, nframes = x.shape
    ncorr = 2 * (nbands - 1)
    R = np.zeros((ncorr, nframes))

    R[0:nbands, :] = x
    for i in range(nbands - 1):
        R[i + nbands - 1, :] = x[nbands - (i + 1), :]

    r = scipy.fftpack.ifft(R.T).real.T
    r = r[0:nbands, :]

    y = np.ones((nframes, modelorder + 1))
    e = np.zeros((nframes, 1))

    for i in range(nframes):
        y_tmp, e_tmp, _ = _levinson(
            r[:, i], modelorder, allow_singularity=True)
        if modelorder != 0:
            y[i, 1:modelorder + 1] = y_tmp
        e[i, 0] = e_tmp

    y = y.T / (np.tile(e.T, (modelorder + 1, 1)) + 1e-8)

    return y


# TODO optimize: this is the most inefficient function in the
# processor, takes about half of the compute time. Complexity is
# o(N**2 + N). See scipy.linalg.solve_toeplitz: can be
# solve_toeplitz((r[:-1], r.conj()[:-1]), -r[1:]) but the function does
# not return prediction error...
def _levinson(r, order=None, allow_singularity=False):
    r"""Levinson-Durbin recursion.

    Find the coefficients of a length(r)-1 order autoregressive linear
    process

    Parameters
    ----------
    r : numpy array
        Autocorrelation sequence of length N + 1 (first element being
        the zero-lag autocorrelation)
    order : int, optional
        Requested order of the autoregressive coefficients, default is N
    allow_singularity : bool, optional
        False by default. Other implementations may be True (e.g., octave)

    Returns
    -------

    * the `N+1` autoregressive coefficients :math:`A=(1, a_1...a_N)`
    * the prediction errors
    * the `N` reflections coefficients values

    This algorithm solves the set of complex linear simultaneous equations
    using Levinson algorithm.

    .. math::

        \bold{T}_M \left(\begin{array}{c} 1 \\ \bold{a}_M \end{array} \right) =
        \left( \begin{array}{c} \rho_M \\ \bold{0}_M  \end{array} \right)

    where :math:`\bold{T}_M` is a Hermitian Toeplitz matrix with elements
    :math:`T_0, T_1, \dots ,T_M`.

    .. note:: Solving this equations by Gaussian elimination would
        require :math:`M^3` operations whereas the levinson algorithm
        requires :math:`M^2+M` additions and :math:`M^2+M` multiplications.

    This is equivalent to solve the following symmetric Toeplitz system of
    linear equations

    .. math::

        \left( \begin{array}{cccc}
        r_1 & r_2^* & \dots & r_{n}^*\\
        r_2 & r_1^* & \dots & r_{n-1}^*\\
        \dots & \dots & \dots & \dots\\
        r_n & \dots & r_2 & r_1 \end{array} \right)
        \left( \begin{array}{cccc}
        a_2\\
        a_3 \\
        \dots \\
        a_{N+1}  \end{array} \right)
        =
        \left( \begin{array}{cccc}
        -r_2\\
        -r_3 \\
        \dots \\
        -r_{N+1}  \end{array} \right)

    where :math:`r = (r_1 ... r_{N+1})` is the input autocorrelation
    vector, and :math:`r_i^*` denotes the complex conjugate of
    :math:`r_i`. The input r is typically a vector of autocorrelation
    coefficients where lag 0 is the first element :math:`r_1`.


    Examples
    --------

    >>> import numpy as np
    >>> from shennong.processor.rastaplp import _levinson
    >>> T = np.array([3., -2+0.5j, .7-1j])
    >>> a, e, k = _levinson(T)
    >>> a
    array([0.86315789+0.03157895j, 0.34736842+0.21052632j])
    >>> e
    1.322105263157895
    >>> k
    array([0.66666667-0.16666667j, 0.34736842+0.21052632j])

    Notes
    -----

    This function is taken from the Python spectrum package
    (https://github.com/cokelaer/spectrum) and is under BSD-3-Clause
    license. **Copyright 2011, Thomas Cokelaer**.

    """
    T0 = np.real(r[0])
    T = r[1:]
    M = len(T)

    if order is None:
        M = len(T)
    else:
        if order > M:  # pragma: nocover (checked by RastaPlpProcessor.order)
            raise ValueError(
                f'order ({order}) must be less than size of '
                f'the input data ({M})')
        M = order

    realdata = np.isrealobj(r)
    dtype = float if realdata else complex
    A = np.zeros(M, dtype=dtype)
    ref = np.zeros(M, dtype=dtype)

    P = T0

    for k in range(0, M):
        save = T[k]
        if k == 0:
            temp = -save / P
        else:
            # save += sum([A[j]*T[k-j-1] for j in range(0,k)])
            for j in range(0, k):
                save = save + A[j] * T[k-j-1]
            temp = -save / P

        if realdata:
            P *= 1. - temp ** 2
        else:
            P *= 1. - (temp.real ** 2 + temp.imag ** 2)

        if P <= 0 and allow_singularity is False:  # pragma: nocover
            raise ValueError("singular matrix")

        A[k] = temp
        ref[k] = temp  # save reflection coeff at each step
        if k == 0:
            continue

        khalf = (k+1) // 2
        if realdata:
            for j in range(0, khalf):
                kj = k-j-1
                save = A[j]
                A[j] = save + temp * A[kj]
                if j != kj:
                    A[kj] += temp*save
        else:
            for j in range(0, khalf):
                kj = k-j-1
                save = A[j]
                A[j] = save + temp * A[kj].conjugate()
                if j != kj:
                    A[kj] = A[kj] + temp * save.conjugate()

    return A, P, ref


def _lpc2cep(a, nout):
    nin, ncol = a.shape

    cep = np.zeros((nout, ncol))
    cep[0, :] = -np.log(a[0, :])

    norm_a = a / (np.tile(a[0, :], (nin, 1)) + 1e-8)

    for n in range(1, nout):
        _sum = 0
        for m in range(1, n):
            _sum += (n - m) * norm_a[m, :] * cep[(n - m), :]

        cep[n, :] = -(norm_a[n, :] + _sum / n)

    return cep


def _lpc2spec(lpcas, nout, fmout=False):
    rows, cols = lpcas.shape
    order = rows - 1

    gg = lpcas[0, :]
    aa = lpcas / np.tile(gg, (rows, 1))

    # Calculate the actual z-plane polyvals: nout points around unit circle
    tmp_1 = np.array(np.arange(0, nout), ndmin=2).T
    tmp_1 = -1j * tmp_1 * np.pi / (nout - 1)
    tmp_2 = np.array(np.arange(0, order + 1), ndmin=2)
    zz = np.exp(np.matmul(tmp_1, tmp_2))

    # Actual polyvals, in power (mag^2)
    features = np.divide(
        np.power(np.divide(1, np.abs(np.matmul(zz, aa))), 2),
        np.tile(gg, (nout, 1)))
    F = np.zeros((cols, int(np.ceil(rows / 2))))
    M = F

    if fmout is True:  # pragma: nocover
        for c in range(cols):
            aaa = aa[:, c]
            rr = np.roots(aaa)
            ff_tmp = np.angle(rr)
            ff = np.array(ff_tmp, ndmin=2).T
            zz = np.exp(np.multiply(1j, np.matmul(ff, np.array(
                np.arange(0, aaa.shape[0]), ndmin=2))))
            mags = np.sqrt(np.divide(np.power(np.divide(
                1, np.abs(np.matmul(
                    zz, np.array(aaa, ndmin=2).T))), 2), gg[c]))

            ix = np.argsort(ff_tmp)
            dummy = np.sort(ff_tmp)
            tmp_F_list = []
            tmp_M_list = []
            for i in range(ff.shape[0]):
                if dummy[i] > 0:
                    tmp_F_list = np.append(tmp_F_list, dummy[i])
                    tmp_M_list = np.append(tmp_M_list, mags[ix[i]])

            M[c, 0:tmp_M_list.shape[0]] = tmp_M_list
            F[c, 0:tmp_F_list.shape[0]] = tmp_F_list

    return features, F, M


def _spec2cep(spec, ncep=13, dcttype=2):
    nrow = spec.shape[0]
    dctm = np.zeros((ncep, nrow))

    if dcttype == 2 or dcttype == 3:
        for i in range(ncep):
            dctm[i, :] = (
                np.cos(i * np.arange(1, 2 * nrow, 2) / (2 * nrow) * np.pi)
                * np.sqrt(2 / nrow))

        if dcttype == 2:
            dctm[0, :] = dctm[0, :] / np.sqrt(2)

    elif dcttype == 4:  # pragma: nocover
        for i in range(ncep):
            dctm[i, :] = (2 * np.cos(
                np.pi * (i * np.arange(1, nrow + 1)) / (nrow + 1)))
            dctm[i, 0] += 1
            dctm[i, nrow - 1] *= np.power(-1, i)
        dctm /= 2 * (nrow + 1)

    else:  # pragma: nocover
        for i in range(ncep):
            dctm[i, :] = np.divide(
                np.multiply(np.cos(np.multiply(
                    np.divide(np.multiply(i, np.arange(0, nrow)), (nrow - 1)),
                    np.pi)), 2), 2 * (nrow - 1))
        dctm[:, 0] = dctm[:, 0] * 0.5
        dctm[:, int(nrow - 1)] = dctm[:, int(nrow - 1)] * 0.5

    cep = np.matmul(dctm, np.log(spec + 1e-8))

    return cep, dctm


def _lifter(x, log, lift=0.6, inverse=False):
    ncep = x.shape[0]

    if lift == 0:  # pragma: nocover
        return x

    if lift < 0:  # pragma: nocover
        log.warning(
            'HTK liftering does not support yet; default liftering')
        lift = 0.6

    liftwts = np.power(np.arange(1, ncep), lift)
    liftwts = np.append(1, liftwts)

    if inverse:  # pragma: nocover
        liftwts = 1 / liftwts

    y = np.matmul(np.diag(liftwts), x)
    return y


class RastaPlpProcessor(FramesProcessor):
    def __init__(self, sample_rate=16000, do_rasta=True, order=8,
                 frame_shift=0.01, frame_length=0.025, dither=1.0,
                 preemph_coeff=0.97, remove_dc_offset=True,
                 window_type='povey', round_to_power_of_two=True,
                 blackman_coeff=0.42, snip_edges=True):
        super().__init__(
            sample_rate=sample_rate,
            frame_shift=frame_shift,
            frame_length=frame_length,
            dither=dither,
            preemph_coeff=preemph_coeff,
            remove_dc_offset=remove_dc_offset,
            window_type=window_type,
            round_to_power_of_two=round_to_power_of_two,
            blackman_coeff=blackman_coeff,
            snip_edges=snip_edges)

        self.do_rasta = do_rasta
        self.order = order

    @property
    def name(self):
        return 'rasta-plp'

    @property
    def ndims(self):
        return 13 if self.order == 0 else self.order + 1

    @property
    def do_rasta(self):
        """If False, just calculate the PLP, default to True"""
        return self._do_rasta

    @do_rasta.setter
    def do_rasta(self, value):
        self._do_rasta = bool(value)

    @property
    def order(self):
        """Order of the PLP model

        Must be an integer in [0, 12], 0 means no PLP

        """
        return self._order

    @order.setter
    def order(self, value):
        if not isinstance(value, int) or value < 0 or value > 12:
            raise ValueError('order must be an integer in [0, 12]')
        self._order = value

    def _power_spectrum(self, signal, block_size=64):
        num_frames = kaldi.feat.window.num_frames(
            signal.nsamples, self._frame_options)
        window_size = self._frame_options.padded_window_size()
        window_function = kaldi.feat.window.FeatureWindowFunction.from_options(
            self._frame_options)

        # convert the audio input to a Kaldi vector view
        kaldi_signal = kaldi.matrix.SubVector(signal.data)

        # preallocate the power spectrum matrix
        power_spectrum = np.empty(
            (1 + window_size // 2, num_frames), dtype=np.complex)

        # preallocate frames buffer
        single_frame = kaldi.matrix.Vector(window_size)
        buffer_frames = np.empty((window_size, block_size), dtype=np.float32)

        for min_frame in range(0, num_frames, block_size):
            max_frame = min(min_frame + block_size, num_frames)

            for i in range(min_frame, max_frame):
                # extract the frame i with Kaldi (result goes in single_frame)
                kaldi.feat.window.extract_window(
                    0, kaldi_signal, i, self._frame_options,
                    window_function, single_frame)
                buffer_frames[:, i - min_frame] = single_frame.numpy()

            # compute the power spectrum per block
            power_spectrum[:, min_frame:max_frame] = np.fft.rfft(
                buffer_frames[:, :max_frame - min_frame], axis=0)

        return np.abs(power_spectrum) ** 2

    def _rastaplp(self, signal):
        # compute power spectrum
        pow_spectrum = self._power_spectrum(signal)

        # group to critical bands
        aspectrum = _audspec(pow_spectrum, signal.sample_rate)
        nbands = aspectrum.shape[0]

        if self.do_rasta:
            # log domain, add an epsilon in case of log(0)
            nl_aspectrum = np.log(aspectrum + 1e-8)
            # rasta filtering
            ras_nl_aspectrum = _rastafilt(nl_aspectrum)
            # inverse log
            aspectrum = np.exp(ras_nl_aspectrum)

        postspectrum, _ = _postaud(aspectrum, signal.sample_rate / 2)

        lpcas = _dolpc(postspectrum, self.order)
        cepstra = _lpc2cep(lpcas, self.order + 1)

        if self.order > 0:
            lpcas = _dolpc(postspectrum, self.order)
            cepstra = _lpc2cep(lpcas, self.order + 1)
            spectra, F, M = _lpc2spec(lpcas, nbands)
        else:
            spectra = postspectrum
            cepstra, _ = _spec2cep(spectra)

        cepstra = _lifter(cepstra, self.log, 0.6)
        return cepstra

    def process(self, signal):
        # ensure the signal is correct
        if signal.nchannels != 1:
            raise ValueError(
                'signal must have one dimension, but it has {}'
                .format(signal.nchannels))

        if self.sample_rate != signal.sample_rate:
            raise ValueError(
                'processor and signal mismatch in sample rates: '
                '{} != {}'.format(self.sample_rate, signal.sample_rate))

        # force the signal to be int16
        signal = signal.astype(np.int16)

        # extract the features
        data = self._rastaplp(signal)

        return Features(
            data.T.astype(np.float32),
            self.times(data.T.shape[0]),
            properties=self.get_properties())
