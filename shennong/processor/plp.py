"""Provides the PlpProcessor class to extract PLP features

Extract PLP (Perceptual Linear Predictive analysis of speech) from an
audio signal. Uses the Kaldi implementation (see [Hermansky1990]_ and
[kaldi-plp]_). Optionally apply RASTA filtering (see [Herm94]_).

    :class:`~shennong.audio.Audio` ---> PlpProcessor \
    ---> :class:`~shennong.features.Features`

Examples
--------

>>> from shennong.audio import Audio
>>> from shennong.processor.plp import PlpProcessor
>>> audio = Audio.load('./test/data/test.wav')

Initialize the PLP processor with some options. Options can be
specified at construction, or after:

>>> processor = PlpProcessor()
>>> processor.sample_rate = audio.sample_rate

Here we apply RASTA filters

>>> processor.rasta = True

Compute the PLP features with the specified options, the output is an
instance of :class:`~shennong.features.Features`:

>>> plp = processor.process(audio)
>>> type(plp)
<class 'shennong.features.Features'>
>>> plp.shape[1] == processor.num_ceps
True

References
----------

.. [Hermansky1990]
     H. Hermansky, "Perceptual linear predictive (PLP) analysis of speech",
     Journal of the Acoustical Society of America, vol. 87, no. 4, pages
     1738–1752 (1990)`

.. [Herm94]
     H. Hermansky and N. Morgan, "RASTA processing of speech", IEEE Trans. on
     Speech and Audio Proc., vol. 2, no. 4, pp. 578-589, Oct. 1994.

.. [kaldi-plp]
     http://kaldi-asr.org/doc/feat.html#feat_plp

"""

import numpy as np
import scipy.signal
import kaldi.base.math
import kaldi.feat.mel
import kaldi.feat.window
import kaldi.matrix

from shennong import Features
from shennong.processor.base import MelFeaturesProcessor


class RastaFilter:
    """Rasta filter for Rasta PLP implementation

    Reimplemented after [labrosa]_ and [rastapy]_ on a frame by frame basis.
    Original implementation takes the whole signal at once.

    Parameters
    ----------
    size : int
        The dimension of the frames to filter

    References
    ----------

    .. [labrosa]
         https://labrosa.ee.columbia.edu/matlab/rastamat/

    .. [rastapy]
         https://github.com/mystlee/rasta_py

    """
    def __init__(self, size):
        # filter numerator and denominator
        self._num = -np.arange(-2, 3) / np.sum(np.arange(-2, 3) ** 2)
        self._den = np.array([1, -0.94])

        # dimensionality of the frames to filter
        self._size = size

        # the following attributes are initialized in the reset() method
        self._delay = None
        self._count = None
        self._first_frames = None
        self.reset()

    def reset(self):
        """Initializes the filter state"""
        self._count = 0
        self._first_frames = []
        self._delay = np.dstack(
            (scipy.signal.lfilter_zi(self._num, 1),) * self._size).squeeze()

    def filter(self, frame, do_log=True):
        """RASTA filtering of a mel frame

        Parameters
        ----------
        frame : numpy array, shape = [size, 1]
            The frame vector to filter.
        do_log : bool, optional
            When True move to the log domain before filtering, and do inverse
            log after. When False the frame is expected to be in log domain
            already. Default to true.

        Returns
        -------
        filtered : numpy array, shape = [size, 1]
            The filtered frame.

        """
        x = frame
        if do_log:
            x = np.log(x + np.finfo(x.dtype).eps)

        if self._count < 4:
            # stack the 4 first frames into a buffer and initialize the filter
            self._first_frames.append(x)
            y = np.zeros((x.shape))
        if self._count == 3:
            x = np.asarray(self._first_frames)
            _, self._delay = scipy.signal.lfilter(
                self._num, 1, x, zi=self._delay * x[0], axis=0)
        if self._count >= 4:
            y, self._delay = scipy.signal.lfilter(
                self._num, self._den, [x], zi=self._delay, axis=0)

        self._count += 1

        y = np.atleast_2d(y)[0, :].astype(x.dtype)
        if do_log:
            y = np.exp(y)

        return y


def _lpc2cepstrum(lpc_order, lpc, cepstrum):
    """Reimplementation of Kaldi Lpc2Cepstrum

     from src/feat/mel_computations.h (missing in pykadli)

    Parameters
    ----------
    lpc_order : int
       LPC order
    lpc : kaldi.matrix.Vector
       LPC coefficients of size `lpc_order`
    cepstrum : kaldi.matrix.Vector
       Output cepstrum, must be preallocated with size `lpc_order`

    """
    for i in range(lpc_order):
        _sum = 0.0
        for j in range(i):
            _sum += float(i - j) * lpc[j] * cepstrum[i - j - 1]
        cepstrum[i] = -lpc[i] - _sum / float(i + 1)


def _process_window(opts, window_function, window, do_log_energy):
    """Reimplmentation of Kaldi ProcessWindow

    from src/feat/feature-window.h

    This function is already wrapped in pykaldi but without estimation of log
    energy. Because the PLP recipe needs it, it has been reimplemented.

    """
    frame_length = opts.window_size()
    assert window.dim == frame_length

    if opts.dither != 0:
        kaldi.feat.window.dither(window, opts.dither)

    if opts.remove_dc_offset:
        window.add_(-window.sum() / frame_length)

    log_energy = 0
    if do_log_energy:
        log_energy = kaldi.base.math.log(max(
            kaldi.matrix.functions.vec_vec(window, window),
            np.finfo(float).eps))

    if opts.preemph_coeff != 0:
        kaldi.feat.window.preemphasize(window, opts.preemph_coeff)

    window.mul_elements_(window_function.window)

    return log_energy


def _extract_window(
        sample_offset, wave, frame, opts, window_function, window,
        do_log_energy):
    """Reimplmentation of Kaldi ExtractWindow

    from src/feat/feature-window.h

    This function is already wrapped in pykaldi but without estimation of log
    energy. Because the PLP recipe needs it, it has been reimplemented.

    """
    assert sample_offset >= 0 and wave.dim != 0
    frame_length = opts.window_size()
    frame_length_padded = opts.padded_window_size()
    num_samples = sample_offset + wave.dim
    start_sample = kaldi.feat.window.first_sample_of_frame(frame, opts)
    end_sample = start_sample + frame_length

    if opts.snip_edges:
        assert start_sample >= sample_offset and end_sample <= num_samples
    else:
        assert sample_offset == 0 or start_sample >= sample_offset

    if window.dim != frame_length_padded:
        window.resize_(
            frame_length_padded,
            kaldi.matrix.common.MatrixResizeType.UNDEFINED)

    # wave_start and wave_end are start and end indexes into 'wave', for the
    # piece of wave that we're trying to extract.
    wave_start = int(start_sample - sample_offset)
    wave_end = wave_start + frame_length
    if wave_start >= 0 and wave_end <= wave.dim:
        # the normal case -- no edge effects to consider
        window[:frame_length] = wave[wave_start:wave_start+frame_length]
    else:
        # Deal with any end effects by reflection, if needed. This code will
        # only be reached for about two frames per utterance, so we don't
        # concern ourselves excessively with efficiency.
        for s in range(frame_length):
            s_in_wave = s + wave_start
            while s_in_wave < 0 or s_in_wave >= wave.dim:
                # reflect around the beginning or end of the wave.
                # e.g. -1 -> 0, -2 -> 1.
                # dim -> dim - 1, dim + 1 -> dim - 2.
                # the code supports repeated reflections, although this
                # would only be needed in pathological cases.
                if s_in_wave < 0:
                    s_in_wave = -s_in_wave - 1
                else:
                    s_in_wave = 2 * wave.dim - 1 - s_in_wave
            window[s] = wave[s_in_wave]

    if frame_length_padded > frame_length:
        window[frame_length:frame_length_padded] = 0

    return _process_window(
        opts, window_function, window[:frame_length], do_log_energy)


class PlpProcessor(MelFeaturesProcessor):
    """Perceptive linear predictive features"""
    def __init__(self, sample_rate=16000, frame_shift=0.01, frame_length=0.025,
                 rasta=False, dither=1.0, preemph_coeff=0.97,
                 remove_dc_offset=True, window_type='povey',
                 round_to_power_of_two=True, blackman_coeff=0.42,
                 snip_edges=True, num_bins=23, low_freq=20, high_freq=0,
                 vtln_low=100, vtln_high=-500, lpc_order=12, num_ceps=13,
                 use_energy=True, energy_floor=0.0, raw_energy=True,
                 compress_factor=1.0/3.0, cepstral_lifter=22,
                 cepstral_scale=1.0, htk_compat=False):
        # Forward options to MelFeaturesProcessor
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
            snip_edges=snip_edges,
            num_bins=num_bins,
            low_freq=low_freq,
            high_freq=high_freq,
            vtln_low=vtln_low,
            vtln_high=vtln_high)

        self._options = kaldi.feat.plp.PlpOptions()
        self._options.frame_opts = self._frame_options
        self._options.mel_opts = self._mel_options

        self.rasta = rasta
        self.lpc_order = lpc_order
        self.num_ceps = num_ceps
        self.use_energy = use_energy
        self.energy_floor = energy_floor
        self.raw_energy = raw_energy
        self.compress_factor = compress_factor
        self.cepstral_lifter = cepstral_lifter
        self.cepstral_scale = cepstral_scale
        self.htk_compat = htk_compat

        # will store the Rasta filter if `rasta` option is True
        self._rasta_filter = None
        # will store mel banks and equal loudness coeffs for various VTLN warps
        self._mel_banks = {}
        self._equal_loudness = {}

        # the buffers needed by the recipe are allocated in
        # self._reset_buffers()
        self._mel_energies_duplicated = kaldi.matrix.Vector()
        self._autocorr_coeffs = kaldi.matrix.Vector()
        self._lpc_coeffs = kaldi.matrix.Vector()
        self._raw_cepstrum = kaldi.matrix.Vector()
        self._lifter_coeffs = kaldi.matrix.Vector()
        self._idft_bases = kaldi.matrix.Matrix()
        self._log_energy_floor = 0

        # this recipe does not rely on a Kaldi class as all is reimplemented in
        # python/pykaldi
        self._kaldi_processor = None

    @property
    def name(self):
        return 'plp'

    @property
    def rasta(self):
        """Whether to do RASTA filtering"""
        return self._rasta

    @rasta.setter
    def rasta(self, value):
        self._rasta = bool(value)

    @property
    def lpc_order(self):
        """Order of LPC analysis in PLP computation"""
        return self._options.lpc_order

    @lpc_order.setter
    def lpc_order(self, value):
        self._options.lpc_order = value

    @property
    def num_ceps(self):
        """Number of cepstra in PLP computation (including C0)

        Must be positive and  smaller or equal to `lpc_order` + 1.

        """
        return self._options.num_ceps

    @num_ceps.setter
    def num_ceps(self, value):
        value = int(value)
        if value <= 0:
            raise ValueError('num_ceps must be > 0')
        if value > self.lpc_order + 1:
            raise ValueError(
                'We must have num_ceps <= lpc_order+1, but {} > {}+1'.format(
                    value, self.lpc_order))
        self._options.num_ceps = value

    @property
    def use_energy(self):
        """Use energy (instead of C0) for zeroth PLP feature"""
        return self._options.use_energy

    @use_energy.setter
    def use_energy(self, value):
        self._options.use_energy = value

    @property
    def energy_floor(self):
        """Floor on energy (absolute, not relative) in PLP computation"""
        return self._options.energy_floor

    @energy_floor.setter
    def energy_floor(self, value):
        self._options.energy_floor = value

    @property
    def raw_energy(self):
        """If true, compute energy before preemphasis and windowing"""
        return self._options.raw_energy

    @raw_energy.setter
    def raw_energy(self, value):
        self._options.raw_energy = value

    @property
    def compress_factor(self):
        """Compression factor in PLP computation"""
        return np.float32(self._options.compress_factor)

    @compress_factor.setter
    def compress_factor(self, value):
        self._options.compress_factor = value

    @property
    def cepstral_lifter(self):
        """Constant that controls scaling of PLPs"""
        return self._options.cepstral_lifter

    @cepstral_lifter.setter
    def cepstral_lifter(self, value):
        self._options.cepstral_lifter = value

    @property
    def cepstral_scale(self):
        """Scaling constant in PLP computation"""
        return self._options.cepstral_scale

    @cepstral_scale.setter
    def cepstral_scale(self, value):
        self._options.cepstral_scale = value

    @property
    def htk_compat(self):
        """If True, get closer to HTK PLP features

        Put energy or C0 last.

        Warning: Not sufficient to get HTK compatible features (need
        to change other parameters)

        """
        return self._options.htk_compat

    @htk_compat.setter
    def htk_compat(self, value):
        self._options.htk_compat = value

    @property
    def ndims(self):
        return self.num_ceps

    def _reset_buffers(self):
        if self._mel_energies_duplicated.dim != self.num_bins + 2:
            self._mel_energies_duplicated.resize_(
                self.num_bins + 2,
                kaldi.matrix.common.MatrixResizeType.UNDEFINED)

        if self._autocorr_coeffs.dim != self.lpc_order + 1:
            self._autocorr_coeffs.resize_(
                self.lpc_order + 1,
                kaldi.matrix.common.MatrixResizeType.UNDEFINED)

        if self._lpc_coeffs.dim != self.lpc_order:
            self._lpc_coeffs.resize_(
                self.lpc_order,
                kaldi.matrix.common.MatrixResizeType.UNDEFINED)

        if self._raw_cepstrum.dim != self.lpc_order:
            self._raw_cepstrum.resize_(
                self.lpc_order,
                kaldi.matrix.common.MatrixResizeType.UNDEFINED)

        if (
                self.cepstral_lifter != 0 and
                self._lifter_coeffs.dim != self.num_ceps):
            self._lifter_coeffs.resize_(self.num_ceps)
            kaldi.feat.mel.compute_lifter_coeffs(
                self.cepstral_lifter, self._lifter_coeffs)
        elif self.cepstral_lifter == 0:
            self._lifter_coeffs.resize_(0)

        self._idft_bases = kaldi.feat.functions.init_idft_bases(
            self.lpc_order + 1, self.num_bins + 2)

        if self.energy_floor > 0:
            self._log_energy_floor = kaldi.base.math.log(self.energy_floor)

        if self.rasta:
            self._rasta_filter = RastaFilter(self.num_bins)

    def _get_mel_banks(self, vtln_warp):
        """Returns MEL banks for a given VTLN warp

        Banks are generated on demand and stored for fastest retrieval.

        """
        try:
            return self._mel_banks[vtln_warp]
        except KeyError:
            mel_banks = kaldi.feat.mel.MelBanks(
                self._mel_options, self._frame_options, vtln_warp)
            self._mel_banks[vtln_warp] = mel_banks
            return mel_banks

    def _get_equal_loudness(self, vtln_warp):
        """Returns equal loudness coefficient for a given VTLN warp

        Coefficients are generated on demand and stored for fastest retrieval.

        """
        mel_banks = self._get_mel_banks(vtln_warp)
        try:
            return self._equal_loudness[vtln_warp]
        except KeyError:
            ans = kaldi.feat.mel.get_equal_loudness_vector(mel_banks)
            self._equal_loudness[vtln_warp] = ans
            return ans

    def _compute(self, signal, vtln_warp):
        """Reimplementation of Kaldi OfflineFeatureTpl::Compute

        From src/feat/feature-common.h. Reimplementation needed to integrate
        Rasta filtering.

        """
        rows_out = kaldi.feat.window.num_frames(
            signal.nsamples, self._frame_options)
        cols_out = self.ndims

        if rows_out == 0:  # pragma: nocover
            return np.zeros((0, 0))

        # force the input signal to be 16 bits integers
        signal = kaldi.matrix.SubVector(signal.astype(np.int16).data)

        # allocate the output data
        output = kaldi.matrix.Matrix(rows_out, cols_out)

        # windowed waveform and windowing function
        window = kaldi.matrix.Vector()
        window_function = kaldi.feat.window.FeatureWindowFunction.from_options(
            self._frame_options)

        # for each frame
        for row in range(rows_out):
            # extract its window and its log energy...
            raw_log_energy = _extract_window(
                0, signal, row, self._frame_options, window_function, window,
                self.use_energy and self.raw_energy)

            # ... and extract PLP with optional Rasta filtering
            self._compute_frame(
                raw_log_energy, vtln_warp, window, output[row, :])

        return output.numpy()

    def _compute_frame(
            self, signal_log_energy, vtln_warp, signal_frame, output_frame):
        """Reimplementation of Kaldi PlpComputer::Computer

        From src/feat/feature-plp.h. This implementation includes Rasta
        filtering.

        """
        assert signal_frame.dim == self._frame_options.padded_window_size()
        assert output_frame.dim == self.ndims

        mel_banks = self._get_mel_banks(vtln_warp)
        equal_loudness = self._get_equal_loudness(vtln_warp)

        assert self.num_ceps <= self.lpc_order + 1  # our num-ceps includes C0

        if self.use_energy and not self.raw_energy:
            signal_log_energy = kaldi.base.math.log(max(
                kaldi.matrix.functions.vec_vec(signal_frame, signal_frame),
                np.finfo(float).eps))

        # split radix FFT not wrapped in pykaldi so fall back to real_fft even
        # if frame length is a power of two...
        kaldi.matrix.functions.real_fft(signal_frame, True)

        # convert the FFT into a power spectrum, elements 0 ...
        # signal_frame.dim/2
        kaldi.feat.functions.compute_power_spectrum(signal_frame)
        power_spectrum = signal_frame[:int(signal_frame.dim / 2 + 1)]

        # convert power spectrum to mel filterbank
        mel_energies = self._mel_energies_duplicated[1:self.num_bins+1]
        mel_banks.compute(power_spectrum, mel_energies)

        if self.rasta:
            # do rasta filtering
            mel_energies[:] = self._rasta_filter.filter(
                mel_energies.numpy(), do_log=True)

        mel_energies.mul_elements_(equal_loudness)
        mel_energies.apply_pow_(self.compress_factor)

        # duplicate first and last elements
        self._mel_energies_duplicated[0] = self._mel_energies_duplicated[1]
        self._mel_energies_duplicated[self.num_bins+1] = \
            self._mel_energies_duplicated[self.num_bins]

        self._autocorr_coeffs.set_zero_()  # in case of NaNs or infs
        self._autocorr_coeffs.add_mat_vec_(
            1.0, self._idft_bases,
            kaldi.matrix.common.MatrixTransposeType.NO_TRANS,
            self._mel_energies_duplicated, 0.0)

        residual_log_energy = kaldi.feat.mel.compute_lpc(
            self._autocorr_coeffs, self._lpc_coeffs)
        residual_log_energy = max(residual_log_energy, np.finfo(float).eps)

        _lpc2cepstrum(self.lpc_order, self._lpc_coeffs, self._raw_cepstrum)
        output_frame[1:self.num_ceps] = self._raw_cepstrum[:self.num_ceps - 1]
        output_frame[0] = residual_log_energy

        if self.cepstral_lifter != 0:
            output_frame.mul_elements_(self._lifter_coeffs)

        if self.cepstral_scale != 1.0:
            output_frame.scale_(self.cepstral_scale)

        if self.use_energy:
            if (
                    self.energy_floor > 0 and
                    signal_log_energy < self._log_energy_floor):
                signal_log_energy = self._log_energy_floor
            output_frame[0] = signal_log_energy

        if self.htk_compat:  # reorder the features
            log_energy = output_frame[0]
            for i in range(self.num_ceps - 1):
                output_frame[i] = output_frame[i+1]
            output_frame[self.num_ceps - 1] = log_energy

    def process(self, signal, vtln_warp=1.0):
        """Compute Rasta-PLP features with the specified options

        Do an optional feature-level vocal tract length normalization
        (VTLN) when `vtln_warp` != 1.0.

        Parameters
        ----------
        signal : Audio, shape = [nsamples, 1]
            The input audio signal to compute the features on, must be
            mono
        vtln_warp : float, optional
            The VTLN warping factor to be applied when computing
            features. Be 1.0 by default, meaning no warping is to be
            done.

        Returns
        -------
        features : `Features`, shape = [nframes, `ndims`]
            The computed features, output will have as many rows as there
            are frames (depends on the specified options `frame_shift`
            and `frame_length`).

        Raises
        ------
        ValueError
            If the input `signal` has more than one channel (i.e. is
            not mono). If `sample_rate` != `signal.sample_rate`.

        """
        # ensure the signal is correct
        if signal.nchannels != 1:
            raise ValueError(
                'signal must have one dimension, but it has {}'
                .format(signal.nchannels))

        if self.sample_rate != signal.sample_rate:
            raise ValueError(
                'processor and signal mismatch in sample rates: '
                '{} != {}'.format(self.sample_rate, signal.sample_rate))

        # extract the PLP features
        self._reset_buffers()
        data = self._compute(signal, vtln_warp)

        return Features(
            data, self.times(data.shape[0]),
            properties=self.get_properties(vtln_warp=vtln_warp),
            validate=False)
