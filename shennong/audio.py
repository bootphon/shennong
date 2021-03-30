"""Provides the :class:`Audio` class that handles audio signals

.. note::

   Supports all audio format from ffmpeg (wav, mp3, flac, etc...). See
   https://www.ffmpeg.org/general.html#File-Formats for details.

The :class:`Audio` class allows to load, save and manipulate
multichannels audio data. The underlying audio samples can be of one
of the following types (with the corresponding min and max):

    ========== =========== ===========
    Type       Min         Max
    ========== =========== ===========
    np.int16   -32768      +32767
    np.int32   -2147483648 +2147483647
    np.float32 -1.0        +1.0
    np.float64 -1.0        +1.0
    ========== =========== ===========

When loading an audio file with :func:`Audio.load`, those min/max are
expected to be respected. When creating an :class:`Audio` instance
from a raw data array, the ``validate`` parameter in the class
constructor and the method :func:`Audio.is_valid` make sure the data
type and min/max are respected.

Examples
--------

>>> import os
>>> import numpy as np
>>> from shennong.audio import Audio

Create 1000 samples of a stereo signal at 16 kHz:

>>> audio = Audio(np.random.random((1000, 2)), 16000)
>>> audio.data.shape
(1000, 2)
>>> audio.dtype
dtype('float64')
>>> audio.sample_rate
16000
>>> audio.nchannels
2
>>> audio.duration
0.0625

Resample the signal to 8 kHz and convert it to 16 bits integers:

>>> audio2 = audio.resample(8000).astype(np.int16)
>>> audio2.sample_rate
8000
>>> audio2.duration == audio.duration
True
>>> audio2.dtype
dtype('int16')
>>> audio2.is_valid()
True

Save the :class:`Audio` instance as a wav file, load an existing wav
file as an :class:`Audio` instance:

>>> audio.save('stereo.wav')
>>> audio3 = Audio.load('stereo.wav')
>>> audio == audio3
True
>>> os.remove('stereo.wav')

Extract mono signal from a stereo one (`left` and `right` are instances of
:class:`Audio` as well):

>>> left = audio.channel(0)
>>> right = audio.channel(1)
>>> left.duration == right.duration == audio.duration
True
>>> left.nchannels == right.nchannels == 1
True

"""

import collections
import functools
import io
import os
import warnings

import numpy as np
import scipy.io.wavfile
import scipy.signal
import sox
import pydub


class Audio:
    """Create an audio signal with the given `data` and `sample_rate`

    Attributes
    ----------
    data : numpy array, shape = [nsamples, nchannels]
        The waveform audio signal, must be of one of the supported
        types (see above)
    sample_rate : float
        The sample frequency of the `data`, in Hertz
    validate : bool, optional
        When True, make sure the underlying data is valid (see
        :meth:`is_valid`), default to True

    Raises
    ------
    ValueError
        If `validate` is True and the audio data if not valid (see
        :meth:`is_valid`)

    """
    _metawav = collections.namedtuple(
        '_metawav', 'nchannels sample_rate nsamples duration')
    """A structure to store wavs metadata, see :meth:`Audio.scan`"""

    def __init__(self, data, sample_rate, validate=True):
        self._sample_rate = int(sample_rate)

        # force shape (n, 1) to be (n,)
        self._data = (
            data[:, 0] if data.ndim > 1 and data.shape[1] == 1 else data)

        if validate and not self.is_valid():
            raise ValueError(f'invalid audio data for type {self.dtype}')

    def __eq__(self, other):
        if self.sample_rate != other.sample_rate:
            return False
        return np.array_equal(self.data, other.data)

    @property
    def data(self):
        """The numpy array of audio data"""
        return self._data

    @property
    def sample_rate(self):
        """The sample frequency of the signal in Hertz"""
        return self._sample_rate

    @property
    def duration(self):
        """The duration of the signal in seconds"""
        return self.nsamples / self.sample_rate

    @property
    def nchannels(self):
        """The number of audio channels in the signal"""
        if self.data.ndim == 1:
            return 1
        return self.data.shape[1]

    @property
    def nsamples(self):
        """The number of samples in the signal"""
        return self.data.shape[0]

    @property
    def shape(self):
        """Return the shape of the underlying data"""
        return self.data.shape

    @property
    def dtype(self):
        """The numeric type of samples"""
        return self.data.dtype

    @property
    def precision(self):
        """The number of bits per sample"""
        return self.dtype.itemsize * 8

    @classmethod
    @functools.lru_cache()
    def scan(cls, filename):
        """Returns the audio metadata without loading the file

        Returns a Python namespace (a named tuple) `metadata` with the
        following fields:

          - metadata.nchannels : int, number of channels
          - metadata.sample_rate : int, sample frequency in Hz
          - metadata.nsamples : int, number of audio samples in the file
          - metadata.duration : float, audio duration in seconds

        This method is usefull to access metadata of an audio file without
        loading it into memory, far more faster than :func:`load`.

        Parameters
        ----------
        filename : str
            Audio filename on which to retrieve metadata, must be
            an existing file

        Returns
        -------
        metadata : namespace
            A namespace with fields as described above

        Raises
        ------
        ValueError
            If the `filename` is not a valid audio file that ffmpeg can
            process.

        """
        filename = str(filename)
        if not os.path.isfile(filename):
            raise ValueError(f'{filename}: file not found')

        try:
            info = pydub.utils.mediainfo(filename)
            return cls._metawav(
                int(info['channels']),
                int(info['sample_rate']),
                int(int(info['sample_rate']) * float(info['duration'])),
                float(info['duration']))
        except Exception:
            raise ValueError(f'cannot scan audio file {filename}') from None

    # we use a memoize cache because Audio.load is often called to
    # load only segments of a file. So the cache avoid to reload again
    # and again the same file to extract only a chunk of it. A little
    # maxsize is enough because access to audio chunks are usually
    # ordered.
    @classmethod
    @functools.lru_cache(maxsize=2)
    def load(cls, filename):
        """Creates an `Audio` instance from a WAV file

        Parameters
        ----------
        filename : str
            Path to the audio file to load, must be an existing file

        Returns
        -------
        audio : Audio
            The Audio instance initialized from the `filename`

        Raises
        ------
        ValueError
            If the `filename` is not a valid audio file.

        """
        filename = str(filename)
        if not os.path.isfile(filename):
            raise ValueError(f'{filename}: file not found')

        # load the audio signal
        try:
            # first try with scipy. It only supports wav files, but support
            # float32 wavs (which pydub/ffmpeg or sox don't support)
            rate, data = scipy.io.wavfile.read(filename)
        except ValueError:
            try:
                # if scipy failed (mostly because it is not a WAV file), give a
                # try to ffmpeg with pydub
                segment = pydub.AudioSegment.from_file(filename)
                rate = segment.frame_rate
                data = np.atleast_2d(np.array(
                    [c.get_array_of_samples()
                     for c in segment.split_to_mono()])).T
            except pydub.exceptions.PydubException as err:
                raise ValueError(
                    f'{filename}: cannot read file, {err}') from None

        return cls(data, rate, validate=False)

    def save(self, filename):
        """Saves the audio data to a `filename`

        Parameters
        ----------
        filename : str
            The audio file to create, format is guessed from extension

        Raises
        ------
        ValueError
            If the file already exists or is unreachable

        """
        filename = str(filename)
        if os.path.isfile(filename):
            raise ValueError(f'{filename}: file already exists')

        if '.' not in filename:
            raise ValueError(
                f'{filename}: cannot write audio file without extension')
        extension = filename.split('.')[-1]

        if extension.lower() == 'wav':
            # saving wav files using scipy
            try:
                scipy.io.wavfile.write(filename, self.sample_rate, self.data)
            except ValueError as err:  # pragma: nocover
                raise ValueError(
                    f'{filename}: cannot write file, {err}') from None
        else:
            # all other audio extensions are handled by pydub/ffmpeg.
            self._aspydub().export(filename, format=extension)

    def _aspydub(self):
        """Converts the audio to a pydub.AudioSegment instance"""
        with io.BytesIO() as wav:
            scipy.io.wavfile.write(wav, self.sample_rate, self.data)
            wav.seek(0)
            return pydub.AudioSegment.from_wav(wav)

    def channel(self, index):
        """Builds a mono signal from a multi-channel one

        Parameters
        ----------
        index : int
            The audio channel to extract from the original signal

        Returns
        -------
        mono : Audio
            The extracted single-channel data

        Raises
        ------
        ValueError
            If `index` >= :func:`nchannels`

        """
        if index == 0 and self.nchannels == 1:
            return self

        if index >= self.nchannels:
            raise ValueError(
                f'not enough channels ({self.nchannels}) to extract '
                f'the index {index} (indices count starts at 0)')

        return Audio(self.data[:, index], self.sample_rate)

    def resample(self, sample_rate, backend='sox'):
        """Returns the audio signal resampled at the given `sample_rate`

        This method first rely on `pysox
        <https://github.com/rabitt/pysox>`_ (excepted if `backend` is
        'scipy') and, if sox is not installed on your system or
        anything goes wrong it falls back to `scipy.signal.resample`.

        The sox backend is very fast and accurate but relies on an
        external binary whereas scipy backend can be very slow but
        works in pure Python.

        Parameters
        ----------
        sample_rate : int
            The sample frequency used to resample the signal, in Hz

        Returns
        -------
        audio : Audio
            An Audio instance containing the resampled signal
        backend : str, optional
            The backend to use for resampling, must be 'sox' or
            'scipy', default to 'sox'

        Raises
        ------
        ValueError
            If the `backend` is not 'sox' or 'scipy', or if the
            resampling failed

        """
        if backend not in ('sox', 'scipy'):
            raise ValueError(f'backend must be sox or scipy, it is {backend}')

        if backend == 'sox':
            return self._resample_sox(sample_rate)
        return self._resample_scipy(sample_rate)

    def _resample_sox(self, sample_rate):
        """Resample the audio signal to the given `sample_rate` using sox"""
        try:
            tfm = sox.Transformer()
            tfm.set_output_format(rate=sample_rate)
            data = tfm.build_array(
                input_array=self.data, sample_rate_in=self.sample_rate)

            return Audio(data, sample_rate, validate=False)
        except (sox.core.SoxError, ValueError):
            raise ValueError(f'resampling at {sample_rate} failed!')

    def _resample_scipy(self, sample_rate):
        """Resample the audio signal to the given `sample_rate` using scipy"""
        if sample_rate == self.sample_rate:
            return self

        # number of samples in the resampled signal
        nsamples = int(self.nsamples * sample_rate / self.sample_rate)

        # scipy method issues warnings we want to inhibit
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=FutureWarning)
            data = scipy.signal.resample(self.data, nsamples)

        # resampling cast to float64, reformat to the original dtype
        return Audio(data.astype(self.dtype), sample_rate, validate=False)

    @staticmethod
    def _is_valid_dtype(dtype):
        """Returns True if `dtype` is a supported data type, False otherwise"""
        supported_types = [np.dtype(t) for t in (
            np.int16, np.int32, np.float32, np.float64)]
        return dtype in supported_types

    def is_valid(self):
        """Returns True if the audio data is valid, False otherwise

        An `Audio` instance is valid if the underlying data type
        is supported (must be np.int16, np.int32, np.float32 or
        np.float64), and if the samples min/max are within the
        expected boundaries for the given data type (see above).

        """
        # make sure the data type is valid
        if not self._is_valid_dtype(self.dtype):
            warnings.warn(f'unsupported audio data type: {self.dtype}')
            return False

        # get the theoretical min/max
        if self.dtype is np.dtype(np.int16):
            emin = -2**15
            emax = 2**15 - 1
        elif self.dtype is np.dtype(np.int32):
            emin = -2**31
            emax = 2**31 - 1
        else:  # float32 or float64
            emin = -1
            emax = 1

        # get the data min/max and checks they are within theoretical
        # boundaries
        dmin = np.amin(self.data)
        dmax = np.amax(self.data)
        if dmin < emin or dmax > emax:
            warnings.warn(
                'invalid audio for type {self.dtype}: '
                f'boundaries must be in ({emin}, {emax}) '
                f'but are ({dmin}, {dmax})')
            return False
        return True

    def astype(self, dtype):
        """Returns the audio signal converted to the `dtype` numeric type

        The valid types are np.int16, np.int32, np.float32 or
        np.float64, see above for the types min and max.

        Parameters
        ----------
        dtype : numeric type
            Must be an integer or a floating-point type in the types
            described above.

        Raises
        ------
        ValueError
            If the requested `dtype` is not supported

        """
        # do nothing if we already have the requested dtype
        if self.dtype is np.dtype(dtype):
            return self

        # make sure we support the requested dtype
        if not self._is_valid_dtype(dtype):
            raise ValueError(f'unsupported audio data type: {dtype}')

        # starting from int16
        if self.dtype is np.dtype(np.int16):
            if dtype is np.int32:
                data = self.data * 2**15
            else:  # float32 or float64
                data = self.data / 2**15

        # starting from int32
        elif self.dtype is np.dtype(np.int32):
            if dtype is np.int16:
                data = self.data / 2**15
            else:  # float32 or float64
                data = self.data / 2**30

        # starting from float32 or float64
        else:
            if dtype is np.int16:
                data = self.data * 2**15
            elif dtype is np.int32:
                data = self.data * 2**30
            else:  # float32 or float64
                data = self.data

        return Audio(data.astype(dtype), self.sample_rate, validate=False)

    def segment(self, segments):
        """Returns audio chunks segmented from the original signal

        Parameters
        ----------
        segments : list of pairs of floats
             A list of pairs (tstart, tstop) of the start and stop
             indices (in seconds) of the signal chunks we are going to
             extract. The times `tstart` and `tstop` must be float,
             with `tstart` < `tstop`.

        Returns
        -------
        chunks : list of Audio
            The signal chunks created from the given `segments`

        Raises
        ------
        ValueError
            If one element in `segments` is not a pair of float or if
            `tstart` >= `tstop`. If `segments` is not a list.

        """
        # ensure segments is well formatted
        if not isinstance(segments, list):
            raise ValueError('segments must be a list')
        for segment in segments:
            try:
                if not len(segment) == 2:
                    raise ValueError('segments elements must be pairs')
            except TypeError:
                raise ValueError('segments elements must be pairs')
            if segment[0] >= segment[1]:
                raise ValueError('time indices in segments must be sorted')

        chunks = []
        for segment in segments:
            istart = int(segment[0] * self.sample_rate)
            istop = int(segment[1] * self.sample_rate)
            chunks.append(Audio(
                self.data[istart:istop], self.sample_rate, validate=False))
        return chunks
