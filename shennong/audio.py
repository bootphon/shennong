"""Provides the `AudioData` class that handles audio signals

The `AudioData` class allows to load, save and manipulate
multichannels audio data. `AudioData` is the input to feature
extraction models.

.. note::

   For now, only WAV files are supported for input/output.

Examples
--------

>>> import os
>>> import numpy as np
>>> from shennong.audio import AudioData

Create 1000 samples of a stereo signal at 16 kHz:

>>> audio = AudioData(np.random.random((1000, 2)), 16000)
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

Save the `AudioData` instance as a wav file, load an existing wav
file as an `AudioData` instance:

>>> audio.save('stereo.wav')
>>> audio2 = AudioData.load('stereo.wav')
>>> audio == audio2
True
>>> os.remove('stereo.wav')

Extract mono signal from a stereo one (`left` and `right` are
instances of AudioData as well):

>>> left = audio.channel(0)
>>> right = audio.channel(1)
>>> left.duration == right.duration == audio.duration
True
>>> left.nchannels == right.nchannels == 1
True

"""

import os
import numpy as np
import scipy.signal
import scipy.io.wavfile
import warnings

from shennong.utils import get_logger


class AudioData:
    """Audio signal with associated sample rate and dtype

    Attributes
    ----------
    data : numpy array, shape = [nsamples, nchannels]
        The waveform audio signal
    sample_rate : float
        The sample frequency of the `data`, in Hertz
    validate : bool, optional
        When True, make sure the underlying data is valid (see
        :method:`is_valid`), default to True

    Raises
    ------
    ValueError
        If `validate` is True and the audio data if not valid (see
        :method:`is_valid`)

    """
    _log = get_logger(__name__)

    def __init__(self, data, sample_rate, validate=True):
        self._data = data
        self._sample_rate = sample_rate

        if validate and not self.is_valid():
            raise ValueError(
                'invalid audio data for type {}'.format(self.dtype))

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
        """The sample frequency of the signal, in Hertz"""
        return self._sample_rate

    @property
    def duration(self):
        """The duration of the signal, in seconds"""
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
    def dtype(self):
        """The numeric type of samples"""
        return self.data.dtype

    @classmethod
    def load(cls, wav_file):
        """Initialize an AudioData instance from a WAV file

        Parameters
        ----------
        wav_file : str
            Filename of the WAV to load, must be an existing file

        Returns
        -------
        audio : AudioData
            The AudioData instance initialized from the `wav_file`

        Raises
        ------
        ValueError
            If the `wav_file` is not a valid WAV file.

        """
        if not os.path.isfile(wav_file):
            raise ValueError('{}: file not found'.format(wav_file))

        try:
            # load the audio signal
            cls._log.debug('loading %s', wav_file)
            sample_rate, data = scipy.io.wavfile.read(wav_file)

            # build and return the AudioData instance, we assume the
            # underlying audio samples are valid
            return AudioData(data, sample_rate, validate=False)
        except ValueError:
            raise ValueError(
                '{}: cannot read file, is it a wav?'.format(wav_file))

    def save(self, wav_file):
        """Save the audio data to a `wav_file`

        Parameters
        ----------
        wav_file : str
            The WAV file to create

        Raises
        ------
        ValueError, FileNotFoundError, PermissionError
            If the file already exists or is unreachable

        """
        if os.path.isfile(wav_file):
            raise ValueError(
                '{}: file already exists'.format(wav_file))

        scipy.io.wavfile.write(wav_file, self.sample_rate, self.data)

    def channel(self, index):
        """Build a mono signal from a multi-channel one

        Parameters
        ----------
        index : int
            The audio channel to extract from the original signal

        Returns
        -------
        mono : AudioData
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
                'not enough channels ({}) to extract the index {} '
                '(indices count starts at 0)'.format(
                    self.nchannels, index))

        return AudioData(self.data[:, index], self.sample_rate)

    def resample(self, sample_rate):
        """Returns the audio signal resampled at the given `sample_rate`

        This method relies on :func:`scipy.signal.resample`

        Parameters
        ----------
        sample_rate : int
            The sample frequency used to resample the signal, in Hz

        Returns
        -------
        audio : AudioData
            An AudioData instance containing the resampled signal

        """
        if sample_rate == self.sample_rate:
            return self

        # number of samples in the resampled signal
        nsamples = int(self.nsamples * sample_rate / self.sample_rate)

        # scipy method issues warnings we want to inhibit
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=FutureWarning)
            data = scipy.signal.resample(self.data, nsamples)

        # resampling cast to float64, reformat to the original dtype
        return AudioData(data.astype(self.dtype), sample_rate)

    @staticmethod
    def _is_valid_dtype(dtype):
        """Return True if `dtype` is a supported data type, False otherwise"""
        supported_types = [np.dtype(t) for t in (
            np.int16, np.int32, np.float32, np.float64)]
        return dtype in supported_types

    def is_valid(self):
        """Return True if the audio data are valid, False otherwise

        An AudioData instance is valid if the underlying data type is
        supported (must be np.int16, np.int32, np.float32 or
        np.float64), and if the samples min/max are within the
        expected boundaries for the given data type.

        """
        # make sure the data type is valid
        if not self._is_valid_dtype(self.dtype):
            self._log.warning(
                'unsupported audio data type: {}'.format(self.dtype))
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
        if dmin <= emin or dmax >= emax:
            self._log.warning(
                'invalid audio for type %s: boundaries must be in (%s, %s) '
                'but are (%s, %s)', self.dtype, emin, emax, dmin, dmax)
            return False

        return True

    def astype(self, dtype):
        """Returns the audio signal converted to the `dtype` numeric type

        The valid types are:

        ========== =========== ===========
        Type       Min         Max
        ========== =========== ===========
        np.int16   -32768      +32767
        np.int32   -2147483648 +2147483647
        np.float32 -1.0        +1.0
        np.float64 -1.0        +1.0
        ========== =========== ===========

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
        if self.dtype == dtype:
            return self

        # make sure we support the requested dtype
        if not self._is_valid_dtype(dtype):
            raise ValueError('unsupported audio data type: {}'.format(dtype))

        # starting from int16
        if self.dtype is np.dtype(np.int16):
            if dtype is np.int32:
                data = (self.data * 2**15).astype(dtype)
            else:  # float32 or float64
                data = (self.data / 2**15).astype(dtype)

        # starting from int32
        elif self.dtype is np.dtype(np.int32):
            if dtype is np.int16:
                data = (self.data / 2**15).astype(dtype)
            else:  # float32 or float64
                data = (self.data / 2**31).astype(dtype)

        # starting from float32 or float64
        else:
            if dtype is np.int16:
                data = (self.data * 2**15).astype(dtype)
            elif dtype is np.int32:
                data = (self.data * 2**31).astype(dtype)
            else:  # float32 or float64
                data = self.data.astype(dtype)

        return AudioData(data, self.sample_rate, validate=False)
