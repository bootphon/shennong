"""Provides the `AudioData` class that handles audio signals

The `AudioData` class allows to load, save and manipulate
multichannels audio data. `AudioData` is the input to feature
extraction models.

.. note::

   For now, only WAV files are supported for input/output. Resampling
   is not implemented.

Examples
--------

>>> import os
>>> import numpy as np
>>> from shennong.audio import AudioData

Create 1000 samples of a stereo signal at 16 kHz:

>>> audio = AudioData(np.random.random((1000, 2)), 16000)
>>> audio.data.shape
(1000, 2)
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

from shennong import log


class AudioData:
    """Audio signal with associated sample rate and dtype

    Attributes
    ----------
    data : numpy array, shape = [nsamples, nchannels]
        The waveform audio signal
    sample_rate : float
        The sample frequency of the `data`, in Hertz

    """
    def __init__(self, data, sample_rate):
        self._data = data
        self._sample_rate = sample_rate

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

    @staticmethod
    def load(wav_file):
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
            log.debug('loading %s', wav_file)
            sample_rate, data = scipy.io.wavfile.read(wav_file)

            # build and return the AudioData instance
            return AudioData(data, sample_rate)
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

        # scipy method issues warnings we want to inhibate
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=FutureWarning)
            data = scipy.signal.resample(self.data, nsamples)

        return AudioData(data, sample_rate)
