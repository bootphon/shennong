"""Utility functions to read audio WAV files"""

import os
import scipy.io.wavfile
import wave


def read(wav_file, safe=False):
    """Load a WAV file as a numpy array

    Parameters
    ----------
    wav_file : str
        Filename of the WAV to load, must be an existing file
    safe : bool, optional
        When True, ensures the WAV file is mono, sampled at 16kHz and
        at a 16 bits resolution. When False (default) this is not
        checked.

    Returns
    -------
    samplerate : int
        The sample rate (in Hz) of the loaded signal
    signal : np.array
        The audio signal loaded from the `wav_file`

    Raises
    ------
    ValueError
        If the `wav_file` is not a valid WAV file or, when `safe` is
        True, if the file is not a 16kHz, 16 bits mono file.

    See Also
    --------
    scipy.io.wavfile.read

    """
    if not os.path.isfile(wav_file):
        raise ValueError('file not found: {}'.format(wav_file))

    if safe is True:
        check_format(wav_file, framerate=16000, bitrate=16, nchannels=1)

    try:
        return scipy.io.wavfile.read(wav_file)
    except ValueError:
        raise ValueError('{}: cannot read file, is it a wav?'.format(wav_file))


def check_format(wav_file, framerate=16000, bitrate=16, nchannels=1):
    """Return True if the `wav_file` has the requested format

    If the format of the `wav_file` is not as expected, raises a ValueError.

    Parameters
    ----------
    wav_file : str
        Filename of the WAV to load, must be an existing file
    framerate : int, optional
        The desired frame rate in Hz
    bitrate : int, optional
        The desired bit rate
    nchannels : int, optional
        The desired number of channels (1 for mono, 2 for stereo)

    Returns
    -------
    True when the wav format is as expected

    Raises
    ------
    ValueError
        If one of the requested specification is not met in the
        `wav_file`.

    """
    error = []
    try:
        with wave.open(wav_file, 'rb') as fp:
            # check frame rate
            _framerate = fp.getframerate()
            if _framerate != framerate:
                error.append('framerate is {} but must be {}'.format(
                    _framerate, framerate))

            # check bit rate
            _bitrate = fp.getsampwidth() * 8
            if _bitrate != bitrate:
                error.append('bitrate is {} but must be {}'.format(
                    _bitrate, bitrate))

            # check number of channels
            _nchannels = fp.getnchannels()
            if _nchannels != nchannels:
                error.append('nchannels is {} but must be {}'.format(
                    _nchannels, nchannels))
    except wave.Error:
        raise ValueError('{}: cannot read file, is it a wav?'.format(wav_file))

    if len(error) != 0:
        raise ValueError('{}: '.format(wav_file) + ', '.join(error))
    else:
        return True
