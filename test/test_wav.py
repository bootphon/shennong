"""Test of the module shennong.features.wav"""

import os
import pytest
import shennong.features.wav as wav


wav_file = os.path.join(os.path.dirname(__file__), 'data', 'test.wav')


@pytest.mark.parametrize('safe', [True, False])
def test_read(safe):
    fs, data = wav.read(wav_file, safe=safe)
    assert fs == 16000
    assert data.shape == (23001,)


def test_read_notwav():
    with pytest.raises(ValueError):
        wav.read(__file__)


def test_check_notwav():
    with pytest.raises(ValueError):
        wav.check_format(__file__)


@pytest.mark.parametrize(
    'framerate, bitrate, nchannels',
    [(0, 0, 0), (16000, 8, 1), (16000, 16, 2)])
def test_check_bad(framerate, bitrate, nchannels):
    with pytest.raises(ValueError):
        wav.check_format(
            wav_file,
            framerate=framerate,
            bitrate=bitrate,
            nchannels=nchannels)


def test_check_good_default():
    assert wav.check_format(wav_file) is True


def test_check_good():
    assert wav.check_format(wav_file, framerate=16000, bitrate=16, nchannels=1)
