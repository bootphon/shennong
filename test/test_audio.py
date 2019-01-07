"""Test of the module shennong.audio"""

import tempfile
import numpy as np
import pytest

from kaldi.util.table import SequentialWaveReader
from shennong.audio import AudioData


def test_load(audio):
    assert audio.sample_rate == 16000
    assert audio.nchannels == 1
    assert audio.duration == pytest.approx(1.419, rel=1e-3)
    assert audio.data.shape == (22713,)
    assert audio.nsamples == 22713
    assert audio.dtype == np.int16


def test_load_notwav():
    with pytest.raises(ValueError):
        AudioData.load(__file__)


def test_equal(audio):
    assert audio == audio

    audio2 = AudioData(audio.data, audio.sample_rate)
    assert audio == audio2

    audio2 = AudioData(audio.data, audio.sample_rate + 1)
    assert audio != audio2

    audio2 = AudioData(audio.data * 2, audio.sample_rate)
    assert audio.duration == audio2.duration
    assert audio.sample_rate == audio2.sample_rate
    assert audio != audio2


def test_channels_mono(audio):
    assert audio.nchannels == 1
    assert audio.channel(0) == audio
    with pytest.raises(ValueError):
        audio.channel(1)


def test_channels_stereo():
    data = np.random.random((1000, 2))
    audio2 = AudioData(data, sample_rate=16000)
    assert audio2.nchannels == 2

    audio1 = audio2.channel(0)
    assert audio1.nchannels == 1
    assert all(np.equal(audio1.data, audio2.data[:, 0]))
    assert not all(np.equal(audio1.data, audio2.data[:, 1]))
    assert audio1.duration == audio2.duration

    audio1 = audio2.channel(1)
    assert audio1.nchannels == 1
    assert all(np.equal(audio1.data, audio2.data[:, 1]))
    assert not all(np.equal(audio1.data, audio2.data[:, 0]))

    with pytest.raises(ValueError):
        audio2.channel(2)


def test_isvalid(audio):
    assert audio.dtype is np.dtype(np.int16)
    assert audio.is_valid()

    # brutal cast from int16 to float32, still with values greater than 1
    audio2 = AudioData(
        audio.data.astype(np.float32), audio.sample_rate, validate=False)
    assert audio2.dtype is np.dtype(np.float32)
    assert not audio2.is_valid()
    with pytest.raises(ValueError) as err:
        AudioData(audio.data.astype(np.float32),
                  audio.sample_rate, validate=True)
        'invalid audio data' in err

    # smooth cast from int16 to float32
    audio3 = audio.astype(np.float32)
    assert audio3.dtype is np.dtype(np.float32)
    assert audio3.is_valid()

    # just add a silly value in float32 audio
    data = audio3.data.copy()
    data[6] = 1.1
    with pytest.raises(ValueError) as err:
        AudioData(data, audio.sample_rate)
        assert 'invalid audio data for type' in err

    audio4 = AudioData(data, audio.sample_rate, validate=False)
    assert not audio4.is_valid()


@pytest.mark.parametrize(
    'dtype', [np.int16, np.int32, np.float32, np.float64, float])
def test_astype(audio, dtype):
    assert audio.dtype is np.dtype(np.int16)

    # from int16 to dtype
    audio2 = audio.astype(dtype)
    assert audio2.data.min() < 0 and audio2.data.max() > 0
    assert audio2.dtype is np.dtype(dtype)
    assert audio.dtype is np.dtype(np.int16)

    # back to int16
    audio3 = audio2.astype(np.int16)
    assert audio3.data == pytest.approx(audio.data)
    assert audio3.dtype is np.dtype(np.int16)
    assert audio.dtype is np.dtype(np.int16)


@pytest.mark.parametrize(
    'dtype', [np.uint8, np.int64, np.float128, str, int])
def test_asbadtype(audio, dtype):
    with pytest.raises(ValueError) as err:
        audio.astype(dtype)
        assert 'unsuported audio data type' in err


@pytest.mark.parametrize(
    'fs', [4000, 8000, 16000, 32000, 44100, 48000])
def test_resample(audio, fs):
    audio2 = audio.resample(fs)
    assert audio2.nchannels == audio.nchannels
    assert audio2.sample_rate == fs
    assert audio2.nsamples == int(
        audio.nsamples * fs / audio.sample_rate)
    assert audio2.data.mean() == pytest.approx(audio.data.mean(), abs=0.25)
    assert audio2.dtype == audio.dtype

    # back to original sample rate
    if fs >= audio.sample_rate:
        audio3 = audio2.resample(audio.sample_rate)
        assert audio3.nchannels == audio.nchannels
        assert audio3.sample_rate == audio.sample_rate
        assert audio3.dtype == audio.dtype
        if audio3.nsamples == audio.nsamples:
            assert audio3.data == pytest.approx(audio.data, abs=2)
        else:
            assert audio2.data.mean() == pytest.approx(
                audio.data.mean(), abs=0.25)


def test_compare_kaldi(wav_file):
    a1 = AudioData.load(wav_file).data

    with tempfile.NamedTemporaryFile('w+') as tfile:
        tfile.write('test {}\n'.format(wav_file))
        tfile.seek(0)
        with SequentialWaveReader('scp,t:' + tfile.name) as reader:
            for key, wave in reader:
                a2 = wave.data().numpy()

    assert a1.max() == a2.max()
    assert a1.min() == a2.min()
    assert len(a1) == len(a2.flatten()) == 22713
    assert a1.dtype == np.int16 and a2.dtype == np.float32
    assert a1.shape == (22713,) and a2.shape == (1, 22713)
    assert pytest.approx(a1, a2)
