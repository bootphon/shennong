"""Test of the module shennong.audio"""

import tempfile
import numpy as np
import pytest

from kaldi.util.table import SequentialWaveReader
from shennong.audio import Audio


def test_scan(wav_file, audio):
    meta = Audio.scan(wav_file)
    assert meta.sample_rate == audio.sample_rate == 16000
    assert meta.nchannels == audio.nchannels == 1
    assert meta.nsamples == audio.nsamples == 22713
    assert meta.duration == audio.duration == pytest.approx(1.419, rel=1e-3)


def test_scan_bad():
    with pytest.raises(ValueError) as err:
        Audio.scan(__file__)
    assert 'SoXI failed' in str(err.value)

    with pytest.raises(ValueError) as err:
        Audio.scan('/path/to/some/lost/place')
    assert 'file not found' in str(err.value)


def test_load(audio):
    assert audio.sample_rate == 16000
    assert audio.nchannels == 1
    assert audio.duration == pytest.approx(1.419, rel=1e-3)
    assert audio.data.shape == (22713,)
    assert audio.nsamples == 22713
    assert audio.dtype == np.int16


def test_load_notwav():
    with pytest.raises(ValueError) as err:
        Audio.load(__file__)
    assert 'SoXI failed' in str(err.value)


def test_load_badfile():
    with pytest.raises(ValueError) as err:
        Audio.load('/spam/spam/with/eggs')
    assert 'file not found' in str(err.value)


def test_save(tmpdir, audio):
    p = str(tmpdir.join('test.wav'))
    audio.save(p)

    # cannot overwrite an existing file
    with pytest.raises(ValueError) as err:
        audio.save(p)
    assert 'file already exist' in str(err.value)

    audio2 = Audio.load(p)
    assert audio == audio2

    # test with float32 wav
    signal = np.zeros((1000,), dtype=np.float32)
    signal[10] = 1.0
    signal[20] = -1.0
    p = str(tmpdir.join('test2.wav'))
    audio = Audio(signal, 1000)
    audio.save(p)
    meta = Audio.scan(p)
    assert meta.nchannels == 1
    assert meta.nsamples == 1000

    audio2 = Audio.load(p)
    assert audio2.nchannels == 1
    assert audio2.nsamples == 1000
    assert audio2 == audio
    assert audio2.data.min() == -1.0
    assert audio2.data.max() == 1.0


def test_save_bad(tmpdir, audio):
    p = str(tmpdir.join('test.notavalidextension'))
    with pytest.raises(ValueError) as err:
        audio.save(p)
    assert 'failed to write' in str(err.value)


def test_equal(audio):
    assert audio == audio

    audio2 = Audio(audio.data, audio.sample_rate)
    assert audio == audio2

    audio2 = Audio(audio.data, audio.sample_rate + 1)
    assert audio != audio2

    audio2 = Audio(audio.data * 2, audio.sample_rate)
    assert audio.duration == audio2.duration
    assert audio.sample_rate == audio2.sample_rate
    assert audio != audio2


def test_shape():
    # it was a bug when audio data is shaped (n, 1): must be reshaped
    # as (n,). The bug appens when converting audio data to pykaldi
    # vector.
    d1 = np.random.random((100,))
    assert d1.shape == (100,)

    d2 = np.random.random((100, 1))
    assert d2.shape == (100, 1)

    for d in (d1, d2):
        a = Audio(d, 10)
        assert a.shape == (100,)


def test_channels_mono(audio):
    assert audio.nchannels == 1
    assert audio.shape == (audio.nsamples,)
    assert audio.channel(0) == audio
    with pytest.raises(ValueError):
        audio.channel(1)


def test_channels_stereo():
    data = np.random.random((1000, 2))
    audio2 = Audio(data, sample_rate=16000)
    assert audio2.nchannels == 2
    assert audio2.shape == (1000, 2)

    audio1 = audio2.channel(0)
    assert audio1.nchannels == 1
    assert audio1.shape == (1000,)
    assert all(np.equal(audio1.data, audio2.data[:, 0]))
    assert not all(np.equal(audio1.data, audio2.data[:, 1]))
    assert audio1.duration == audio2.duration

    audio1 = audio2.channel(1)
    assert audio1.nchannels == 1
    assert audio1.shape == (1000,)
    assert all(np.equal(audio1.data, audio2.data[:, 1]))
    assert not all(np.equal(audio1.data, audio2.data[:, 0]))

    with pytest.raises(ValueError):
        audio2.channel(2)


def test_isvalid(audio):
    assert audio.dtype is np.dtype(np.int16)
    assert audio.is_valid()

    # brutal cast from int16 to float32, still with values greater than 1
    audio2 = Audio(
        audio.data.astype(np.float32), audio.sample_rate, validate=False)
    assert audio2.dtype is np.dtype(np.float32)
    assert not audio2.is_valid()
    with pytest.raises(ValueError) as err:
        Audio(audio.data.astype(np.float32),
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
        Audio(data, audio.sample_rate)
        assert 'invalid audio data for type' in err

    audio4 = Audio(data, audio.sample_rate, validate=False)
    assert not audio4.is_valid()

    # brutal cast to invalid uint8 dtype
    audio5 = Audio(
        audio.data.astype(np.uint8), audio.sample_rate, validate=False)
    assert audio5.dtype is np.dtype(np.uint8)
    assert not audio5.is_valid()


DTYPES = [np.int16, np.int32, np.float32, np.float64, float]


@pytest.mark.parametrize('dtype', DTYPES)
def test_astype(audio_tiny, dtype):
    audio = audio_tiny
    assert audio.dtype is np.dtype(np.int16)

    # from int16 to dtype
    audio2 = audio.astype(dtype)
    assert audio2.dtype is np.dtype(dtype)
    assert audio.dtype is np.dtype(np.int16)
    assert audio2.is_valid()

    # back to int16
    audio3 = audio2.astype(np.int16)
    assert audio3.data == pytest.approx(audio.data)
    assert audio3.dtype is np.dtype(np.int16)
    assert audio.dtype is np.dtype(np.int16)

    # from dtype to all other types that are not int16
    for dtype2 in set(DTYPES) - set([np.int16, dtype]):
        print(dtype2)
        audio4 = audio2.astype(dtype2)
        assert audio4.is_valid()
        assert audio4.dtype is np.dtype(dtype2)
        assert audio4.astype(np.int16).data == pytest.approx(audio.data)


@pytest.mark.parametrize(
    'dtype', [np.uint8, np.int64, np.float128, str, int])
def test_asbadtype(audio, dtype):
    with pytest.raises(ValueError) as err:
        audio.astype(dtype)
        assert 'unsuported audio data type' in err


@pytest.mark.parametrize(
    'fs, backend', [
        (f, b) for f in [4000, 8000, 16000, 32000, 44100, 48000]
        for b in ('sox', 'scipy')])
def test_resample(audio, fs, backend):
    audio2 = audio.resample(fs, backend=backend)
    assert audio2.nchannels == audio.nchannels
    assert audio2.sample_rate == fs
    assert audio2.nsamples == pytest.approx(int(
        audio.nsamples * fs / audio.sample_rate), abs=1)
    assert audio2.data.mean() == pytest.approx(audio.data.mean(), abs=0.25)
    assert audio2.dtype == audio.dtype

    # back to original sample rate
    if fs >= audio.sample_rate:
        audio3 = audio2.resample(audio.sample_rate, backend=backend)
        assert audio3.nchannels == audio.nchannels
        assert audio3.sample_rate == audio.sample_rate
        assert audio3.dtype == audio.dtype
        assert audio2.data.mean() == pytest.approx(audio.data.mean(), abs=0.25)


def test_resample_bad(audio):
    with pytest.raises(ValueError) as err:
        audio.resample(5, backend='a_bad_one')
    assert 'backend must be sox or scipy, it is' in str(err.value)

    with pytest.raises(ValueError) as err:
        audio.resample(0)
    assert 'resampling at 0 failed' in str(err.value)


def test_compare_kaldi(wav_file):
    a1 = Audio.load(wav_file).data

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


def test_segment(audio):
    d = audio.duration
    assert audio.segment([(0., d)])[0] == audio
    assert audio.segment([(0., d+10)])[0] == audio

    chunks = audio.segment([(0, d/2), (d/2, d)])
    assert all(c.duration == pytest.approx(d/2, rel=1e-3) for c in chunks)
    assert sum(c.nsamples for c in chunks) == audio.nsamples
    assert Audio(
        np.concatenate([c.data for c in chunks]), audio.sample_rate) == audio

    chunks = audio.segment([(0, d/3), (d/3, 2*d/3), (2*d/3, d)])
    assert all(c.duration == pytest.approx(d/3, rel=1e-3) for c in chunks)
    assert sum(c.nsamples for c in chunks) == audio.nsamples
    assert Audio(
        np.concatenate([c.data for c in chunks]), audio.sample_rate) == audio


def test_segment_bad(audio):
    with pytest.raises(ValueError) as err:
        audio.segment(0)
    assert 'segments must be a list' in str(err.value)

    with pytest.raises(ValueError) as err:
        audio.segment([0, 1])
    assert 'must be pairs' in str(err.value)
    with pytest.raises(ValueError) as err:
        audio.segment([(0, 1, 2)])
    assert 'must be pairs' in str(err.value)

    with pytest.raises(ValueError) as err:
        audio.segment([(1, 0)])
    assert 'must be sorted' in str(err.value)
