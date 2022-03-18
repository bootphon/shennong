"""Test of the module shennong.processor.mfcc"""

import tempfile
import numpy as np
import pytest

from kaldi.util.table import SequentialWaveReader
from shennong import Audio
from shennong.processor import MfccProcessor


def test_params():
    assert len(MfccProcessor().get_params()) == 21

    p = {'htk_compat': True, 'num_bins': 20, 'energy_floor': 1.0, 'dither': 2}
    f = MfccProcessor(**p)

    params_out = f.get_params()
    assert len(params_out) == 21
    for k, v in p.items():
        assert params_out[k] == v
    assert f.get_params() == params_out

    f = MfccProcessor()
    f.set_params(**params_out)
    params_out = f.get_params()
    assert len(params_out) == 21
    for k, v in p.items():
        assert params_out[k] == v
    assert f.get_params() == params_out
    assert f.ndims == f.num_ceps


def test_set_params():
    m = MfccProcessor()

    assert m.get_params()['sample_rate'] == 16000
    m.set_params(sample_rate=0)
    assert m.get_params()['sample_rate'] == 0

    assert m.get_params()['window_type'] == 'povey'
    m.set_params(window_type='hanning')
    assert m.get_params()['window_type'] == 'hanning'
    with pytest.raises(ValueError):
        m.set_params(window_type='foo')


def test_dither(audio):
    p1 = MfccProcessor()
    p1.dither = 0
    f1 = p1.process(audio)

    p2 = MfccProcessor(dither=0)
    f2 = p2.process(audio)

    p3 = MfccProcessor()
    p3.set_params(**{'dither': 0})
    f3 = p3.process(audio)

    assert f1 == f2 == f3


def test_from_badshape(audio):
    p = MfccProcessor()
    audio = Audio(audio.data.reshape((audio.nsamples, 1)), audio.sample_rate)
    assert p.process(audio).shape == (140, 13)


@pytest.mark.parametrize('num_ceps', [0, 1, 5, 13, 23, 25])
def test_num_ceps(audio, num_ceps):
    proc = MfccProcessor(num_ceps=num_ceps)
    if 0 < proc.num_ceps <= proc.num_bins:
        feat = proc.process(audio)
        assert feat.shape == (140, num_ceps)

        proc.use_energy = False
        feat = proc.process(audio)
        assert feat.shape == (140, num_ceps)
    else:
        with pytest.raises(RuntimeError):
            proc.process(audio)


@pytest.mark.parametrize('num_bins', [0, 1, 5, 23])
def test_num_bins(audio, num_bins):
    proc = MfccProcessor(num_bins=num_bins)
    proc.num_ceps = min(proc.num_ceps, num_bins)
    if 3 <= proc.num_bins:
        feat = proc.process(audio)
        assert feat.shape == (140, proc.num_ceps)

        proc.use_energy = False
        feat = proc.process(audio)
        assert feat.shape == (140, proc.num_ceps)
    else:
        with pytest.raises(RuntimeError):
            proc.process(audio)


def test_htk_compat(audio):
    p1 = MfccProcessor(
        use_energy=True, htk_compat=False, dither=0).process(audio)
    p2 = MfccProcessor(
        use_energy=True, htk_compat=True, dither=0).process(audio)
    assert p1.data[:, 0] == pytest.approx(p2.data[:, -1])

    p1 = MfccProcessor(
        use_energy=False, htk_compat=False, dither=0).process(audio)
    p2 = MfccProcessor(
        use_energy=False, htk_compat=True, dither=0).process(audio)
    assert p1.data[:, 0] * 2**0.5 == pytest.approx(p2.data[:, -1])


def test_output(audio):
    assert MfccProcessor(frame_shift=0.01).process(audio).shape == (140, 13)
    assert MfccProcessor(frame_shift=0.02).process(audio).shape == (70, 13)
    assert MfccProcessor(
        frame_shift=0.02, frame_length=0.05).process(audio).shape == (69, 13)

    # sample rate mismatch
    with pytest.raises(ValueError):
        MfccProcessor(sample_rate=8000).process(audio)

    # only mono signals are accepted
    with pytest.raises(ValueError):
        data = np.random.random((1000, 2))
        stereo = Audio(data, sample_rate=16000)
        MfccProcessor(sample_rate=stereo.sample_rate).process(stereo)


@pytest.mark.parametrize('sample_rate', [8000, 44100])
def test_subover_sample(audio, sample_rate):
    audio_resamp = audio.resample(sample_rate)

    proc1 = MfccProcessor(sample_rate=sample_rate)
    feat = proc1.process(audio_resamp)
    assert feat.shape == (140, 13)

    proc2 = MfccProcessor()
    with pytest.raises(ValueError) as err:
        proc2.process(audio_resamp)
        assert 'mismatch in sample rate' in err


@pytest.mark.parametrize('dtype', [np.int16, np.int32, np.float32, np.float64])
def test_kaldi_audio(wav_file, audio, dtype):
    # make sure we have results when loading a wav file with
    # shennong.Audio and with the Kaldi code.
    with tempfile.NamedTemporaryFile('w+') as tfile:
        tfile.write('test {}\n'.format(wav_file))
        tfile.seek(0)
        with SequentialWaveReader('scp,t:' + tfile.name) as reader:
            for key, wave in reader:
                audio_kaldi = Audio(
                    wave.data().numpy().reshape(audio.data.shape) / 2**15,
                    audio.sample_rate, validate=True)

    assert audio_kaldi.dtype == np.float32
    assert audio_kaldi.is_valid()

    audio = audio.astype(dtype)
    assert audio.duration == audio_kaldi.duration
    assert audio.dtype == dtype
    assert audio.is_valid()

    # no dither to compare the 2 resulting arrays
    mfcc = MfccProcessor(dither=0).process(audio)
    mfcc_kaldi = MfccProcessor(dither=0).process(audio_kaldi)
    assert mfcc.shape == mfcc_kaldi.shape
    assert np.array_equal(mfcc.times, mfcc_kaldi.times)
    assert mfcc.properties == mfcc_kaldi.properties
    assert mfcc.dtype == mfcc_kaldi.dtype
    assert mfcc.data == pytest.approx(mfcc_kaldi.data)
