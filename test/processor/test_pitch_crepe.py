"""Test of the module shennong.features.processor.crepepitch"""

import numpy as np
import pytest
from shennong import Audio, Features
from shennong.processor.pitch_crepe import (
    CrepePitchPostProcessor, CrepePitchProcessor)


@pytest.fixture
def raw_pitch(audio):
    return CrepePitchProcessor(model_capacity='tiny').process(audio)


def test_crepe_pitch_params():
    opts = {
        'frame_shift': 0,
        'frame_length': 0,
        'model_capacity': 'full',
        'center': True,
        'viterbi': True
    }
    p = CrepePitchProcessor(**opts)
    assert p.get_params() == opts

    p = CrepePitchProcessor()
    p.set_params(**opts)
    assert p.get_params() == opts

    assert p.sample_rate == 16000

    with pytest.raises(ValueError) as err:
        p = CrepePitchProcessor(model_capacity='wrong')
    assert 'Model capacity wrong is not recognized' in str(err.value)


@pytest.mark.parametrize(
    'viterbi, center',
    [(v, c) for v in (True, False) for c in (True, False)])
def test_output(audio, viterbi, center):
    pitch = CrepePitchProcessor(
        model_capacity='tiny', viterbi=viterbi, center=center).process(audio)
    assert pitch.shape == (140, 2)

    if center:
        assert pitch.data[:, 0].mean() == pytest.approx(0.440450713829631)
    else:
        assert pitch.data[:, 0].mean() == pytest.approx(0.4569764207391177)

    if viterbi:
        assert np.all(pitch.data[:, 1] > 0)
    else:
        assert not np.all(pitch.data[:, 1] > 0)

    if viterbi and center:
        assert pitch.data[:, 1].mean() == pytest.approx(121.04003190158486)
    elif viterbi and not center:
        assert pitch.data[:, 1].mean() == pytest.approx(122.78609105951135)
    elif not viterbi and center:
        assert pitch.data[:, 1].mean() == pytest.approx(282.34977980138643)
    else:
        assert pitch.data[:, 1].mean() == pytest.approx(265.5468749764539)


def test_output_frames(audio, audio_8k):
    ndims = CrepePitchProcessor().ndims
    assert ndims == 2
    assert CrepePitchProcessor(
        model_capacity='tiny',
        frame_shift=0.01).process(audio).shape == (140, ndims)
    assert CrepePitchProcessor(
        model_capacity='tiny',
        frame_shift=0.02).process(audio).shape == (70, ndims)
    assert CrepePitchProcessor(
        model_capacity='tiny',
        frame_shift=0.02, frame_length=0.05).process(audio).shape == (69, 2)

    # resample audio when processing
    assert CrepePitchProcessor(
        model_capacity='tiny',
        frame_shift=0.01).process(audio_8k).shape == (140, ndims)

    # only mono signals are accepted
    with pytest.raises(ValueError) as err:
        data = np.random.random((1000, 2))
        stereo = Audio(data, sample_rate=16000)
        CrepePitchProcessor(model_capacity='tiny').process(stereo)
    assert 'audio must have one channel' in str(err.value)


def test_post_crepe_pitch_params():
    opts = {
        'pitch_scale': 0,
        'delta_pitch_scale': 0,
        'delta_pitch_noise_stddev': 0,
        'normalization_left_context': 0,
        'normalization_right_context': 0,
        'delta_window': 0,
        'delay': 0,
        'add_pov_feature': bool(10),  # implicit cast
        'add_normalized_log_pitch': False,
        'add_delta_pitch': False,
        'add_raw_log_pitch': False}
    p = CrepePitchPostProcessor(**opts)
    assert p.get_params() == opts

    p = CrepePitchPostProcessor()
    p.set_params(**opts)
    assert p.get_params() == opts


def test_post_crepe_pitch(raw_pitch):
    post_processor = CrepePitchPostProcessor()
    params = post_processor.get_params()
    data = post_processor.process(raw_pitch)
    assert data.shape[1] == 3
    assert raw_pitch.shape[0] == data.shape[0]
    assert np.array_equal(raw_pitch.times, data.times)
    assert params == post_processor.get_params()

    bad_pitch = Features(
        np.random.random((raw_pitch.nframes, 1)), raw_pitch.times)
    with pytest.raises(ValueError) as err:
        post_processor.process(bad_pitch)
    assert 'data shape must be (_, 2), but it is (_, 1)' in str(err.value)

    bad_pitch = Features(
        np.random.random((raw_pitch.nframes, 3)), raw_pitch.times)
    with pytest.raises(ValueError) as err:
        post_processor.process(bad_pitch)
    assert 'data shape must be (_, 2), but it is (_, 3)' in str(err.value)

    bad_pitch = Features(
        np.zeros((raw_pitch.nframes, 2)), raw_pitch.times)
    with pytest.raises(ValueError) as err:
        post_processor.process(bad_pitch)
    assert 'No voiced frames' in str(err.value)

    bad_pitch = Features(np.ones((raw_pitch.nframes, 2)), raw_pitch.times)
    bad_pitch.data[:, 1] = - np.random.random((raw_pitch.nframes))
    with pytest.raises(ValueError) as err:
        post_processor.process(bad_pitch)
    assert 'Not all pitch values are positive' in str(err.value)


@pytest.mark.parametrize('options', [
    (True, True, True, True),
    (True, True, True, False),
    (False, False, True, True),
    (False, False, False, False)])
def test_post_pitch_output(raw_pitch, options):
    p = CrepePitchPostProcessor(
        add_pov_feature=options[0],
        add_normalized_log_pitch=options[1],
        add_delta_pitch=options[2],
        add_raw_log_pitch=options[3])

    raw_pitch.data[1, 0] = 0

    if sum(options):
        d = p.process(raw_pitch)
        assert p.ndims == sum(options)
        assert d.shape == (raw_pitch.shape[0], sum(options))
        assert d.shape[1] == sum(options)
        assert np.array_equal(raw_pitch.times, d.times)
    else:  # all False not supported
        with pytest.raises(ValueError) as err:
            p.process(raw_pitch)
        assert 'must be True' in str(err.value)
