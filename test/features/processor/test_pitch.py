"""Test of the module shennong.features.pitch"""

import numpy as np
import pytest

from shennong.audio import Audio
from shennong.features import Features
from shennong.features.processor.pitch import (
    PitchProcessor, PitchPostProcessor)


@pytest.fixture
def raw_pitch(audio):
    return PitchProcessor().process(audio)


def test_pitch_params():
    opts = {
        'sample_rate': 0,
        'frame_shift': 0,
        'frame_length': 0,
        'min_f0': 0,
        'max_f0': 0,
        'soft_min_f0': 0,
        'penalty_factor': 0,
        'lowpass_cutoff': 0,
        'resample_freq': 0,
        'delta_pitch': 0,
        'nccf_ballast': 0,
        'lowpass_filter_width': 0,
        'upsample_filter_width': 0}
    p = PitchProcessor(**opts)
    assert p.get_params() == opts

    p = PitchProcessor()
    p.set_params(**opts)
    assert p.get_params() == opts


def test_output(audio):
    ndims = PitchProcessor().ndims
    assert ndims == 2
    assert PitchProcessor(
        frame_shift=0.01).process(audio).shape == (140, ndims)
    assert PitchProcessor(
        frame_shift=0.02).process(audio).shape == (70, ndims)
    assert PitchProcessor(
        frame_shift=0.02, frame_length=0.05).process(audio).shape == (69, 2)

    # sample rate mismatch
    with pytest.raises(ValueError):
        PitchProcessor(sample_rate=8000).process(audio)

    # only mono signals are accepted
    with pytest.raises(ValueError):
        data = np.random.random((1000, 2))
        stereo = Audio(data, sample_rate=16000)
        PitchProcessor(sample_rate=stereo.sample_rate).process(stereo)


def test_post_pitch_params():
    opts = {
        'pitch_scale': 0,
        'pov_scale': 0,
        'pov_offset': 0,
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
    p = PitchPostProcessor(**opts)
    assert p.get_params() == opts

    p = PitchPostProcessor()
    p.set_params(**opts)
    assert p.get_params() == opts


def test_post_pitch(raw_pitch):
    post_processor = PitchPostProcessor()
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
    assert 'data shape must be (_, 2), but it is (_, 1)' in str(err)

    bad_pitch = Features(
        np.random.random((raw_pitch.nframes, 3)), raw_pitch.times)
    with pytest.raises(ValueError) as err:
        post_processor.process(bad_pitch)
    assert 'data shape must be (_, 2), but it is (_, 3)' in str(err)


@pytest.mark.parametrize('options', [
    (True, True, True, True),
    (True, True, True, False),
    (False, False, True, True),
    (False, False, False, False)])
def test_post_pitch_output(raw_pitch, options):
    p = PitchPostProcessor(
        add_pov_feature=options[0],
        add_normalized_log_pitch=options[1],
        add_delta_pitch=options[2],
        add_raw_log_pitch=options[3])

    if sum(options):
        d = p.process(raw_pitch)
        assert p.ndims == sum(options)
        assert d.shape == (raw_pitch.shape[0], sum(options))
        assert d.shape[1] == sum(options)
        assert np.array_equal(raw_pitch.times, d.times)
        assert d.times.shape[1] == 2
    else:  # all False not supported by Kaldi
        with pytest.raises(ValueError) as err:
            p.process(raw_pitch)
        assert 'must be True' in str(err)
