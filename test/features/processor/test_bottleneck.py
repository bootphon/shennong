"""Test of the module shennong.features.bottleneck"""

import os
import numpy as np
import pytest

from shennong.audio import Audio
from shennong.utils import null_logger, get_logger
from shennong.features.processor.bottleneck import (
    BottleneckProcessor, _compute_vad)


@pytest.mark.parametrize('weights', ['BabelMulti', 'FisherMono', 'FisherTri'])
def test_params(weights):
    p = {'weights': weights, 'dither': 0.1}
    assert BottleneckProcessor(**p).get_params() == p

    b = BottleneckProcessor()
    assert b.weights == 'BabelMulti'
    b.set_params(**p)
    assert BottleneckProcessor(**p).get_params() == p
    assert b.weights == weights


def test_bad_params():
    w = 'BadWeights'
    with pytest.raises(ValueError) as err:
        BottleneckProcessor(w)
    assert 'invalid weights' in str(err.value)

    b = BottleneckProcessor()
    with pytest.raises(ValueError) as err:
        b.set_params(**{'weights': w})
    assert 'invalid weights' in str(err.value)


def test_available_weights():
    weights = BottleneckProcessor.available_weights()
    assert len(weights) == 3
    for w in ('BabelMulti', 'FisherMono', 'FisherTri'):
        assert w in weights
        assert os.path.isfile(weights[w])


@pytest.mark.parametrize('weights', ['BabelMulti', 'FisherMono', 'FisherTri'])
def test_weights(weights):
    # make sure all the pretrained weights are here, and contains the
    # required entries
    proc = BottleneckProcessor(weights=weights)
    assert proc.weights == weights
    w = proc._get_weights()
    assert list(w.keys()) == [
        'bn_std', 'input_mean', 'b2', 'b5',
        'input_std', 'W5', 'W7', 'W6', 'b6', 'b7', 'W3', 'W2',
        'context', 'b3', 'bn_mean', 'W1', 'b1']


@pytest.mark.parametrize('weights', ['BabelMulti', 'FisherMono', 'FisherTri'])
def test_process(capsys, audio, mfcc, weights):
    get_logger(level='debug')

    proc = BottleneckProcessor(weights=weights)
    feat = proc.process(audio)
    assert feat.shape == (140, 80)
    assert feat.shape[1] == proc.ndims
    assert np.allclose(feat.times, mfcc.times)
    assert proc.frame_length == 0.025
    assert proc.frame_shift == 0.01
    assert proc.sample_rate == 8000

    # check the log messages
    captured = capsys.readouterr().err
    assert 'resampling audio from 16000Hz@16b to 8000Hz@16b' in captured
    assert '{} frames of speech detected (on 140 total frames)'.format(
        '118' if audio._sox_found() else '121') in captured


def test_compare_original(audio_8k, bottleneck_original):
    feat = BottleneckProcessor(
        weights='BabelMulti', dither=0).process(audio_8k)
    assert bottleneck_original.shape == feat.shape
    assert bottleneck_original == pytest.approx(feat.data, abs=2e-2)


def test_silence():
    silence = Audio(np.zeros((100,)), 16000)

    with pytest.raises(RuntimeError) as err:
        BottleneckProcessor().process(silence)
    assert 'no voice detected in signal' in str(err.value)

    # silence VAD all false
    vad = _compute_vad(silence.data, null_logger(), bugfix=True)
    assert not vad.any()
