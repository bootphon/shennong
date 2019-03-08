"""Test of the module shennong.features.bottleneck"""

import os
import numpy as np
import pytest

from shennong.audio import AudioData
from shennong.utils import null_logger
from shennong.features.processor.bottleneck import (
    BottleneckProcessor, _compute_vad)


@pytest.mark.parametrize('weights', ['BabelMulti', 'FisherMono', 'FisherTri'])
def test_params(weights):
    p = {'weights': weights}
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
    assert 'invalid weights' in str(err)

    b = BottleneckProcessor()
    with pytest.raises(ValueError) as err:
        b.set_params(**{'weights': w})
    assert 'invalid weights' in str(err)


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
    assert list(proc._weights_data.keys()) == [
        'bn_std', 'input_mean', 'b2', 'b5',
        'input_std', 'W5', 'W7', 'W6', 'b6', 'b7', 'W3', 'W2',
        'context', 'b3', 'bn_mean', 'W1', 'b1']


@pytest.mark.parametrize('weights', ['BabelMulti', 'FisherMono', 'FisherTri'])
def test_process(audio, mfcc, weights):
    proc = BottleneckProcessor(weights=weights)
    feat = proc.process(audio)
    assert feat.shape == (140, 80)
    assert np.allclose(feat.times, mfcc.times)


# may fail to have approx arrays (because of random signal
# dithering), so we authorize 10 successive runs
@pytest.mark.flaky(reruns=10)
def test_compare_original(audio_8k, bottleneck_original):
    feat = BottleneckProcessor(weights='BabelMulti').process(audio_8k)
    assert bottleneck_original.shape == feat.shape
    assert bottleneck_original == pytest.approx(feat.data, abs=2e-2)


def test_silence():
    silence = AudioData(np.zeros((100,)), 16000)

    with pytest.raises(RuntimeError) as err:
        BottleneckProcessor().process(silence)
    assert 'no voice detected in signal' in str(err)

    # silence VAD all false
    vad = _compute_vad(silence.data, null_logger(), bugfix=True)
    assert not vad.any()
