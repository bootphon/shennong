"""Test of the module shennong.processor.spectrogram"""

import numpy as np
import pytest

from shennong import Audio
from shennong.processor import SpectrogramProcessor


def test_bad_signal(audio):
    signal = Audio(np.random.random((10, 2)), 50)
    proc = SpectrogramProcessor(sample_rate=signal.sample_rate)
    with pytest.raises(ValueError) as err:
        proc.process(signal)
        assert 'signal must have one dimension' in str(err.value)

    with pytest.raises(ValueError) as err:
        proc = SpectrogramProcessor(sample_rate=signal.sample_rate + 1)
        proc.process(audio)
    assert 'mismatch in sample rates' in str(err.value)


def test_simple(audio):
    proc = SpectrogramProcessor(sample_rate=audio.sample_rate)
    feats = proc.process(audio)
    assert feats.shape == (140, 257)
    assert feats.shape[1] == proc.ndims
