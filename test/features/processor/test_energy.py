"""Test of the module shennong.features.energy"""

import numpy as np
import pytest

from shennong.audio import AudioData
from shennong.features.processor.energy import EnergyProcessor


def test_params(audio):
    c = {'window_type': 'hanning', 'compression': 'sqrt', 'dither': 0}
    p1 = EnergyProcessor(**c)
    p2 = EnergyProcessor().set_params(**c)
    p3 = EnergyProcessor()
    p3.window_type = c['window_type']
    p3.compression = c['compression']
    p3.dither = c['dither']

    assert p1.get_params() == p2.get_params() == p3.get_params()
    assert p1.process(audio) == p2.process(audio) == p3.process(audio)
    assert p1.ndims == p2.ndims == p3.ndims == 1


@pytest.mark.parametrize('compression', ['log', 'sqrt', 'off', 'bad'])
def test_compression(audio, compression):
    p = EnergyProcessor()
    if compression == 'bad':
        with pytest.raises(ValueError) as err:
            p.compression = compression
        assert 'compression must be in ' in str(err)
    else:
        p.compression = compression
        assert p.process(audio).shape == (140, 1)


def test_output(audio):
    assert EnergyProcessor(frame_shift=0.01).process(audio).shape == (140, 1)
    assert EnergyProcessor(frame_shift=0.02).process(audio).shape == (70, 1)
    assert EnergyProcessor(
        frame_shift=0.02, frame_length=0.05).process(audio).shape == (69, 1)

    # sample rate mismatch
    with pytest.raises(ValueError) as err:
        EnergyProcessor(sample_rate=8000).process(audio)
    assert 'mismatch in sample rate' in str(err)

    # only mono signals are accepted
    with pytest.raises(ValueError) as err:
        data = np.random.random((1000, 2))
        stereo = AudioData(data, sample_rate=16000)
        EnergyProcessor(sample_rate=stereo.sample_rate).process(stereo)
    assert 'must have one dimension' in str(err)
