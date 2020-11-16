"""Test of the module shennong.features.processor.energy"""

import numpy as np
import pytest

from shennong.audio import Audio
from shennong.features.processor.energy import EnergyProcessor
from shennong.features.processor.mfcc import MfccProcessor
from shennong.features.processor.plp import PlpProcessor


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
        assert 'compression must be in ' in str(err.value)
    else:
        p.compression = compression
        assert p.process(audio).shape == (140, 1)


@pytest.mark.parametrize('raw_energy', [True, False])
def test_raw(audio, raw_energy):
    p = {'raw_energy': raw_energy, 'dither': 0}
    mfcc = MfccProcessor(**p).process(audio).data[:, 0]
    plp = PlpProcessor(**p).process(audio).data[:, 0]
    energy = EnergyProcessor(**p).process(audio).data[:, 0]

    assert np.allclose(mfcc, energy)
    assert np.allclose(plp, energy)


def test_output_shape(audio):
    assert EnergyProcessor(frame_shift=0.01).process(audio).shape == (140, 1)
    assert EnergyProcessor(frame_shift=0.02).process(audio).shape == (70, 1)
    assert EnergyProcessor(
        frame_shift=0.02, frame_length=0.05).process(audio).shape == (69, 1)

    # sample rate mismatch
    with pytest.raises(ValueError) as err:
        EnergyProcessor(sample_rate=8000).process(audio)
    assert 'mismatch in sample rate' in str(err.value)

    # only mono signals are accepted
    with pytest.raises(ValueError) as err:
        data = np.random.random((1000, 2))
        stereo = Audio(data, sample_rate=16000)
        EnergyProcessor(sample_rate=stereo.sample_rate).process(stereo)
    assert 'must have one dimension' in str(err.value)
