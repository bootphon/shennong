"""Test of the module shennong.features.postprocessor.vad"""

import numpy as np
import pytest

from shennong import Features
from shennong.processor.energy import EnergyProcessor
from shennong.postprocessor.vad import VadPostProcessor


def test_bad_params():
    p = VadPostProcessor()

    with pytest.raises(ValueError) as err:
        p.energy_mean_scale = -1
    assert 'must be >= 0' in str(err.value)

    with pytest.raises(ValueError) as err:
        p.frames_context = -1
    assert 'must be >= 0' in str(err.value)

    with pytest.raises(ValueError) as err:
        p.proportion_threshold = 0
    assert 'must be in ]0, 1[' in str(err.value)


def test_params():
    p = VadPostProcessor(
        energy_threshold=0,
        energy_mean_scale=0,
        frames_context=0,
        proportion_threshold=0.1)

    assert p.get_params() == pytest.approx({
        'energy_threshold': 0,
        'energy_mean_scale': 0,
        'frames_context': 0,
        'proportion_threshold': 0.1})

    assert p.ndims == 1


def test_mfcc(mfcc):
    p = VadPostProcessor()
    vad = p.process(mfcc)
    assert mfcc.shape[0] == vad.shape[0]
    assert vad.dtype is np.dtype(np.uint8)

    # may be true or false
    assert not np.all(vad.data)
    assert (len(vad.data[vad.data == 0]) + len(vad.data[vad.data == 1])
            == mfcc.shape[0])

    # always true
    p.energy_threshold = 0
    vad = p.process(mfcc)
    assert np.all(vad.data)

    # always false
    p.energy_threshold = 1e10
    vad = p.process(mfcc)
    assert not np.any(vad.data)

    # always 1
    assert p.ndims == 1
    assert vad.shape[1] == p.ndims


def test_energy(audio, mfcc):
    # VAD from mfcc or energy is the same excepted properties
    vad1 = VadPostProcessor().process(EnergyProcessor().process(audio))
    vad2 = VadPostProcessor().process(mfcc)
    vad1 = Features(vad1.data, vad1.times)
    vad2 = Features(vad2.data, vad2.times)
    assert vad1 == vad2
