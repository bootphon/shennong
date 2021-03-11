"""Test of the module shennong.processor.filterbank"""

import numpy as np
import pytest

from shennong import Audio
from shennong.processor.filterbank import FilterbankProcessor


def test_params():
    params = {
        'num_bins': 0,
        'use_energy': True,
        'energy_floor': 10.0,
        'raw_energy': False,
        'htk_compat': True,
        'use_log_fbank': False,
        'use_power': False}
    p = FilterbankProcessor(**params)

    params_out = p.get_params()
    assert len(params_out) == 21
    for k, v in params.items():
        assert params_out[k] == v

    p = FilterbankProcessor()
    p.set_params(**params_out)

    params_out = p.get_params()
    for k, v in params.items():
        assert params_out[k] == v

    p = FilterbankProcessor()
    p.set_params(**params_out)
    assert p.get_params() == params_out


@pytest.mark.parametrize(
    'use_energy, num_bins',
    [(e, b) for e in (True, False) for b in (0, 1, 10, 23, 30)])
def test_num_bins(audio, use_energy, num_bins):
    ncols = num_bins + use_energy
    p = FilterbankProcessor(use_energy=use_energy, num_bins=num_bins)

    assert p.ndims == ncols
    if num_bins >= 3:
        assert p.process(audio).shape == (140, ncols)
    else:
        with pytest.raises(RuntimeError):
            p.process(audio)


def test_energy(audio):
    p1 = FilterbankProcessor(use_energy=False).process(audio)
    p2 = FilterbankProcessor(use_energy=True).process(audio)

    assert p1.shape[1] == p2.shape[1] - 1
    assert p1.data == pytest.approx(p2.data[:, 1:], rel=1e-1)


def test_output(audio):
    assert FilterbankProcessor(
        frame_shift=0.01).process(audio).shape == (140, 23)
    assert FilterbankProcessor(
        frame_shift=0.02).process(audio).shape == (70, 23)
    assert FilterbankProcessor(
        frame_shift=0.02, frame_length=0.05).process(audio).shape == (69, 23)

    # sample rate mismatch
    with pytest.raises(ValueError):
        FilterbankProcessor(sample_rate=8000).process(audio)

    # only mono signals are accepted
    with pytest.raises(ValueError):
        data = np.random.random((1000, 2))
        stereo = Audio(data, sample_rate=16000)
        FilterbankProcessor(sample_rate=stereo.sample_rate).process(stereo)
