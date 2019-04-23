"""Test of the module shennong.features.plp"""

import numpy as np
import pytest

from shennong.audio import Audio
from shennong.features.processor.plp import PlpProcessor


def test_params():
    assert len(PlpProcessor().get_params()) == 24

    params = {
        'num_bins': 0,
        'use_energy': True,
        'energy_floor': 10.0,
        'raw_energy': False,
        'htk_compat': True,
        'htk_compat': True}
    p = PlpProcessor(**params)
    out_params = p.get_params()
    assert len(out_params) == 24

    assert PlpProcessor().set_params(**params).get_params() == out_params


@pytest.mark.parametrize('num_ceps', [-1, 0, 1, 5, 13, 23, 25])
def test_num_ceps(audio, num_ceps):
    if num_ceps >= 23:
        with pytest.raises(ValueError) as err:
            PlpProcessor(num_ceps=num_ceps)
        assert 'We must have num_ceps <= lpc_order+1' in str(err)
    else:
        proc = PlpProcessor(num_ceps=num_ceps)
        if 0 < proc.num_ceps:
            feat = proc.process(audio)
            assert proc.num_ceps == num_ceps == proc.ndims
            assert feat.shape == (140, num_ceps)

            proc.use_energy = False
            feat = proc.process(audio)
            assert feat.shape == (140, num_ceps)
        else:
            with pytest.raises(RuntimeError):
                proc.process(audio)


@pytest.mark.flaky(reruns=20)
def test_htk_compat(audio):
    p1 = PlpProcessor(use_energy=True, htk_compat=False).process(audio)
    p2 = PlpProcessor(use_energy=True, htk_compat=True).process(audio)
    assert p1.data[:, 0] == pytest.approx(p2.data[:, -1], rel=1e-2)

    p1 = PlpProcessor(use_energy=False, htk_compat=False).process(audio)
    p2 = PlpProcessor(use_energy=False, htk_compat=True).process(audio)
    assert p1.data[:, 0] == pytest.approx(p2.data[:, -1], rel=1e-1)


def test_output(audio):
    assert PlpProcessor(frame_shift=0.01).process(audio).shape == (140, 13)
    assert PlpProcessor(frame_shift=0.02).process(audio).shape == (70, 13)
    assert PlpProcessor(
        frame_shift=0.02, frame_length=0.05).process(audio).shape == (69, 13)

    # sample rate mismatch
    with pytest.raises(ValueError):
        PlpProcessor(sample_rate=8000).process(audio)

    # only mono signals are accepted
    with pytest.raises(ValueError):
        data = np.random.random((1000, 2))
        stereo = Audio(data, sample_rate=16000)
        PlpProcessor(sample_rate=stereo.sample_rate).process(stereo)
