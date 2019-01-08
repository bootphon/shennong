"""Test of the module shennong.features.plp"""

import numpy as np
import pytest

from shennong.audio import AudioData
from shennong.features.plp import PlpProcessor


def test_params():
    assert len(PlpProcessor().get_params()) == 24


@pytest.mark.parametrize('num_ceps', [-1, 0, 1, 5, 13, 23, 25])
def test_num_ceps(audio, num_ceps):
    proc = PlpProcessor(num_ceps=num_ceps)
    if 0 < proc.num_ceps:
        feat = proc.process(audio)
        assert proc.num_ceps == num_ceps
        assert feat.shape == (140, num_ceps)

        proc.use_energy = False
        feat = proc.process(audio)
        assert feat.shape == (140, num_ceps)
    else:
        with pytest.raises(RuntimeError):
            proc.process(audio)


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
        stereo = AudioData(data, sample_rate=16000)
        PlpProcessor(sample_rate=stereo.sample_rate).process(stereo)
