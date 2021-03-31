"""Test of the shennong.processor.rastaplp module"""

import numpy as np
import pytest

from shennong import Audio
from shennong.processor import RastaPlpProcessor


def test_params():
    assert len(RastaPlpProcessor().get_params()) == 12

    p = {'frame_shift': 1, 'frame_length': 1, 'order': 1, 'do_rasta': False}
    f = RastaPlpProcessor(**p)
    for k, v in p.items():
        assert f.get_params()[k] == v
    assert f.do_rasta is False
    assert f.frame_length == f.frame_shift == f.order == 1

    with pytest.raises(ValueError) as err:
        f.order = -1
    assert 'must be an integer in [0, 12]' in str(err.value)

    with pytest.raises(ValueError) as err:
        f.order = 13
    assert 'must be an integer in [0, 12]' in str(err.value)


def test_bad_signal():
    signal = Audio(np.random.random((10, 2)), 50)
    proc = RastaPlpProcessor(sample_rate=signal.sample_rate)
    with pytest.raises(ValueError) as err:
        proc.process(signal)
        assert 'signal must have one dimension' in str(err.value)


@pytest.mark.parametrize('order', [0, 1, 2, 5, 10, 12])
def test_order(order, audio):
    with pytest.raises(ValueError) as err:
        proc = RastaPlpProcessor(
            order=order, sample_rate=audio.sample_rate + 1)
        proc.process(audio)
    assert 'mismatch in sample rates' in str(err.value)

    proc = RastaPlpProcessor(
        order=order, sample_rate=audio.sample_rate, dither=0)

    feats1 = proc.process(audio)
    assert feats1.shape[1] == proc.ndims
    if order != 0:
        assert feats1.shape[1] == order + 1

    proc = RastaPlpProcessor()
    proc.order = order
    proc.sample_rate = audio.sample_rate
    proc.dither = 0
    feats2 = proc.process(audio)
    assert feats1.is_close(feats2)
    assert feats1.dtype == np.float32
    assert np.all(np.isfinite(feats1.data))
