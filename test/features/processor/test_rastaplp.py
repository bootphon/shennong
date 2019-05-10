"""Test of the shennong.features.processor.rastaplp module"""

import copy
import numpy as np
import os
import pytest

from shennong.audio import Audio
from shennong.features.processor.rastaplp import RastaPlpProcessor


def test_params():
    assert len(RastaPlpProcessor().get_params()) == 4

    p = {'frame_shift': 1, 'frame_length': 1, 'order': 1, 'do_rasta': False}
    f = RastaPlpProcessor(**p)
    assert f.get_params() == p
    assert f.do_rasta is False
    assert f.frame_length == f.frame_shift == f.order == 1

    with pytest.raises(ValueError) as err:
        f.order = -1
    assert 'must be an integer in [0, 12]' in str(err)

    with pytest.raises(ValueError) as err:
        f.order = 13
    assert 'must be an integer in [0, 12]' in str(err)


@pytest.mark.parametrize('order', [0, 1, 2, 5, 10, 12])
def test_order(order):
    signal = Audio(np.random.random((1000,)), 5000)
    proc = RastaPlpProcessor(order=order)
    feats = proc.process(signal)
    assert feats.shape[1] == proc.ndims
    if order != 0:
        assert feats.shape[1] == order + 1


def test_replicate(audio, data_path):
    audio2 = copy.deepcopy(audio)
    # make sure we replicate the implementation from rastapy (was
    # extracted from `audio` file using default parameters)
    original = np.load(os.path.join(data_path, 'test.rastaplp.npy')).T
    proc = RastaPlpProcessor(order=8, frame_shift=0.010, frame_length=0.025)
    feats = proc.process(audio).data

    assert feats.shape == original.shape
    assert np.allclose(feats, original)
    assert audio == audio2
