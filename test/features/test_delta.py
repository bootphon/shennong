"""Test of the module shennong.features.delta"""

import numpy as np
import pytest

from shennong.features.delta import DeltaProcessor


def test_params():
    d = DeltaProcessor()
    d.order = 0
    with pytest.raises(ValueError):
        d.window = 0
    with pytest.raises(ValueError):
        d.window = 2000
    d.window = 1

    assert d.get_params() == {'order': 0, 'window': 1}


@pytest.mark.parametrize(
    'order, window',
    [(o, w) for o in [0, 1, 2, 5] for w in [1, 2, 5]])
def test_output(mfcc, order, window):
    delta = DeltaProcessor(order=order, window=window).process(mfcc)
    assert delta.shape[0] == mfcc.shape[0]
    assert delta.shape[1] == mfcc.shape[1] * (order+1)
    assert np.array_equal(delta.times, mfcc.times)
    assert delta.data[:, :mfcc.shape[1]] == pytest.approx(mfcc.data)
