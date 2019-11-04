"""Test of the module shennong.features.delta"""

import numpy as np
import pytest

from shennong.features.postprocessor.delta import DeltaPostProcessor


def test_params():
    d = DeltaPostProcessor()
    d.order = 0
    with pytest.raises(ValueError):
        d.window = 0
    with pytest.raises(ValueError):
        d.window = 2000
    d.window = 1

    assert d.get_params() == {'order': 0, 'window': 1}

    p = {'order': 0, 'window': 1}
    d = DeltaPostProcessor()
    assert d.get_params()['order'] == 2
    d.set_params(**p)
    assert d.get_params() == p


@pytest.mark.parametrize(
    'order, window',
    [(o, w) for o in [0, 1, 2, 5] for w in [1, 2, 5]])
def test_output(mfcc, order, window):
    delta = DeltaPostProcessor(order=order, window=window).process(mfcc)
    assert delta.shape[0] == mfcc.shape[0]
    assert delta.shape[1] == mfcc.shape[1] * (order+1)
    assert np.array_equal(delta.times, mfcc.times)
    assert delta.data[:, :mfcc.shape[1]] == pytest.approx(mfcc.data)


def test_ndims():
    with pytest.raises(ValueError) as err:
        DeltaPostProcessor().ndims
    assert 'output dimension for delta processor depends on input' in str(err.value)
