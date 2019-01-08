"""Test of the module shennong.window"""

import numpy as np
import pytest

from shennong.features import window


@pytest.mark.parametrize(
    'type, length',
    [(t, l) for t in window.types() for l in (1, 2, 3, 10, 100)])
def test_window(type, length):
    win = window.window(length, type=type)
    assert win.ndim == 1
    assert win.shape == (length,)

    assert not np.any(np.isnan(win))
    assert win.max() <= 1.0
    assert win.min() >= 0.0

    assert not np.all(win == 0.0)
    if type == 'rectangular':
        assert np.all(win == 1.0)
    elif length > 2:
        assert not np.all(win == 1.0)

    if type == 'povey' and length > 2:
        assert win[0] == win[-1] == 0.0
