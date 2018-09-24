"""Test of the module shennong.features.onehot"""

import numpy as np
import pytest

from shennong.features.onehot import OneHotProcessor, FramedOneHotProcessor


def test_simple(alignments):
    ali1 = alignments['S01F1522_0001']
    ali2 = alignments['S01F1522_0002']

    feat1 = OneHotProcessor().process(ali1)
    assert feat1.shape == (ali1.phones.shape[0], len(set(ali1.phones)))
    assert all(feat1.data.sum(axis=1) != 0)

    feat2 = OneHotProcessor(phones=alignments.phones_set).process(ali1)
    assert feat2.shape == (ali1.phones.shape[0], len(ali1.phones_set))
    assert all(feat2.data.sum(axis=1) != 0)
    assert np.array_equal(feat1.times, feat2.times)

    feat3 = OneHotProcessor(phones=alignments.phones_set).process(ali2)
    assert feat3.shape == (ali2.phones.shape[0], len(ali2.phones_set))
    assert all(feat3.data.sum(axis=1) != 0)
    assert feat2.shape[1] == feat3.shape[1]
    assert feat1.shape[1] < feat3.shape[1]
