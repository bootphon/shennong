# coding: utf-8

"""Test of the module shennong.features.handlers"""

import os
import pytest

from shennong.features import Features, FeaturesCollection
from shennong.features.handlers import NumpyHandler, MatlabHandler, JsonHandler


HANDLERS = [NumpyHandler, MatlabHandler, JsonHandler]


@pytest.fixture(scope='session')
def mfcc_col(mfcc):
    return FeaturesCollection(mfcc=mfcc)


@pytest.fixture(scope='session')
def mfcc_utf8(mfcc):
    props = mfcc.properties
    props['comments'] = '使用人口について正確な統計はないが、日本国'

    feats = FeaturesCollection()
    feats['æðÐ'] = Features(mfcc.data, mfcc.times, props)
    return feats


@pytest.mark.parametrize(
    'handler, ext', [(h, b) for h in HANDLERS for b in [True, False]])
def test_simple(mfcc_col, handler, ext, tmpdir):
    tmpfile = str(tmpdir.join('feats'))
    h = handler(tmpfile, append_ext=ext)
    h.save(mfcc_col)

    if ext:
        tmpfile += h._extension()
    assert os.path.exists(tmpfile)

    assert handler(tmpfile).load() == mfcc_col


@pytest.mark.parametrize('handler', HANDLERS)
def test_utf8(mfcc_utf8, handler, tmpdir):
    h = handler(str(tmpdir.join('feats')))
    h.save(mfcc_utf8)
    assert h.load() == mfcc_utf8
