"""Test of the module shennong.base"""

import pytest

from shennong import version, version_long, url
from shennong.base import BaseProcessor
from shennong.features.processor.mfcc import MfccProcessor


def test_version():
    assert isinstance(version(), str)
    assert isinstance(version(type=str), str)
    assert isinstance(version(type=tuple), tuple)
    assert isinstance(version(type='str'), str)
    assert isinstance(version(type='tuple'), tuple)
    assert '.'.join(version(type=tuple)) == version()
    assert (
        len(version(type=tuple, full=False))
        <= len(version(type=tuple, full=True)))

    with pytest.raises(ValueError) as err:
        version(type=int)
    assert 'version type must be str or tuple' in str(err.value)

    with pytest.raises(ValueError) as err:
        version(type='abc')
    assert 'version type must be str or tuple' in str(err.value)

    assert version() in version_long()
    assert 'gpl' in version_long().lower()
    assert 'copyright' in version_long().lower()
    assert 'shennong' in url()


class ProcessorTest(BaseProcessor):
    def __init__(self, *params):
        pass


class ProcessorNested(BaseProcessor):
    def __init__(self, a, mfcc):
        self.a = a
        self.mfcc = mfcc


def test_get_params():
    assert BaseProcessor._get_param_names() == []

    p = ProcessorTest()
    with pytest.raises(RuntimeError) as err:
        p.get_params()
    assert 'specify their parameters in the signature' in str(err.value)

    m = MfccProcessor()
    p = ProcessorNested(1, m)
    print(p.get_params())
    assert m.get_params() == {
        k.replace('mfcc__', ''): v for k, v in p.get_params().items()
        if 'mfcc__' in k}


def test_set_params():
    p = ProcessorNested(1, MfccProcessor())
    assert p.set_params() == p

    with pytest.raises(ValueError) as err:
        p.set_params(spam=True)
    assert 'invalid parameter spam' in str(err.value)

    p.set_params(mfcc__sample_rate=2)
    assert p.mfcc.sample_rate == 2
    assert p.get_params()['mfcc__sample_rate'] == 2
