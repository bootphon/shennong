"""Test of the module shennong.features.features"""

import numpy as np
import pytest

from shennong import Features, FeaturesCollection
from shennong.processor.mfcc import MfccProcessor
from shennong.logger import get_logger


def test_init_bad():
    with pytest.raises(ValueError) as err:
        Features(0, 0, properties=0)
    assert 'data must be a numpy array' in str(err.value)

    with pytest.raises(ValueError) as err:
        Features(np.asarray([0]), 0, properties=0)
    assert 'times must be a numpy array' in str(err.value)

    with pytest.raises(ValueError) as err:
        Features(np.asarray([0]), np.asarray([0]), properties=0)
    assert 'properties must be a dictionnary' in str(err.value)

    with pytest.raises(ValueError) as err:
        Features(np.asarray([0]), np.asarray([0]), properties={0: 0})
    assert 'data dimension must be 2' in str(err.value)

    with pytest.raises(ValueError) as err:
        Features(np.asarray([[0], [0]]), np.random.random((2, 2, 2)))
    assert 'times dimension must be 1 or 2' in str(err.value)

    with pytest.raises(ValueError) as err:
        data = np.random.random((12, 2))
        data[2, 1] = np.nan
        Features(data, np.ones((12,)))
    assert 'data contains non-finite numbers' in str(err.value)


def test_tofrom_dict(mfcc):
    a = mfcc._to_dict()
    b = Features._from_dict(a)
    assert b == mfcc

    with pytest.raises(ValueError) as err:
        Features._from_dict({'data': a['data'], 'properties': a['properties']})
    assert 'missing keys: times' in str(err.value)


def test_equal(mfcc):
    # same object
    assert mfcc == mfcc
    assert mfcc.is_close(mfcc)
    # same data
    mfcc2 = mfcc.copy()
    assert mfcc == mfcc2
    assert mfcc.is_close(mfcc2)
    # not same shape
    mfcc2 = mfcc.concatenate(mfcc)
    assert not mfcc == mfcc2
    assert not mfcc.is_close(mfcc2)
    # not same dtype
    mfcc64 = mfcc.copy(dtype=np.float64)
    assert not mfcc == mfcc64
    assert mfcc.is_close(mfcc64)
    # not same properties
    mfcc2 = Features(mfcc.data, mfcc.times, properties={'foo': 0})
    assert not mfcc == mfcc2
    assert not mfcc.is_close(mfcc2)
    # not same times
    mfcc2 = Features(mfcc.data, mfcc.times + 1, properties=mfcc.properties)
    assert not mfcc == mfcc2
    assert not mfcc.is_close(mfcc2)
    # not same data
    mfcc2 = Features(mfcc.data + 1, mfcc.times, properties=mfcc.properties)
    assert not mfcc == mfcc2
    assert not mfcc.is_close(mfcc2)
    # not same data but close
    mfcc2 = Features(mfcc.data + 1, mfcc.times, properties=mfcc.properties)
    assert not mfcc == mfcc2
    assert mfcc.is_close(mfcc2, atol=1)
    # not same times but close
    mfcc2 = Features(mfcc.data, mfcc.times + 1, properties=mfcc.properties)
    assert not mfcc == mfcc2
    assert not mfcc.is_close(mfcc2, atol=1)


def test_validate(mfcc):
    feat = Features(mfcc.data, mfcc.times[:-2, :], validate=False)
    with pytest.raises(ValueError) as err:
        feat.validate()
    assert 'mismatch in number of frames' in str(err.value)


def test_copy(mfcc):
    # by copy we allocate new arrays
    mfcc2 = mfcc.copy()
    assert mfcc2 == mfcc
    assert mfcc2 is not mfcc
    assert mfcc2.data is not mfcc.data
    assert mfcc2.times is not mfcc.times
    assert mfcc2.properties is not mfcc.properties

    # by explicit construction the arrays are shared
    mfcc2 = Features(
        mfcc.data, mfcc.times, properties=mfcc.properties, validate=False)
    assert mfcc2 == mfcc
    assert mfcc2 is not mfcc
    assert mfcc2.data is mfcc.data
    assert mfcc2.times is mfcc.times
    assert mfcc2.properties is mfcc.properties

    # subsample must be a strictly positive integer
    with pytest.raises(ValueError):
        mfcc2 = mfcc.copy(subsample=9.12)
    with pytest.raises(ValueError):
        mfcc2 = mfcc.copy(subsample=0)
    with pytest.raises(ValueError):
        mfcc2 = mfcc.copy(subsample=-10)


def test_concatenate(mfcc):
    mfcc2 = mfcc.concatenate(mfcc)
    assert mfcc2.nframes == mfcc.nframes
    assert mfcc2.ndims == mfcc.ndims * 2
    assert mfcc2.properties != mfcc.properties
    assert mfcc2.properties['mfcc'] == mfcc.properties['mfcc']

    mfcc2 = Features(mfcc.data, mfcc.times + 1)
    with pytest.raises(ValueError) as err:
        mfcc.concatenate(mfcc2)
    assert 'times are not equal' in str(err.value)


def test_concatenate_tolerance(capsys):
    f1 = Features(np.random.random((12, 2)), np.ones((12,)))
    f2 = Features(np.random.random((10, 2)), np.ones((10,)))

    with pytest.raises(ValueError) as err:
        f1.concatenate(f2, tolerance=0)
    assert 'features have a different number of frames' in str(err.value)

    with pytest.raises(ValueError) as err:
        f1.concatenate(f2, tolerance=1)
    assert 'features differs number of frames, and greater than ' in str(
        err.value)

    f3 = f1.concatenate(f2, tolerance=2, log=get_logger('test', 'info'))
    assert f3.shape == (10, 4)
    assert 'WARNING' in capsys.readouterr().err

    f3 = f2.concatenate(f1, tolerance=2, log=get_logger('test', 'warning'))
    assert f3.shape == (10, 4)
    assert 'WARNING' in capsys.readouterr().err


def test_collection(mfcc):
    assert FeaturesCollection().is_valid()
    assert FeaturesCollection(mfcc=mfcc).is_valid()
    assert not FeaturesCollection(mfcc=Features(
        np.asarray([0]), 0, validate=False)).is_valid()


def test_collection_isclose():
    f1 = Features(np.random.random((10, 2)), np.ones((10,)))
    f2 = Features(np.random.random((10, 2)), np.ones((10,)))

    fc1 = FeaturesCollection(f1=f1, f2=f2)
    fc2 = FeaturesCollection(f1=f1, f2=Features(f2.data+1, f2.times))
    fc3 = FeaturesCollection(f1=f1, f3=f2)

    assert fc1.is_close(fc1)
    assert not fc1.is_close(fc2)
    assert fc1.is_close(fc2, atol=1)
    assert not fc1.is_close(fc3)


def test_partition():
    f1 = Features(np.random.random((10, 2)), np.ones((10,)))
    f2 = Features(np.random.random((5, 2)), np.ones((5,)))
    f3 = Features(np.random.random((5, 2)), np.ones((5,)))
    fc = FeaturesCollection(f1=f1, f2=f2, f3=f3)

    with pytest.raises(ValueError) as err:
        fp = fc.partition({'f1': 'p1', 'f2': 'p1'})
    assert ('following items are not defined in the partition index: f3'
            in str(err.value))

    fp = fc.partition({'f1': 'p1', 'f2': 'p1', 'f3': 'p2'})
    assert sorted(fp.keys()) == ['p1', 'p2']
    assert sorted(fp['p1'].keys()) == ['f1', 'f2']
    assert sorted(fp['p2'].keys()) == ['f3']

    assert fc.is_valid()
    for fc in fp.values():
        assert fc.is_valid()


def test_trim():
    f1 = Features(np.random.random((10, 2)), np.ones((10,)))
    f2 = Features(np.random.random((10, 2)), np.ones((10,)))
    fc = FeaturesCollection(f1=f1, f2=f2)

    with pytest.raises(ValueError) as err:
        vad = {'f3': np.random.choice([True, False], size=10),
               'f4': np.random.choice([True, False], size=10)}
        fct = fc.trim(vad)
    assert 'Vad keys are different from this keys.' in str(err.value)

    with pytest.raises(ValueError) as err:
        vad = {'f1': np.random.randint(0, 10, 10),
               'f2': np.random.randint(0, 10, 10)}
        fct = fc.trim(vad)
    assert 'Vad arrays must be arrays of bool.' in str(err.value)

    with pytest.raises(ValueError) as err:
        vad = {'f1': np.random.choice([True, False], size=10),
               'f2': np.random.choice([True, False], size=5)}
        fct = fc.trim(vad)
    assert 'Vad arrays length must be equal to the number of frames.' in str(
        err.value)

    vad = {'f1': np.array([True]*7+[False]*3),
           'f2': np.array([True]*5+[False]*5)}
    fct = fc.trim(vad)
    assert sorted(fct.keys()) == ['f1', 'f2']
    assert fct['f1'].shape == (7, 2)
    assert fct['f2'].shape == (5, 2)

    assert fc.is_valid()
    for fc in fct.values():
        assert fc.is_valid()


def test_1d_times_sorted():
    # 10 frames, 5 dims
    data = np.random.random((10, 5))

    p = MfccProcessor()
    times = p.times(10)
    assert times.shape == (10, 2)

    feats = Features(data, times[:, 1], validate=False)
    assert feats.is_valid()


def test_2d_times_unsorted():
    with pytest.raises(ValueError) as err:
        Features(np.random.random((10, 3)), np.random.random((10, 2)))
    assert 'times is not sorted in increasing order' in str(err.value)


def test_2d_times_badshape():
    with pytest.raises(ValueError) as err:
        Features(np.random.random((10, 3)), np.random.random((10, 3)))
    assert 'times shape[1] must be 2, it is 3' in str(err.value)
