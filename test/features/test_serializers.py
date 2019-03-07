# coding: utf-8

"""Test of the module shennong.features.serializers"""

import getpass
import json
import numpy as np
import os
import pytest
import shutil

from shennong.features import Features, FeaturesCollection
import shennong.features.serializers as serializers


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


SERIALIZERS = [
    serializers.NumpySerializer,
    serializers.MatlabSerializer,
    serializers.JsonSerializer,
    serializers.H5featuresSerializer,
    serializers.KaldiSerializer]


@pytest.mark.parametrize('name', serializers.supported_serializers().keys())
def test_get_serializer_byname(name):
    filename = 'foo.file'
    if name == 'kaldi':
        with pytest.raises(ValueError) as err:
            serializers.get_serializer(FeaturesCollection, 'foo.file', name)
        assert 'the file extension must be ".ark", it is ".file"' in str(err)
        filename = 'foo.ark'

    h = serializers.get_serializer(FeaturesCollection, filename, name)
    assert not os.path.isfile(filename)
    assert isinstance(h, serializers.supported_serializers()[name])


@pytest.mark.parametrize('ext', serializers.supported_extensions().keys())
def test_get_serializer_byext(ext):
    h = serializers.get_serializer(FeaturesCollection, 'foo' + ext, None)
    assert not os.path.isfile('foo' + ext)
    assert isinstance(h, serializers.supported_extensions()[ext])


def test_get_serializer_bad():
    with pytest.raises(ValueError) as err:
        serializers.get_serializer(int, 'foo', None)
    assert 'must be shennong.features.FeaturesCollection' in str(err)

    with pytest.raises(ValueError) as err:
        serializers.get_serializer(FeaturesCollection, 'foo.spam', None)
    assert 'invalid extension .spam' in str(err)

    with pytest.raises(ValueError) as err:
        serializers.get_serializer(FeaturesCollection, 'foo', None)
    assert 'no extension nor serializer name specified' in str(err)

    with pytest.raises(ValueError) as err:
        serializers.get_serializer(FeaturesCollection, 'foo.spam', 'spam')
    assert 'invalid serializer spam' in str(err)


def test_load_nofile():
    h = serializers.get_serializer(FeaturesCollection, 'foo.json', None)
    with pytest.raises(IOError) as err:
        h.load()
    assert 'file not found' in str(err)


@pytest.mark.skipif(getpass.getuser() == 'root', reason='executed as root')
def test_load_noreadable(tmpdir):
    f = str(tmpdir.join('foo.json'))
    h = serializers.get_serializer(FeaturesCollection, f, None)
    open(f, 'w').write('spam a lot')
    os.chmod(f, 0o222)  # write-only
    with pytest.raises(IOError) as err:
        h.load()
    assert 'file not readable' in str(err)


def test_load_invalid(tmpdir, mfcc_col):
    f = str(tmpdir.join('foo.json'))
    h = serializers.get_serializer(FeaturesCollection, f, None)
    h.save(mfcc_col)

    # remove 2 lines in the times array to corrupt the file
    data = json.load(open(f, 'r'))
    data['mfcc']['attributes']['_times']['__ndarray__'] = (
        data['mfcc']['attributes']['_times']['__ndarray__'][2:])
    open(f, 'w').write(json.dumps(data))

    with pytest.raises(ValueError) as err:
        h.load()
    assert 'features not valid in file' in str(err)


def test_save_exists(tmpdir, mfcc_col):
    f = str(tmpdir.join('foo.json'))
    open(f, 'w').write('something')
    h = serializers.get_serializer(FeaturesCollection, f, None)
    with pytest.raises(IOError) as err:
        h.save(mfcc_col)
    assert 'file already exists' in str(err)


def test_save_not_collection(tmpdir, mfcc):
    f = str(tmpdir.join('foo.json'))
    h = serializers.get_serializer(FeaturesCollection, f, None)
    with pytest.raises(ValueError) as err:
        h.save(mfcc)
    assert 'features must be FeaturesCollection but are Features' in str(err)


def test_save_invalid(tmpdir, mfcc):
    f = str(tmpdir.join('foo.json'))
    h = serializers.get_serializer(FeaturesCollection, f, None)
    feats = FeaturesCollection(mfcc=Features(
        data=mfcc.data, times=mfcc.data, validate=False))
    with pytest.raises(ValueError) as err:
        h.save(feats)
    assert 'features are not valid' in str(err)


@pytest.mark.parametrize('serializer', SERIALIZERS)
def test_simple(mfcc_col, serializer, tmpdir):
    filename = ('feats.ark' if serializer is serializers.KaldiSerializer
                else 'feats')
    tmpfile = str(tmpdir.join(filename))
    h = serializer(mfcc_col.__class__, tmpfile)
    h.save(mfcc_col)

    assert os.path.exists(tmpfile)
    mfcc_col2 = serializer(mfcc_col.__class__, tmpfile).load()
    assert mfcc_col2 == mfcc_col


@pytest.mark.parametrize('serializer', SERIALIZERS)
def test_utf8(mfcc_utf8, serializer, tmpdir):
    filename = ('feats.ark' if serializer is serializers.KaldiSerializer
                else 'feats')
    h = serializer(mfcc_utf8.__class__, str(tmpdir.join(filename)))
    h.save(mfcc_utf8)
    mfcc2 = h.load()
    assert mfcc2 == mfcc_utf8


@pytest.mark.parametrize('serializer', SERIALIZERS)
def test_heterogeneous(mfcc, serializer, tmpdir):
    mfcc_col = FeaturesCollection(
        mfcc32=mfcc, mfcc64=mfcc.copy(dtype=np.float64))

    filename = ('feats.ark' if serializer is serializers.KaldiSerializer
                else 'feats')
    h = serializer(mfcc_col.__class__, str(tmpdir.join(filename)))

    # h5features doesn't support heteregoneous data
    if serializer is serializers.H5featuresSerializer:
        with pytest.raises(IOError) as err:
            h.save(mfcc_col)
        assert 'features must be homogeneous' in str(err)
    else:
        h.save(mfcc_col)
        mfcc2 = h.load()
        assert mfcc2 == mfcc_col


@pytest.mark.parametrize('scp', [True, False])
def test_kaldiserializer(mfcc_col, tmpdir, scp):
    mfcc_col.save(str(tmpdir.join('foo.ark')), scp=scp)
    assert os.path.isfile(str(tmpdir.join('foo.ark')))
    assert os.path.isfile(str(tmpdir.join('foo.times.ark')))
    assert os.path.isfile(str(tmpdir.join('foo.properties.json')))
    if scp:
        assert os.path.isfile(str(tmpdir.join('foo.scp')))
        assert os.path.isfile(str(tmpdir.join('foo.times.scp')))

    mfcc_col2 = FeaturesCollection.load(str(tmpdir.join('foo.ark')))
    assert mfcc_col2 == mfcc_col


def test_kaldiserializer_baditems(tmpdir, mfcc_col):
    mfcc_col2 = FeaturesCollection(
        one=mfcc_col['mfcc'], two=mfcc_col['mfcc'])
    mfcc_col.save(str(tmpdir.join('one.ark')))
    mfcc_col2.save(str(tmpdir.join('two.ark')))

    os.remove(str(tmpdir.join('two.times.ark')))
    shutil.copyfile(
        str(tmpdir.join('one.times.ark')),
        str(tmpdir.join('two.times.ark')))
    with pytest.raises(ValueError) as err:
        FeaturesCollection.load(str(tmpdir.join('two.ark')))
    assert 'items differ in data and times' in str(err)

    os.remove(str(tmpdir.join('one.properties.json')))
    shutil.copyfile(
        str(tmpdir.join('two.properties.json')),
        str(tmpdir.join('one.properties.json')))
    with pytest.raises(ValueError) as err:
        FeaturesCollection.load(str(tmpdir.join('one.ark')))
    assert 'items differ in data and properties' in str(err)


@pytest.mark.parametrize(
    'missing', ['foo.ark', 'foo.times.ark', 'foo.properties.json'])
def test_kaldiserializer_badfile(tmpdir, mfcc_col, missing):
    filename = str(tmpdir.join('foo.ark'))
    mfcc_col.save(filename)
    os.remove(str(tmpdir.join(missing)))
    with pytest.raises(IOError) as err:
        FeaturesCollection.load(filename)
    assert 'file not found: {}'.format(str(tmpdir.join(missing))) in str(err)
