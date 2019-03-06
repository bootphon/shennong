# coding: utf-8

"""Test of the module shennong.features.handlers"""

import copy
import getpass
import json
import numpy as np
import os
import pytest
import shutil

from shennong.features import Features, FeaturesCollection
import shennong.features.handlers as handlers


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


HANDLERS = [
    handlers.NumpyHandler,
    handlers.MatlabHandler,
    handlers.JsonHandler,
    handlers.H5featuresHandler,
    handlers.KaldiHandler]


@pytest.mark.parametrize('name', handlers.supported_handlers().keys())
def test_get_handler_byname(name):
    filename = 'foo.file'
    if name == 'kaldi':
        with pytest.raises(ValueError) as err:
            handlers.get_handler(FeaturesCollection, 'foo.file', name)
        assert 'the file extension must be ".ark", it is ".file"' in str(err)
        filename = 'foo.ark'

    h = handlers.get_handler(FeaturesCollection, filename, name)
    assert not os.path.isfile(filename)
    assert isinstance(h, handlers.supported_handlers()[name])


@pytest.mark.parametrize('ext', handlers.supported_extensions().keys())
def test_get_handler_byext(ext):
    h = handlers.get_handler(FeaturesCollection, 'foo' + ext, None)
    assert not os.path.isfile('foo' + ext)
    assert isinstance(h, handlers.supported_extensions()[ext])


def test_get_handler_bad():
    with pytest.raises(ValueError) as err:
        handlers.get_handler(int, 'foo', None)
    assert 'must be shennong.features.FeaturesCollection' in str(err)

    with pytest.raises(ValueError) as err:
        handlers.get_handler(FeaturesCollection, 'foo.spam', None)
    assert 'invalid extension .spam' in str(err)

    with pytest.raises(ValueError) as err:
        handlers.get_handler(FeaturesCollection, 'foo', None)
    assert 'no extension nor handler name specified' in str(err)

    with pytest.raises(ValueError) as err:
        handlers.get_handler(FeaturesCollection, 'foo.spam', 'spam')
    assert 'invalid handler spam' in str(err)


def test_load_nofile():
    h = handlers.get_handler(FeaturesCollection, 'foo.json', None)
    with pytest.raises(IOError) as err:
        h.load()
    assert 'file not found' in str(err)


@pytest.mark.skipif(getpass.getuser() == 'root', reason='executed as root')
def test_load_noreadable(tmpdir):
    f = str(tmpdir.join('foo.json'))
    h = handlers.get_handler(FeaturesCollection, f, None)
    open(f, 'w').write('spam a lot')
    os.chmod(f, 0o222)  # write-only
    with pytest.raises(IOError) as err:
        h.load()
    assert 'file not readable' in str(err)


def test_load_invalid(tmpdir, mfcc_col):
    f = str(tmpdir.join('foo.json'))
    h = handlers.get_handler(FeaturesCollection, f, None)
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
    h = handlers.get_handler(FeaturesCollection, f, None)
    with pytest.raises(IOError) as err:
        h.save(mfcc_col)
    assert 'file already exists' in str(err)


def test_save_not_collection(tmpdir, mfcc):
    f = str(tmpdir.join('foo.json'))
    h = handlers.get_handler(FeaturesCollection, f, None)
    with pytest.raises(ValueError) as err:
        h.save(mfcc)
    assert 'features must be FeaturesCollection but are Features' in str(err)


def test_save_invalid(tmpdir, mfcc):
    f = str(tmpdir.join('foo.json'))
    h = handlers.get_handler(FeaturesCollection, f, None)
    feats = FeaturesCollection(mfcc=Features(
        data=mfcc.data, times=mfcc.data, validate=False))
    with pytest.raises(ValueError) as err:
        h.save(feats)
    assert 'features are not valid' in str(err)


@pytest.mark.parametrize('handler', HANDLERS)
def test_simple(mfcc_col, handler, tmpdir):
    filename = 'feats.ark' if handler is handlers.KaldiHandler else 'feats'
    tmpfile = str(tmpdir.join(filename))
    h = handler(mfcc_col.__class__, tmpfile)
    h.save(mfcc_col)

    assert os.path.exists(tmpfile)
    mfcc_col2 = handler(mfcc_col.__class__, tmpfile).load()
    assert mfcc_col2 == mfcc_col


@pytest.mark.parametrize('handler', HANDLERS)
def test_utf8(mfcc_utf8, handler, tmpdir):
    filename = 'feats.ark' if handler is handlers.KaldiHandler else 'feats'
    h = handler(mfcc_utf8.__class__, str(tmpdir.join(filename)))
    h.save(mfcc_utf8)
    mfcc2 = h.load()
    assert mfcc2 == mfcc_utf8


@pytest.mark.parametrize('handler', HANDLERS)
def test_heterogeneous(mfcc, handler, tmpdir):
    mfcc_col = FeaturesCollection(
        mfcc32=mfcc, mfcc64=mfcc.copy(dtype=np.float64))

    filename = 'feats.ark' if handler is handlers.KaldiHandler else 'feats'
    h = handler(mfcc_col.__class__, str(tmpdir.join(filename)))

    # h5features doesn't support heteregoneous data
    if handler is handlers.H5featuresHandler:
        with pytest.raises(IOError) as err:
            h.save(mfcc_col)
        assert 'features must be homogeneous' in str(err)
    else:
        h.save(mfcc_col)
        mfcc2 = h.load()
        assert mfcc2 == mfcc_col


@pytest.mark.parametrize('scp', [True, False])
def test_kaldihandler(mfcc_col, tmpdir, scp):
    mfcc_col.save(str(tmpdir.join('foo.ark')), scp=scp)
    assert os.path.isfile(str(tmpdir.join('foo.ark')))
    assert os.path.isfile(str(tmpdir.join('foo.times.ark')))
    assert os.path.isfile(str(tmpdir.join('foo.properties.json')))
    if scp:
        assert os.path.isfile(str(tmpdir.join('foo.scp')))
        assert os.path.isfile(str(tmpdir.join('foo.times.scp')))

    mfcc_col2 = FeaturesCollection.load(str(tmpdir.join('foo.ark')))
    assert mfcc_col2 == mfcc_col


def test_kaldihandler_baditems(tmpdir, mfcc_col):
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
def test_kaldihandler_badfile(tmpdir, mfcc_col, missing):
    filename = str(tmpdir.join('foo.ark'))
    mfcc_col.save(filename)
    os.remove(str(tmpdir.join(missing)))
    with pytest.raises(IOError) as err:
        FeaturesCollection.load(filename)
    assert 'file not found: {}'.format(str(tmpdir.join(missing))) in str(err)
