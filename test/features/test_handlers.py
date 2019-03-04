# coding: utf-8

"""Test of the module shennong.features.handlers"""

import getpass
import json
import os
import pytest

from shennong.features import Features, FeaturesCollection
import shennong.features.handlers as handlers


@pytest.fixture(scope='session')
def mfcc_col(mfcc):
    assert mfcc.nframes
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
    handlers.H5featuresHandler]


@pytest.mark.parametrize('name', handlers.supported_handlers().keys())
def test_get_handler_byname(name):
    h = handlers.get_handler(FeaturesCollection, 'foo.file', name)
    assert not os.path.isfile('foo.file')
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
    data['mfcc']['times'] = data['mfcc']['times'][2:]
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
    tmpfile = str(tmpdir.join('feats'))
    h = handler(mfcc_col.__class__, tmpfile)
    h.save(mfcc_col)

    assert os.path.exists(tmpfile)
    assert handler(mfcc_col.__class__, tmpfile).load() == mfcc_col


@pytest.mark.parametrize('handler', HANDLERS)
def test_utf8(mfcc_utf8, handler, tmpdir):
    h = handler(mfcc_utf8.__class__, str(tmpdir.join('feats')))
    h.save(mfcc_utf8)
    assert h.load() == mfcc_utf8
