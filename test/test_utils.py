"""Test of the module shennong.utils"""

import os
import numpy as np
import pytest
import shennong.utils as utils


def test_version():
    from shennong import __version__
    assert isinstance(__version__, str)



@pytest.mark.parametrize('x', ['abc', 0, {'a': 0}, {0, 1}])
def test_listarray_simple(x):
    f = utils.list2array
    g = utils.array2list

    assert x == f(x)
    assert x == g(x)
    assert x == f(g(x))
    assert x == g(f(x))


def test_listarray():
    f = utils.list2array
    g = utils.array2list

    a = [[1, 2], [3, 4]]
    b = np.asarray(a)
    assert a == g(f(a))
    assert np.array_equal(b, f(a))
    assert np.array_equal(b, f(g(a)))

    c = f({'a': a})
    d = g(c)
    assert list(c.keys()) == ['a'] and np.array_equal(b, c['a'])
    assert d == {'a': a}


def test_dict_equal():
    f = utils.dict_equal
    assert f(0, 0)
    assert f({'a': 0}, {'a': 0})
    assert not f({'b': 0}, {'a': 0})
    assert not f({'a': 1}, {'a': 0})

    assert f({'a': np.asarray([1, 2])}, {'a': np.asarray([1, 2])})


def test_listfiles_nodir(data_path):
    f = utils.list_files_with_extension
    assert f('/foo/bar', '.wav') == []


@pytest.mark.parametrize(
    'abspath, realpath, recursive',
    [(a, r, s)
     for a in (True, False)
     for r in (True, False)
     for s in (True, False)])
def test_listfiles(data_path, abspath, realpath, recursive):
    f = utils.list_files_with_extension
    wavs = f(data_path, '.wav',
             abspath=abspath, realpath=realpath, recursive=recursive)
    assert [os.path.basename(w) for w in wavs] == [
        'test.8k.wav', 'test.float32.wav', 'test.wav']


def test_catch_exceptions(capsys):
    def f1():
        raise ValueError('foo')

    @utils.CatchExceptions
    def g1():
        return f1()

    with pytest.raises(ValueError):
        f1()
    with pytest.raises(SystemExit):
        g1()
    assert 'fatal error: foo' in capsys.readouterr().err

    def f2():
        raise KeyboardInterrupt

    @utils.CatchExceptions
    def g2():
        return f2()
    with pytest.raises(SystemExit):
        g2()
    assert 'keyboard interruption' in capsys.readouterr().err
