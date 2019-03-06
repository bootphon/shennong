"""Test of the module shennong.utils"""

import logging
import numpy as np
import pytest
import shennong.utils as utils


def test_version():
    from shennong import __version__
    assert isinstance(__version__, str)


def test_null_logger(capsys):
    log = utils.null_logger()
    log.debug('DEBUG')
    log.info('INFO')
    log.warning('WARNING')
    log.error('ERROR')

    captured = capsys.readouterr()
    assert not captured.out
    assert not captured.err


@pytest.mark.parametrize(
    'level', [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR])
def test_logger(capsys, level):
    log = utils.get_logger(level=level)
    log.debug('DEBUG')
    log.info('INFO')
    log.warning('WARNING')
    log.error('ERROR')

    captured = capsys.readouterr()
    assert not captured.out
    if level is logging.ERROR:
        assert 'ERROR' in captured.err
        assert 'WARNING' not in captured.err
        assert 'INFO' not in captured.err
        assert 'DEBUG' not in captured.err
    if level is logging.WARNING:
        assert 'ERROR' in captured.err
        assert 'WARNING' in captured.err
        assert 'INFO' not in captured.err
        assert 'DEBUG' not in captured.err
    if level is logging.INFO:
        assert 'ERROR' in captured.err
        assert 'WARNING' in captured.err
        assert 'INFO' in captured.err
        assert 'DEBUG' not in captured.err
    if level is logging.DEBUG:
        assert 'ERROR' in captured.err
        assert 'WARNING' in captured.err
        assert 'INFO' in captured.err
        assert 'DEBUG' in captured.err


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
