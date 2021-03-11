"""Test of the module shennong.logger"""

import logging
import pytest

import shennong.logger as logger


def test_null_logger(capsys):
    log = logger.null_logger()
    log.debug('DEBUG')
    log.info('INFO')
    log.warning('WARNING')
    log.error('ERROR')

    captured = capsys.readouterr()
    assert not captured.out
    assert not captured.err


@pytest.mark.parametrize(
    'level', ['debug', 'info', 'warning', 'error'])
def test_logger(capsys, level):
    log = logger.get_logger('test', level=level)
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


def test_logger_bad_level():
    with pytest.raises(ValueError) as err:
        logger.get_logger('test', level='bad')
    assert 'invalid logging level' in str(err.value)
