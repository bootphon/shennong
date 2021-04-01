"""Test of parallel features processing"""

import multiprocessing
import os
import pytest

from shennong import Utterances
from shennong.processor import BottleneckProcessor, MfccProcessor


@pytest.fixture(scope='session')
def utterances(wav_file):
    return Utterances([
        ('u1', wav_file, 0, 0.2),
        ('u2', wav_file, 0, 0.2),
        ('u3', wav_file, 0, 0.2)])


@pytest.mark.parametrize('proc', [MfccProcessor, BottleneckProcessor])
def test_process_all(utterances, proc):
    # the bottleneck test hangs on travis, skipping it
    if proc == BottleneckProcessor and 'ON_TRAVIS' in os.environ:
        pytest.skip('unsupported on travis')

    features = proc().process_all(utterances)
    values = list(features.values())

    assert utterances.by_name().keys() == features.keys()
    assert len(values) == 3
    equal = [values[0].is_close(v, atol=10) for v in values[1:]]
    assert all(equal)


def test_process_all_kwargs(utterances):
    features = MfccProcessor().process_all(
        utterances, vtln_warp={f'u{n+1}': 1.0 for n in range(3)})
    assert utterances.by_name().keys() == features.keys()

    with pytest.raises(TypeError):
        MfccProcessor().process_all(
            utterances,
            bad_name={f'u{n+1}': 1.0 for n in range(3)})

    with pytest.raises(ValueError) as err:
        MfccProcessor().process_all(utterances, vtln_warp=1.0)
    assert 'is not a dict' in str(err.value)

    with pytest.raises(ValueError) as err:
        MfccProcessor().process_all(
            utterances, vtln_warp={f'{n}': 1.0 for n in range(2)})
    assert 'have different names' in str(err.value)


@pytest.mark.parametrize('njobs', [0, 1, 2, 1000])
def test_njobs(capsys, njobs, utterances):
    p = MfccProcessor()
    p.set_logger('debug')

    if njobs == 0:
        with pytest.raises(ValueError) as err:
            p.process_all(utterances, njobs=njobs)
        assert 'must be strictly positive' in str(err.value)
        return

    features = p.process_all(utterances, njobs=njobs)

    if njobs > multiprocessing.cpu_count():
        assert 'CPU cores but reducing to' in capsys.readouterr().err

    assert utterances.by_name().keys() == features.keys()
