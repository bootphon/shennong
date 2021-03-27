"""Test of parallel features processing"""

import multiprocessing
import os
import pytest

from shennong.processor import BottleneckProcessor, MfccProcessor


@pytest.mark.parametrize('proc', [MfccProcessor, BottleneckProcessor])
def test_process_all(audio, proc):
    # the bottleneck test hangs on travis, skipping it
    if proc == BottleneckProcessor and 'ON_TRAVIS' in os.environ:
        pytest.skip('unsupported on travis')

    signals = {f'{n}': audio for n in range(3)}

    features = proc().process_all(signals)
    values = list(features.values())

    assert signals.keys() == features.keys()
    assert len(values) == 3
    equal = [values[0].is_close(v, atol=10) for v in values[1:]]
    assert all(equal)


def test_process_all_kwargs(audio_tiny):
    signals = {f'{n}': audio_tiny for n in range(3)}

    features = MfccProcessor().process_all(
        signals, vtln_warp={f'{n}': 1.0 for n in range(3)})
    assert signals.keys() == features.keys()

    with pytest.raises(TypeError):
        MfccProcessor().process_all(
            signals, bad_name={f'{n}': 1.0 for n in range(3)})

    with pytest.raises(ValueError) as err:
        MfccProcessor().process_all(signals, vtln_warp=1.0)
    assert 'is not a dict' in str(err.value)

    with pytest.raises(ValueError) as err:
        MfccProcessor().process_all(
            signals, vtln_warp={f'{n}': 1.0 for n in range(2)})
    assert 'have different keys' in str(err.value)


@pytest.mark.parametrize('njobs', [0, 1, 2, 1000])
def test_njobs(capsys, njobs, audio):
    signals = {'1': audio}
    p = MfccProcessor(sample_rate=audio.sample_rate)
    p.set_logger('debug')

    if njobs == 0:
        with pytest.raises(ValueError) as err:
            p.process_all(signals, njobs=njobs)
        assert 'must be strictly positive' in str(err.value)
        return

    features = p.process_all(signals, njobs=njobs)

    if njobs > multiprocessing.cpu_count():
        assert 'CPU cores but reducing to' in capsys.readouterr().err

    assert signals.keys() == features.keys()
