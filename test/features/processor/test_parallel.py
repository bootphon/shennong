"""Test of parallel features processing"""

import numpy as np
import pytest

from shennong.utils import get_logger
from shennong.features.processor.mfcc import MfccProcessor
from shennong.features.processor.bottleneck import BottleneckProcessor


@pytest.mark.parametrize('proc', [MfccProcessor, BottleneckProcessor])
def test_process_all(audio, proc):
    signals = {'{}'.format(n): audio for n in range(3)}
    p = proc()
    features = p.process_all(signals)
    values = list(features.values())

    assert signals.keys() == features.keys()
    assert len(values) == 3
    equal = [values[0].is_close(v, atol=10) for v in values[1:]]
    # print(np.array_equal(values[0].times, values[1].times))
    # print(values[0].properties == values[1].properties)
    # print(np.array_equal(values[0].data, values[1].data))
    # print(values[0].shape, values[1].shape)
    # print(values[0].dtype, values[1].dtype)
    print((values[0].data - values[1].data).max())
    print((values[0].data - values[1].data).min())
    # print(values[0].data.min(), values[1].data.min())
    # print(values[0].data.mean(), values[1].data.mean())
    # print()
    print(equal)
    assert all(equal)


@pytest.mark.parametrize('njobs', [0, 1, 2, 1000])
def test_njobs(capsys, njobs, audio):
    get_logger().setLevel(0)
    signals = {'1': audio}
    p = MfccProcessor(sample_rate=audio.sample_rate)

    if njobs == 0:
        with pytest.raises(ValueError) as err:
            p.process_all(signals, njobs=njobs)
        assert 'must be strictly positive' in str(err)
        return

    features = p.process_all(signals, njobs=njobs)

    if njobs > p.max_njobs():
        assert 'CPU cores but reducing to' in capsys.readouterr().err

    assert signals.keys() == features.keys()
