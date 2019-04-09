"""Test the computed features are stable accross computations"""

import pytest

from shennong.features.processor.bottleneck import BottleneckProcessor
from shennong.features.processor.filterbank import FilterbankProcessor
from shennong.features.processor.mfcc import MfccProcessor
from shennong.features.processor.onehot import (
    OneHotProcessor, FramedOneHotProcessor)
from shennong.features.processor.pitch import PitchProcessor
from shennong.features.processor.plp import PlpProcessor


PROCESSORS = [
    FilterbankProcessor,
    MfccProcessor,
    PlpProcessor,
    BottleneckProcessor,
    OneHotProcessor,
    FramedOneHotProcessor,
    PitchProcessor]


# here we computes the same features two times and ensure we obtain the
# same results, allowing reruns because bottleneck has a dither we
# cannot disable.
@pytest.mark.flaky(reruns=20)
@pytest.mark.parametrize(
    'processor, same', [(p, s) for p in PROCESSORS for s in (True, False)])
def test_stable(processor, same, audio, alignments):
    if processor in (OneHotProcessor, FramedOneHotProcessor):
        audio = alignments['S01F1522_0003']

    p1 = processor()
    p2 = p1 if same else processor()

    # disable dithering in mel-based processors to have exactly the
    # same output
    try:
        p1.dither = 0
        p2.dither = 0
    except AttributeError:
        pass

    f1 = p1.process(audio)
    f2 = p2.process(audio)

    if processor is BottleneckProcessor:
        # bottleneck processor adds a little dither we cannot disable
        assert f1.is_close(f2, rtol=5e-1, atol=5e-1)
    else:
        assert f1 == f2
