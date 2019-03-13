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
    BottleneckProcessor,
    FilterbankProcessor,
    MfccProcessor,
    OneHotProcessor,
    FramedOneHotProcessor,
    PitchProcessor,
    PlpProcessor]


@pytest.mark.parametrize('processor', PROCESSORS)
def test_stable_new_instance(processor, audio, alignments):
    if processor in (OneHotProcessor, FramedOneHotProcessor):
        alignment = alignments['S01F1522_0003']
        f1 = processor().process(alignment)
        f2 = processor().process(alignment)
    else:
        f1 = processor().process(audio)
        f2 = processor().process(audio)

    assert f1.is_close(f2)


@pytest.mark.parametrize('processor', PROCESSORS)
def test_stable_same_instance(processor, audio, alignments):
    p = processor()
    if processor in (OneHotProcessor, FramedOneHotProcessor):
        alignment = alignments['S01F1522_0003']
        f1 = p.process(alignment)
        f2 = p.process(alignment)
    else:
        f1 = p.process(audio)
        f2 = p.process(audio)

    assert f1.is_close(f2)
