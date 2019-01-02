"""Test of the module shennong.features.onehot"""

import numpy as np
import pytest

from shennong.features import onehot, mfcc
from shennong import alignment
from shennong.core import window


def test_base(alignments):
    class Base(onehot._OneHotBase):
        def process(self, signal):
            return signal

    ali = alignments['S01F1522_0001']

    base = Base(phones=[])
    with pytest.raises(ValueError):
        base._phones_set(ali)

    base = Base()
    assert base._phones_set(ali) == ali.get_phones_inventory()

    extra = ali.get_phones_inventory()
    extra.add('!!')
    base = Base(phones=extra)
    assert '!!' in base._phones_set(ali)
    assert '!!' not in ali.get_phones_inventory()


def test_bad_phones(alignments):
    phn = alignments.get_phones_inventory()
    phn.remove('SIL')

    # a phone missing in the provided inventory
    proc = onehot.OneHotProcessor(phones=phn)
    with pytest.raises(ValueError):
        proc.process(alignments['S01F1522_0001'])


@pytest.mark.parametrize('times', ['mean', 'onset', 'offset'])
def test_simple(alignments, times):
    # various tests on the class OneHotProcessor
    ali1 = alignments['S01F1522_0001']
    phn1 = ali1.get_phones_inventory()
    if times is 'mean':
        tim1 = ali1._times.mean(axis=1)
    elif times is 'onset':
        tim1 = ali1.onsets
    else:
        tim1 = ali1.offsets

    all_phones = alignments.get_phones_inventory()

    # no phones_set specification, used the ones from ali1
    feat1 = onehot.OneHotProcessor(times=times, phones=phn1).process(ali1)
    assert feat1.shape == (ali1.phones.shape[0], len(phn1))
    assert all(feat1.data.sum(axis=1) == 1)
    assert np.array_equal(feat1.times, tim1)
    assert set(feat1.properties.keys()) == set(
        ['phone2index', 'phones', 'times'])
    assert feat1.properties['times'] == times

    # phones_set used is the one from the whole alignment collection
    feat2 = onehot.OneHotProcessor(
        phones=all_phones, times=times).process(ali1)
    assert feat2.shape == (
        ali1.phones.shape[0], len(all_phones))
    assert all(feat2.data.sum(axis=1) != 0)
    assert np.array_equal(feat1.times, feat2.times)

    # another alignment with the whole phones_set
    ali2 = alignments['S01F1522_0002']
    feat3 = onehot.OneHotProcessor(
        phones=all_phones, times=times).process(ali2)
    assert feat3.shape == (ali2.phones.shape[0], len(all_phones))
    assert all(feat3.data.sum(axis=1) == 1)
    assert feat2.shape[1] == feat3.shape[1]
    assert feat1.shape[1] < feat3.shape[1]


def test_framed(alignments):
    ali = alignments['S01F1522_0010']
    assert ali.duration() == pytest.approx(0.7)

    feat = onehot.FramedOneHotProcessor().process(ali)
    assert all(feat.data.sum(axis=1) != 0)
    assert feat.shape[1] == len(ali.get_phones_inventory())

    length = onehot.FramedOneHotProcessor().frame.frame_length
    assert ali.duration() - length <= feat.times[-1]
    assert feat.times[-1] <= ali.duration() + length/2


def test_compare_mfcc(audio):
    # check if the frames are the same as for mfcc (using the same
    # signal duration)
    ali = alignment.Alignment.from_list(
        [(0, 1, 'a'), (1, audio.duration, 'b')])

    Onehot = onehot.FramedOneHotProcessor

    feat = Onehot(frame_shift=0.01).process(ali)
    feat_mfcc = mfcc.MfccProcessor(frame_shift=0.01).process(audio)

    assert feat.shape == (140, 2)
    assert feat.times == pytest.approx(feat_mfcc.times)
    assert Onehot(frame_shift=0.02).process(ali).shape == (70, 2)
    assert Onehot(
        frame_shift=0.02, frame_length=0.05).process(ali).shape == (69, 2)


@pytest.mark.parametrize('window', window.types())
def test_window(alignments, window):
    ali = alignments['S01F1522_0010']
    nphones = len(ali.get_phones_inventory())
    feat = onehot.FramedOneHotProcessor(window_type=window).process(ali)
    assert all(feat.data.sum(axis=1) != 0)
    assert feat.shape == (68, nphones)
