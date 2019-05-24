"""Test of the module shennong.features.onehot"""

import numpy as np
import pytest

from shennong import alignment
from shennong.features import window, frames
from shennong.features.processor import onehot, mfcc


@pytest.mark.parametrize('params', [
    {'tokens': ['a', 'b', 'c']},
    {'tokens': None}])
def test_params(params):
    proc = onehot.OneHotProcessor(**params)
    assert params == proc.get_params()

    proc = onehot.OneHotProcessor()
    proc.set_params(**params)
    assert params == proc.get_params()


def test_params_framed():
    params = {
        'tokens': ['a', 'b', 'c'],
        'sample_rate': 2,
        'frame_shift': 10,
        'frame_length': 25,
        'window_type': 'blackman',
        'blackman_coeff': 0.5}

    proc = onehot.FramedOneHotProcessor(**params)
    assert params == proc.get_params()

    proc = onehot.FramedOneHotProcessor()
    proc.set_params(**params)
    assert params == proc.get_params()


def test_base(alignments):
    class Base(onehot._OneHotBase):
        def process(self, signal):
            return signal

    ali = alignments['S01F1522_0001']

    base = Base(tokens=[])
    with pytest.raises(ValueError):
        base._tokens_set(ali)

    base = Base()
    assert base._tokens_set(ali) == ali.get_tokens_inventory()
    with pytest.raises(ValueError) as err:
        base.ndims
    assert 'cannot know their dimension' in str(err)

    extra = ali.get_tokens_inventory()
    extra.add('!!')
    base = Base(tokens=extra)
    assert '!!' in base._tokens_set(ali)
    assert '!!' not in ali.get_tokens_inventory()


def test_bad_tokens(alignments):
    phn = alignments.get_tokens_inventory()
    phn.remove('SIL')

    # a token missing in the provided inventory
    proc = onehot.OneHotProcessor(tokens=phn)
    with pytest.raises(ValueError):
        proc.process(alignments['S01F1522_0001'])


def test_simple(alignments):
    # various tests on the class OneHotProcessor
    ali1 = alignments['S01F1522_0001']
    phn1 = ali1.get_tokens_inventory()
    all_tokens = alignments.get_tokens_inventory()

    # no tokens_set specification, used the ones from ali1
    proc = onehot.OneHotProcessor(tokens=phn1)
    feat1 = proc.process(ali1)
    assert proc.ndims == feat1.ndims == len(phn1)
    assert feat1.shape == (ali1.tokens.shape[0], len(phn1))
    assert all(feat1.data.sum(axis=1) == 1)
    assert np.array_equal(feat1.times, ali1.times)
    assert set(feat1.properties['onehot'].keys()) == set(
        ['token2index', 'tokens'])

    # tokens_set used is the one from the whole alignment collection
    feat2 = onehot.OneHotProcessor(tokens=all_tokens).process(ali1)
    assert feat2.shape == (
        ali1.tokens.shape[0], len(all_tokens))
    assert all(feat2.data.sum(axis=1) != 0)
    assert np.array_equal(feat1.times, feat2.times)

    # another alignment with the whole tokens_set
    ali2 = alignments['S01F1522_0002']
    feat3 = onehot.OneHotProcessor(tokens=all_tokens).process(ali2)
    assert feat3.shape == (ali2.tokens.shape[0], len(all_tokens))
    assert all(feat3.data.sum(axis=1) == 1)
    assert feat2.shape[1] == feat3.shape[1]
    assert feat1.shape[1] < feat3.shape[1]


def test_framed(alignments):
    ali = alignments['S01F1522_0010']
    assert ali.duration() == pytest.approx(0.7)

    feat = onehot.FramedOneHotProcessor().process(ali)
    assert all(feat.data.sum(axis=1) != 0)
    assert feat.shape[1] == len(ali.get_tokens_inventory())

    length = onehot.FramedOneHotProcessor().frame.frame_length
    assert ali.duration() - length <= feat.times[-1, -1]
    assert feat.times[-1, -1] <= ali.duration() + length


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
    ntokens = len(ali.get_tokens_inventory())
    feat = onehot.FramedOneHotProcessor(window_type=window).process(ali)
    assert all(feat.data.sum(axis=1) != 0)
    assert feat.shape == (68, ntokens)


def test_sample_rate():
    ali = alignment.Alignment(
        np.asarray([[0, 1], [1, 2]]), np.asarray(['a', 'b']))

    with pytest.raises(ValueError) as err:
        onehot.FramedOneHotProcessor(sample_rate=2).process(ali)
    assert 'sample rate too low' in str(err)

    feats = onehot.FramedOneHotProcessor(sample_rate=1000).process(ali)
    assert feats.nframes == frames.Frames(sample_rate=1000).nframes(2000)
