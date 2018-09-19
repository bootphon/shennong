"""Test of the module shennong.alignment"""

import numpy as np
import pytest

from shennong.alignment import Alignment, AlignmentCollection


def test_simple():
    a = [(0, 1, 'a'), (1, 2, 'b'), (2, 3.001, 'c')]
    ali = Alignment(a)
    assert ali.phones.shape == (3,)
    assert ali.times.shape == (3, 2)
    assert ali.times.dtype == np.float32
    assert ali.duration() == pytest.approx(3.001)

    assert ali.phones_set == {'a', 'b', 'c'}
    assert np.array_equal(np.array(['a', 'b', 'c']), ali.phones)
    assert np.array([[0, 1], [1, 2], [2, 3.001]]) == pytest.approx(ali.times)

    times, phones = ali[0:1]
    assert np.array_equal(np.array([[0, 1]]), times)
    assert np.array_equal(np.array(['a']), phones)

    times, phones = ali[0:2]
    assert np.array_equal(np.array([[0, 1], [1, 2]]), times)
    assert np.array_equal(np.array(['a', 'b']), phones)

    times, phones = ali[:2]
    assert np.array_equal(np.array([[0, 1], [1, 2]]), times)
    assert np.array_equal(np.array(['a', 'b']), phones)

    times, phones = ali[0:]
    assert np.array_equal(ali.times, times)
    assert np.array_equal(ali.phones, phones)

    times, phones = ali[-5:5]
    assert np.array_equal(ali.times, times)
    assert np.array_equal(ali.phones, phones)

    times, phones = ali[:]
    assert np.array_equal(ali.times, times)
    assert np.array_equal(ali.phones, phones)

    times, phones = ali[:-1]
    assert len(times) == 0
    assert len(phones) == 0


def test_realdata(alignments):
    ali = alignments['S01F1522_0003']
    assert ali.phones.shape == (38,)
    assert np.array_equal(ali.phones[:3], np.array(['k', 'y', 'o']))
    assert np.array_equal(ali[:0.1425][1], np.array(['k', 'y', 'o']))

    assert ali.duration() == pytest.approx(3.1)
    assert ali[3.2:][0].shape == (0, 2)


def test_sample_rate():
    ali = Alignment([[0, 1, 'a'], [1, 2, 'b']])
    assert list(ali.at_sample_rate(1)) == ['a', 'b']   # all
    assert list(ali.at_sample_rate(1, 0, 1)) == ['a']  # 1st second
    assert list(ali.at_sample_rate(4, 0, 1)) == ['a'] * 4
    assert list(ali.at_sample_rate(4)) == ['a'] * 4 + ['b'] * 4


def test_bad_init():
    with pytest.raises(ValueError):
        Alignment([(0, 1, 'a'), (1, 2, 'b'), (0, 3, 'a')])

    with pytest.raises(ValueError):
        Alignment([(0, 1, 'a'), (0, 3, 'c')])

    with pytest.raises(ValueError):
        Alignment([(0, 0, 'a')])

    with pytest.raises(ValueError):
        Alignment([(1, 0, 'a')])


def test_load(alignments):
    assert 'S01F1522_0001' in alignments
    assert len(alignments) == 34

    assert 'e:' in alignments.phones_set
    assert len(alignments.phones_set) == 32


@pytest.mark.parametrize(
    'sort, compress',
    [(s, c) for s in (True, False) for c in (True, False)])
def test_save(tmpdir, alignments, sort, compress):
    filename = 'ali.txt'
    if compress is True:
        filename += '.gz'

    alignments.save(tmpdir.join(filename), sort=sort, compress=compress)

    alignments2 = AlignmentCollection.load(
        tmpdir.join(filename), compress=compress)

    assert alignments['S01F1522_0001'] == alignments2['S01F1522_0001']
    assert alignments['S01F1522_0001'] != alignments2['S01F1522_0002']
    assert alignments == alignments2


def test_phones_set(alignments):
    # make sure all the items share the same phone set
    sets = [a.phones_set for a in alignments.values()]
    assert all(sets[i] == sets[i+1] for i in range(len(sets) - 1))

    # wheras this is not required for the real phones in each item
    v = list(alignments.values())
    assert not all(
        set(v[i].phones) == set(v[i+1].phones) for i in range(len(v) - 1))
