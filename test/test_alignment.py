"""Test of the module shennong.alignment"""

import numpy as np
import pytest

from shennong.alignment import Alignment, AlignmentCollection


@pytest.fixture(scope='session')
def ali():
    return Alignment.from_list([(0, 1, 'a'), (1, 2, 'b'), (2, 3.001, 'c')])


def test_simple(ali):
    assert ali.phones.shape == (3,)
    assert ali._times.shape == (3, 2)
    assert ali._times.dtype == np.float
    assert ali.duration() == pytest.approx(3.001)

    assert np.array_equal(np.array(['a', 'b', 'c']), ali.phones)
    assert np.array([[0, 1], [1, 2], [2, 3.001]]) == pytest.approx(ali._times)

    with pytest.raises(ValueError):
        ali[1]

    with pytest.raises(ValueError):
        ali[1:2:0]

    with pytest.raises(ValueError):
        ali[::0]


def test_valid(ali):
    assert ali.is_valid()

    ali = Alignment.from_list([])
    assert ali.is_valid()

    ali = Alignment.from_list([(0, 0, 'a')], validate=False)
    assert not ali.is_valid()

    with pytest.raises(ValueError):
        Alignment.from_list([(0, 1, 'a'), (1, 2, 'b'), (0, 3, 'a')])

    with pytest.raises(ValueError):
        Alignment.from_list([(0, 1, 'a'), (0, 3, 'c')])

    with pytest.raises(ValueError):
        Alignment.from_list([(0, 0, 'a')])

    with pytest.raises(ValueError):
        Alignment.from_list([(1, 0, 'a')])

    with pytest.raises(ValueError):
        Alignment.from_list([('a', 0, 'a')])

    with pytest.raises(ValueError):
        Alignment.from_list([(0, 'a', 'a')])

    with pytest.raises(ValueError):
        Alignment.from_list([(0, 1)])

    with pytest.raises(ValueError):
        Alignment.from_list([(0, 1, 'a', 'a')])


def test_attributes(ali):
    # read only attributes to protect alignment consistency
    with pytest.raises(AttributeError):
        ali.onsets = []

    with pytest.raises(AttributeError):
        ali.offsets = []

    with pytest.raises(AttributeError):
        ali.phones = []


def test_list(ali):
    ali2 = Alignment(np.array([[0, 1], [1, 2]]), np.array(['a', 'b']))

    assert ali[:2] == ali2
    assert ali2.to_list() == [(0, 1, 'a'), (1, 2, 'b')]
    assert Alignment.from_list(ali.to_list()) == ali


def test_repr(ali):
    assert str(ali[:1]) == '0.0 1.0 a'
    assert str(ali) == '0.0 1.0 a\n1.0 2.0 b\n2.0 3.001 c'
    assert str(ali[10:]) == ''


def test_partial_read(ali):
    a = ali[0:1]
    assert np.array_equal(np.array([[0, 1]]), a._times)
    assert np.array_equal(np.array(['a']), a.phones)

    a = ali[0:2]
    assert np.array_equal(np.array([[0, 1], [1, 2]]), a._times)
    assert np.array_equal(np.array(['a', 'b']), a.phones)

    a = ali[:2]
    assert np.array_equal(np.array([[0, 1], [1, 2]]), a._times)
    assert np.array_equal(np.array(['a', 'b']), a.phones)


def test_complete_read(ali):
    a = ali[0:]
    assert np.array_equal(ali._times, a._times)
    assert np.array_equal(ali.phones, a.phones)

    a = ali[-1:]
    assert np.array_equal(ali._times, a._times)
    assert np.array_equal(ali.phones, a.phones)

    a = ali[:5]
    assert np.array_equal(ali._times, a._times)
    assert np.array_equal(ali.phones, a.phones)

    a = ali[-5:5]
    assert np.array_equal(ali._times, a._times)
    assert np.array_equal(ali.phones, a.phones)

    a = ali[:]
    assert np.array_equal(ali._times, a._times)
    assert np.array_equal(ali.phones, a.phones)


def test_empty_read(ali):
    a = ali[:-1]
    assert len(a._times) == 0
    assert len(a.phones) == 0

    a = ali[0:-1]
    assert len(a._times) == 0
    assert len(a.phones) == 0

    a = ali[10:]
    assert len(a._times) == 0
    assert len(a.phones) == 0


@pytest.mark.parametrize('t', [0, 0.5, 1, 3.001])
def test_read_oneinstant(ali, t):
    # read [t, t[ is like reading nothing
    a = ali[t:t]
    assert a.duration() == 0
    assert len(a.phones) == 0


def test_read_interphones(ali):
    a = ali[0:0.8]
    assert np.array([0, 0.8]).reshape(1, 2) == pytest.approx(a._times)
    assert np.array_equal(a.phones, np.array(['a']))

    a = ali[0:1]
    assert np.array([0, 1]).reshape(1, 2) == pytest.approx(a._times)
    assert np.array_equal(a.phones, np.array(['a']))

    a = ali[0.2:0.8]
    assert np.array([0.2, 0.8]).reshape(1, 2) == pytest.approx(a._times)
    assert np.array_equal(a.phones, np.array(['a']))

    a = ali[0.2:1]
    assert np.array([0.2, 1]).reshape(1, 2) == pytest.approx(a._times)
    assert np.array_equal(a.phones, np.array(['a']))

    a = ali[1.2:1.8]
    assert np.array([1.2, 1.8]).reshape(1, 2) == pytest.approx(a._times)
    assert np.array_equal(a.phones, np.array(['b']))

    a = ali[0.2:1.8]
    assert np.array([[0.2, 1], [1, 1.8]]) == pytest.approx(a._times)
    assert np.array_equal(a.phones, np.array(['a', 'b']))

    a = ali[0.2:2.8]
    assert np.array([[0.2, 1], [1, 2], [2, 2.8]]) == pytest.approx(a._times)
    assert np.array_equal(a.phones, np.array(['a', 'b', 'c']))

    a = ali[0.2:4]
    assert np.array([[0.2, 1], [1, 2], [2, 3.001]]) == pytest.approx(a._times)
    assert np.array_equal(a.phones, np.array(['a', 'b', 'c']))


def test_realdata(alignments):
    ali = alignments['S01F1522_0003']
    assert ali.phones.shape == (38,)
    assert np.array_equal(ali.phones[:3], np.array(['k', 'y', 'o']))
    assert np.array_equal(ali[:0.1425].phones, np.array(['k', 'y', 'o']))

    assert ali.duration() == pytest.approx(3.1)
    assert ali[3.2:].phones.shape == (0,)


def test_sample_rate():
    ali = Alignment.from_list([[0, 1, 'a'], [1, 3, 'b']])
    assert list(ali.at_sample_rate(1)) == ['a', 'b', 'b']   # all
    assert list(ali[:1].at_sample_rate(1)) == ['a']  # 1st second
    assert list(ali[:1].at_sample_rate(4)) == ['a'] * 4
    assert list(ali.at_sample_rate(4)) == ['a'] * 4 + ['b'] * 8
    assert len(list(ali.at_sample_rate(100))) == ali.duration() * 100

    ali = Alignment.from_list([[0, 0.8, 'a'], [0.8, 1, 'b']])
    assert list(ali.at_sample_rate(1)) == ['a']
    assert list(ali.at_sample_rate(2)) == ['a', 'a']
    assert list(ali.at_sample_rate(5)) == ['a', 'a', 'a', 'a', 'b']
    assert list(ali.at_sample_rate(10)) == ['a', 'a', 'a', 'a'] * 2 + ['b'] * 2

    ali = Alignment.from_list([[0, 0.2, 'a'], [0.2, 1, 'b']])
    assert list(ali.at_sample_rate(1)) == ['a']
    assert list(ali.at_sample_rate(2)) == ['a', 'b']
    assert list(ali.at_sample_rate(5)) == ['a', 'b', 'b', 'b', 'b']

    ali = Alignment.from_list([[0, 0.5, 'a'], [0.5, 1, 'b']])
    assert list(ali.at_sample_rate(1)) == ['a']
    assert list(ali.at_sample_rate(2)) == ['a', 'b']
    assert list(ali.at_sample_rate(3)) == ['a', 'a', 'b']


def test_load(alignments):
    assert 'S01F1522_0001' in alignments
    assert len(alignments) == 34
    for a in alignments.values():
        assert a.is_valid()


def test_inventory(alignments):
    phones = alignments.get_phones_inventory()
    assert 'e:' in phones
    assert len(phones) == 32


@pytest.mark.parametrize(
    'sort, compress',
    [(s, c) for s in (True, False) for c in (True, False)])
def test_save(tmpdir, alignments, sort, compress):
    filename = 'ali.txt'
    if compress is True:
        filename += '.gz'

    alignments.save(str(tmpdir.join(filename)), sort=sort, compress=compress)

    alignments2 = AlignmentCollection.load(
        str(tmpdir.join(filename)), compress=compress)

    assert alignments['S01F1522_0001'] == alignments2['S01F1522_0001']
    assert alignments['S01F1522_0001'] != alignments2['S01F1522_0002']
    assert alignments == alignments2
