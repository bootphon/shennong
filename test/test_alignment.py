"""Test of the module shennong.alignment"""

import numpy as np
import pytest

from shennong.alignment import Alignment, AlignmentCollection


@pytest.fixture(scope='session')
def ali():
    return Alignment.from_list([(0, 1, 'a'), (1, 2, 'b'), (2, 3.001, 'c')])


def test_bad_file():
    with pytest.raises(ValueError) as err:
        AlignmentCollection.load('/spam/spam/with/eggs')
    assert 'file not found' in str(err.value)


def test_bad_data():
    with pytest.raises(ValueError) as err:
        AlignmentCollection([['a', 1, 2, 'a'], ['a', 2, 3]])
    assert 'alignment must have 4 columns but line 2 has 3' in str(err.value)

    with pytest.raises(ValueError) as err:
        AlignmentCollection([['a', 1, 2, 'a'], ['a', 1, 2, 'b']])
    assert 'item a: mismatch in tstop/tstart timestamps' in str(err.value)


def test_simple(ali):
    assert ali.tokens.shape == (3,)
    assert ali._times.shape == (3, 2)
    assert ali._times.dtype == np.float
    assert ali.duration() == pytest.approx(3.001)

    assert np.array_equal(np.array(['a', 'b', 'c']), ali.tokens)
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

    with pytest.raises(ValueError) as err:
        Alignment(np.asarray([[0, 1], [1, 2]]), np.asarray(['a', 'b', 'c']))
    assert 'timestamps and tokens must have the same length' in str(err.value)


def test_attributes(ali):
    # read only attributes to protect alignment consistency
    with pytest.raises(AttributeError):
        ali.onsets = []

    with pytest.raises(AttributeError):
        ali.offsets = []

    with pytest.raises(AttributeError):
        ali.tokens = []


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
    assert np.array_equal(np.array(['a']), a.tokens)

    a = ali[0:2]
    assert np.array_equal(np.array([[0, 1], [1, 2]]), a._times)
    assert np.array_equal(np.array(['a', 'b']), a.tokens)

    a = ali[:2]
    assert np.array_equal(np.array([[0, 1], [1, 2]]), a._times)
    assert np.array_equal(np.array(['a', 'b']), a.tokens)


def test_complete_read(ali):
    a = ali[0:]
    assert np.array_equal(ali._times, a._times)
    assert np.array_equal(ali.tokens, a.tokens)

    a = ali[-1:]
    assert np.array_equal(ali._times, a._times)
    assert np.array_equal(ali.tokens, a.tokens)

    a = ali[:5]
    assert np.array_equal(ali._times, a._times)
    assert np.array_equal(ali.tokens, a.tokens)

    a = ali[-5:5]
    assert np.array_equal(ali._times, a._times)
    assert np.array_equal(ali.tokens, a.tokens)

    a = ali[:]
    assert np.array_equal(ali._times, a._times)
    assert np.array_equal(ali.tokens, a.tokens)


def test_empty_read(ali):
    a = ali[:-1]
    assert len(a._times) == 0
    assert len(a.tokens) == 0

    a = ali[0:-1]
    assert len(a._times) == 0
    assert len(a.tokens) == 0

    a = ali[10:]
    assert len(a._times) == 0
    assert len(a.tokens) == 0


@pytest.mark.parametrize('t', [0, 0.5, 1, 3.001])
def test_read_oneinstant(ali, t):
    # read [t, t[ is like reading nothing
    a = ali[t:t]
    assert a.duration() == 0
    assert len(a.tokens) == 0


def test_read_intertokens(ali):
    a = ali[0:0.8]
    assert np.array([0, 0.8]).reshape(1, 2) == pytest.approx(a._times)
    assert np.array_equal(a.tokens, np.array(['a']))

    a = ali[0:1]
    assert np.array([0, 1]).reshape(1, 2) == pytest.approx(a._times)
    assert np.array_equal(a.tokens, np.array(['a']))

    a = ali[0.2:0.8]
    assert np.array([0.2, 0.8]).reshape(1, 2) == pytest.approx(a._times)
    assert np.array_equal(a.tokens, np.array(['a']))

    a = ali[0.2:1]
    assert np.array([0.2, 1]).reshape(1, 2) == pytest.approx(a._times)
    assert np.array_equal(a.tokens, np.array(['a']))

    a = ali[1.2:1.8]
    assert np.array([1.2, 1.8]).reshape(1, 2) == pytest.approx(a._times)
    assert np.array_equal(a.tokens, np.array(['b']))

    a = ali[0.2:1.8]
    assert np.array([[0.2, 1], [1, 1.8]]) == pytest.approx(a._times)
    assert np.array_equal(a.tokens, np.array(['a', 'b']))

    a = ali[0.2:2.8]
    assert np.array([[0.2, 1], [1, 2], [2, 2.8]]) == pytest.approx(a._times)
    assert np.array_equal(a.tokens, np.array(['a', 'b', 'c']))

    a = ali[0.2:4]
    assert np.array([[0.2, 1], [1, 2], [2, 3.001]]) == pytest.approx(a._times)
    assert np.array_equal(a.tokens, np.array(['a', 'b', 'c']))


def test_realdata(alignments):
    ali = alignments['S01F1522_0003']
    assert ali.tokens.shape == (38,)
    assert np.array_equal(ali.tokens[:3], np.array(['k', 'y', 'o']))
    assert np.array_equal(ali[:0.1425].tokens, np.array(['k', 'y', 'o']))

    assert ali.duration() == pytest.approx(3.1)
    assert ali[3.2:].tokens.shape == (0,)


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
    tokens = alignments.get_tokens_inventory()
    assert 'e:' in tokens
    assert len(tokens) == 32


@pytest.mark.parametrize(
    'sort, compress',
    [(s, c) for s in (True, False) for c in (True, False)])
def test_save(tmpdir, alignments, sort, compress):
    filename = 'ali.txt'
    if compress is True:
        filename += '.gz'

    alignments.save(str(tmpdir.join(filename)), sort=sort, compress=compress)

    # cannot rewrite an existing file
    with pytest.raises(ValueError) as err:
        alignments.save(str(tmpdir.join(filename)), sort=sort)
    assert 'already exist' in str(err.value)

    # cannot write in an unexisting directory
    with pytest.raises(ValueError) as err:
        alignments.save('/spam/spam/with/eggs', sort=sort)
    assert 'cannot write to' in str(err.value)

    alignments2 = AlignmentCollection.load(
        str(tmpdir.join(filename)), compress=compress)

    assert alignments['S01F1522_0001'] == alignments2['S01F1522_0001']
    assert alignments['S01F1522_0001'] != alignments2['S01F1522_0002']
    assert alignments == alignments2
