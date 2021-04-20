"""Test of the module shennong.features.frame"""

import numpy as np
import pytest
import random

from shennong.frames import Frames


def test_params():
    p = {'sample_rate': 1, 'frame_shift': 1,
         'frame_length': 1, 'snip_edges': False}
    assert Frames(**p).get_params() == p
    assert Frames().set_params(**p).get_params() == p


@pytest.mark.parametrize('snip_edges', [True, False])
def test_frames_1_1(snip_edges):
    f = Frames(sample_rate=1, snip_edges=snip_edges)
    f.frame_shift = 1
    f.frame_length = 1
    assert f.nframes(10) == 10
    assert f.samples_per_frame == 1
    assert f.samples_per_shift == 1
    assert np.array_equal(
        f.boundaries(10),
        np.repeat(np.arange(10), 2).reshape(10, 2) + (0, 1))

    print()
    print(f.make_frames(np.arange(10)))
    print(np.arange(10)[:, np.newaxis])

    assert(np.array_equal(
        f.make_frames(np.arange(10)),
        np.arange(10)[:, np.newaxis]))


@pytest.mark.parametrize('snip_edges', [True, False])
def test_frames_2_1(snip_edges):
    f = Frames(sample_rate=1, snip_edges=snip_edges)
    f.frame_length = 2
    f.frame_shift = 1
    n = 9 if snip_edges else 10
    assert f.nframes(10) == n
    assert f.samples_per_frame == 2
    assert f.samples_per_shift == 1

    assert np.array_equal(
        f.boundaries(n),
        np.repeat(np.arange(n), 2).reshape(n, 2) + (0, 2))

    framed = np.arange(n).repeat(2).reshape(n, 2) + (0, 1)
    if not snip_edges:
        framed[-1, -1] = 8

    print()
    print(f.make_frames(np.arange(10)))
    print(framed)

    assert np.array_equal(f.make_frames(np.arange(10)), framed)


@pytest.mark.parametrize('snip_edges', [True, False])
def test_frames_2_2(snip_edges):
    f = Frames(sample_rate=1, snip_edges=snip_edges)
    f.frame_length = 2
    f.frame_shift = 2
    assert f.nframes(10) == 5
    assert f.samples_per_frame == 2
    assert f.samples_per_shift == 2

    assert np.array_equal(
        f.boundaries(5),
        np.repeat(np.arange(5) * 2, 2).reshape(5, 2) + (0, 2))

    assert np.array_equal(
        f.make_frames(np.arange(10)),
        np.repeat(np.arange(5) * 2, 2).reshape(5, 2) + (0, 1))


@pytest.mark.parametrize('snip_edges', [True, False])
def test_frames_1_2(snip_edges):
    f = Frames(sample_rate=1, snip_edges=snip_edges)
    f.frame_length = 1
    f.frame_shift = 2
    assert f.nframes(10) == 5
    assert f.samples_per_frame == 1
    assert f.samples_per_shift == 2

    assert np.array_equal(
        f.boundaries(5),
        np.repeat(np.arange(5) * 2, 2).reshape(5, 2) + (0, 1))

    assert np.array_equal(
        f.make_frames(np.arange(10)),
        (np.arange(5) * 2)[:, np.newaxis])


@pytest.mark.parametrize('snip_edges', [True, False])
def test_frames_3_1(snip_edges):
    f = Frames(sample_rate=1, snip_edges=snip_edges)
    f.frame_length = 3
    f.frame_shift = 1
    n = 8 if snip_edges else 10
    assert f.nframes(10) == n
    assert f.samples_per_frame == 3
    assert f.samples_per_shift == 1

    assert np.array_equal(
        f.boundaries(n),
        np.repeat(np.arange(n), 2).reshape(n, 2) + (0, 3))

    framed = np.arange(n).repeat(3).reshape(n, 3) + (0, 1, 2)
    if snip_edges is False:
        framed[-2, -1] = 8
        framed[-1, -2:] = (8, 7)
    assert np.array_equal(f.make_frames(np.arange(10)), framed)


@pytest.mark.parametrize('snip_edges', [True, False])
def test_frames_5_3(snip_edges):
    f = Frames(sample_rate=1, snip_edges=snip_edges)
    f.frame_length = 5
    f.frame_shift = 3
    n = (2 if snip_edges else 3)
    assert f.nframes(9) == n
    assert f.samples_per_frame == 5
    assert f.samples_per_shift == 3

    assert np.array_equal(
        f.boundaries(n),
        np.repeat(np.arange(n) * 3, 2).reshape(n, 2) + (0, 5))

    framed = (np.arange(n) * 3).repeat(5).reshape(n, 5) + (0, 1, 2, 3, 4)
    if snip_edges is False:
        framed[-1, -1] = 8
    assert np.array_equal(f.make_frames(np.arange(10)), framed)


@pytest.mark.parametrize(
    'ndim, snip_edges, writeable',
    [(n, bool(s), bool(w)) for n in (1, 2, 3) for s in (0, 1) for w in (0, 1)])
def test_make_frames(ndim, snip_edges, writeable):
    # in that test we use default parameters (fs=16kHz, length=25ms,
    # shift=10ms)
    f = Frames(snip_edges=snip_edges)

    # random time in [0.1, 0.2] seconds
    nsamples = int((random.random() * 0.1 + 0.1) * f.sample_rate)

    # fill a random array
    shape = (nsamples,)
    if ndim >= 2:
        shape = shape + (2,)
    if ndim >= 3:
        shape = shape + (2,)

    aref = np.random.random(shape)
    array = np.copy(aref)

    # make the frames (by copy or by view)
    frames = f.make_frames(array, writeable=writeable)

    assert np.array_equal(array, aref)

    assert frames.shape == (
        f.nframes(aref.shape[0]), f.samples_per_frame) + aref.shape[1:]

    if writeable is False:
        with pytest.raises(ValueError):
            frames[0] = 0
    else:
        frames[0] = -1
        assert (frames[0] == -1).all()


def test_times():
    # default parameters (fs=16kHz, length=25ms, shift=10ms) for 100ms
    assert np.allclose(Frames().times(1600), np.asarray(
        [[0, 0.025],
         [0.01, 0.035],
         [0.02, 0.045],
         [0.03, 0.055],
         [0.04, 0.065],
         [0.05, 0.075],
         [0.06, 0.085],
         [0.07, 0.095]]))
