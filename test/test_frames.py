"""Test of the module shennong.core.frame"""

import numpy as np
import pytest
import random

from shennong.core.frames import Frames


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

    assert(np.array_equal(
        f.framed_array(np.arange(10)),
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
        f.boundaries(10),
        np.repeat(np.arange(n), 2).reshape(n, 2) + (0, 2))

    framed = np.arange(n).repeat(2).reshape(n, 2) + (0, 1)
    if not snip_edges:
        framed[-1, -1] = 8
    assert np.array_equal(f.framed_array(np.arange(10)), framed)


@pytest.mark.parametrize('snip_edges', [True, False])
def test_frames_2_2(snip_edges):
    f = Frames(sample_rate=1, snip_edges=snip_edges)
    f.frame_length = 2
    f.frame_shift = 2
    assert f.nframes(10) == 5
    assert f.samples_per_frame == 2
    assert f.samples_per_shift == 2

    assert np.array_equal(
        f.boundaries(10),
        np.repeat(np.arange(5) * 2, 2).reshape(5, 2) + (0, 2))

    assert np.array_equal(
        f.framed_array(np.arange(10)),
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
        f.boundaries(10),
        np.repeat(np.arange(5) * 2, 2).reshape(5, 2) + (0, 1))

    assert np.array_equal(
        f.framed_array(np.arange(10)),
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
        f.boundaries(10),
        np.repeat(np.arange(n), 2).reshape(n, 2) + (0, 3))

    framed = np.arange(n).repeat(3).reshape(n, 3) + (0, 1, 2)
    if snip_edges is False:
        framed[-2, -1] = 8
        framed[-1, -2:] = (8, 7)
    assert np.array_equal(f.framed_array(np.arange(10)), framed)


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
        f.boundaries(10),
        np.repeat(np.arange(n) * 3, 2).reshape(n, 2) + (0, 5))

    framed = (np.arange(n) * 3).repeat(5).reshape(n, 5) + (0, 1, 2, 3, 4)
    if snip_edges is False:
        framed[-1, -1] = 8
    assert np.array_equal(f.framed_array(np.arange(10)), framed)


@pytest.mark.parametrize(
    'ndim, snip_edges', [(n, s) for n in (1, 2, 3) for s in (True, False)])
def test_framed_array(ndim, snip_edges):
    # in that test we use default parameters (fs=16kHz, length=20ms,
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
    array = np.random.random(shape)

    # frame it
    frames = f.framed_array(array)
    print(array.shape, frames.shape)
    # TODO rebuild it


def test_discountinuous_array():
    pass
