"""Test of the module shennong.processor.plp"""

import numpy as np
import pytest
import scipy.signal

from shennong import Audio
from shennong.processor import PlpProcessor
from shennong.processor.plp import RastaFilter


def test_params():
    assert len(PlpProcessor().get_params()) == 25

    params = {
        'num_bins': 0,
        'use_energy': True,
        'energy_floor': 10.0,
        'raw_energy': False,
        'htk_compat': True}
    p = PlpProcessor(**params)
    out_params = p.get_params()
    assert len(out_params) == 25

    assert PlpProcessor().set_params(**params).get_params() == out_params


@pytest.mark.parametrize('num_ceps', [-1, 0, 1, 5, 13, 23, 25])
def test_num_ceps(audio, num_ceps):
    if num_ceps >= 23:
        with pytest.raises(ValueError) as err:
            PlpProcessor(num_ceps=num_ceps)
        assert 'We must have num_ceps <= lpc_order+1' in str(err.value)
    else:
        if num_ceps > 0:
            proc = PlpProcessor(num_ceps=num_ceps)
            feat = proc.process(audio)
            assert proc.num_ceps == num_ceps == proc.ndims
            assert feat.shape == (140, num_ceps)

            proc.use_energy = False
            feat = proc.process(audio)
            assert feat.shape == (140, num_ceps)
        else:
            with pytest.raises(ValueError) as err:
                PlpProcessor(num_ceps=num_ceps)
            assert 'must be > 0' in str(err.value)


def test_htk_compat(audio):
    p1 = PlpProcessor(
        use_energy=True, htk_compat=False, dither=0).process(audio)
    p2 = PlpProcessor(
        use_energy=True, htk_compat=True, dither=0).process(audio)
    assert p1.data[:, 0] == pytest.approx(p2.data[:, -1])

    p1 = PlpProcessor(
        use_energy=False, htk_compat=False, dither=0).process(audio)
    p2 = PlpProcessor(
        use_energy=False, htk_compat=True, dither=0).process(audio)
    assert p1.data[:, 0] == pytest.approx(p2.data[:, -1])


def test_output(audio):
    assert PlpProcessor(
        cepstral_lifter=0, cepstral_scale=0.9).process(audio).shape \
        == (140, 13)
    assert PlpProcessor(frame_shift=0.01).process(audio).shape == (140, 13)
    assert PlpProcessor(frame_shift=0.02).process(audio).shape == (70, 13)
    assert PlpProcessor(
        frame_shift=0.02, frame_length=0.05).process(audio).shape == (69, 13)
    assert PlpProcessor(snip_edges=False).process(audio).shape == (142, 13)
    assert PlpProcessor(
        snip_edges=False, rasta=True).process(audio).shape \
        == (142, 13)

    feat = PlpProcessor(
        use_energy=True, raw_energy=False,
        energy_floor=np.exp(50)).process(audio)
    assert feat.shape == (140, 13)
    assert np.all(feat.data[:, 0] == 50)

    # sample rate mismatch
    with pytest.raises(ValueError):
        PlpProcessor(sample_rate=8000).process(audio)

    # only mono signals are accepted
    with pytest.raises(ValueError):
        data = np.random.random((1000, 2))
        stereo = Audio(data, sample_rate=16000)
        PlpProcessor(sample_rate=stereo.sample_rate).process(stereo)


# original implementation from https://github.com/mystlee/rasta_py
def _rastafilt(x):
    numer = np.arange(-2, 3)
    numer = -numer / np.sum(numer ** 2)
    denom = np.array([1, -0.94])

    zi = scipy.signal.lfilter_zi(numer, 1)
    y = np.zeros((x.shape))
    for i in range(x.shape[0]):
        y1, zi = scipy.signal.lfilter(
            numer, 1, x[i, 0:4], axis=0, zi=zi * x[i, 0])
        y1 = y1*0
        y2, _ = scipy.signal.lfilter(
            numer, denom, x[i, 4:x.shape[1]], axis=0, zi=zi)
        y[i, :] = np.append(y1, y2)
    return y


# replicate the original implementation on a pulse response
def test_rasta():
    sin = np.sin(2 * np.pi * np.arange(
        16000 * 0.05) * 200 / 16000).astype(np.float32)[5:]
    pulse = np.zeros((sin.shape[0],))
    pulse[0] = 1
    rand = np.random.random((sin.shape))
    data = np.dstack((sin, rand, pulse)).squeeze()

    rasta = RastaFilter(data.shape[1])
    fdata = np.array([rasta.filter(s, do_log=False) for s in data])
    fdata2 = _rastafilt(np.atleast_2d(data.T)).T
    assert np.all(fdata == fdata2)
