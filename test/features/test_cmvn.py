"""Test of the shennong.features.cmvn module"""

import pytest
import numpy as np

from shennong.features import Features, FeaturesCollection
from shennong.features.cmvn import CmvnProcessor, apply_cmvn


def test_params():
    p = {'dim': 1, 'stats': None}
    c = CmvnProcessor(**p)
    assert c.get_params()['dim'] == 1
    assert c.get_params()['stats'].shape == (2, 2)
    assert c.get_params()['stats'].dtype == np.float64
    assert c.get_params()['stats'].sum() == 0.0

    with pytest.raises(ValueError) as err:
        c.set_params(**{'dim': None})
    assert 'cannot set attribute dim for CmvnProcessor' in str(err)

    with pytest.raises(ValueError) as err:
        c.set_params(**{'stats': None})
    assert 'cannot set attribute stats for CmvnProcessor' in str(err)


@pytest.mark.parametrize('norm_vars', [True, False])
def test_cmvn(mfcc, norm_vars):
    backup = mfcc.data.copy()

    proc = CmvnProcessor(mfcc.ndims)
    assert proc.dim == mfcc.ndims

    # cannot process without accumulation
    with pytest.raises(ValueError) as err:
        proc.process(mfcc)
    assert 'insufficient accumulation of stats' in str(err)

    # accumulate
    proc.accumulate(mfcc)
    assert proc.count == mfcc.nframes

    # cmvn
    cmvn1 = proc.process(mfcc, norm_vars=norm_vars)
    assert np.array_equal(backup, mfcc.data)
    assert cmvn1.shape == mfcc.shape
    assert cmvn1.dtype == mfcc.dtype
    assert np.array_equal(cmvn1.times, mfcc.times)
    assert cmvn1.data.mean() == pytest.approx(0, abs=1e-7)
    if norm_vars is True:
        assert cmvn1.data.var(axis=0) == pytest.approx(np.ones(cmvn1.ndims))
    else:
        assert cmvn1.data.var(axis=0) == pytest.approx(mfcc.data.var(axis=0))

    # reverse cmvn
    cmvn2 = proc.process(cmvn1, norm_vars=norm_vars, reverse=True)
    assert cmvn2.shape == mfcc.shape
    assert cmvn1.dtype == mfcc.dtype
    assert np.array_equal(cmvn2.times, mfcc.times)
    assert cmvn2.data == pytest.approx(mfcc.data, abs=1e-5)

    # accumulate a second time
    stats = proc.stats.copy()
    proc.accumulate(mfcc)
    assert proc.stats == pytest.approx(stats * 2)

    assert np.array_equal(backup, mfcc.data)


def test_weights(mfcc):
    weights = np.zeros(mfcc.nframes)
    proc = CmvnProcessor(dim=mfcc.ndims)
    proc.accumulate(mfcc, weights=weights)
    assert proc.count == 0

    weights = np.ones(mfcc.nframes)
    proc = CmvnProcessor(dim=mfcc.ndims)
    proc.accumulate(mfcc, weights=weights)
    assert proc.count == mfcc.nframes

    weights = np.ones(mfcc.nframes) * 0.5
    proc = CmvnProcessor(dim=mfcc.ndims)
    proc.accumulate(mfcc, weights=weights)
    assert proc.count == mfcc.nframes * 0.5

    weights = np.zeros(mfcc.nframes)
    weights[:2] = 0.1
    proc = CmvnProcessor(dim=mfcc.ndims)
    proc.accumulate(mfcc, weights=weights)
    assert proc.count == pytest.approx(0.2)


def test_skip_dims(mfcc):
    proc = CmvnProcessor(mfcc.ndims)
    proc.accumulate(mfcc)

    cmvn1 = proc.process(mfcc, skip_dims=None)
    cmvn2 = proc.process(mfcc, skip_dims=[])
    assert cmvn1 == cmvn2

    cmvn3 = proc.process(mfcc, skip_dims=[0, 1, 2])
    assert cmvn3.data[:, :3] == pytest.approx(mfcc.data[:, :3])
    assert cmvn3.data[:, 3:] == pytest.approx(cmvn1.data[:, 3:])

    cmvn4 = proc.process(mfcc, skip_dims=[1, 2, 0])
    assert cmvn4 == cmvn3

    cmvn5 = proc.process(mfcc, skip_dims=list(range(mfcc.ndims)))
    assert cmvn5 == mfcc

    for d in ([-1], [-1, 2, 3], [100], [100, -1, 5]):
        with pytest.raises(ValueError):
            proc.process(mfcc, skip_dims=d)


def test_apply_weights(features_collection):
    cmvn1 = apply_cmvn(features_collection)

    with pytest.raises(ValueError) as err:
        apply_cmvn(features_collection, weights={})
    assert 'keys differ for ' in str(err)

    weights = {k: None for k in features_collection.keys()}
    cmvn3 = apply_cmvn(features_collection, weights=weights)
    assert cmvn3 == cmvn1


def test_apply_baddim(features_collection):
    feats = FeaturesCollection(
        {k: v.copy() for k, v in features_collection.items()})
    feats['new'] = Features(
        np.random.random((2, 1)), np.asarray([0, 1]))

    with pytest.raises(ValueError) as err:
        apply_cmvn(feats)
    assert 'must have consistent dimensions' in str(err)


def test_apply_cmvn_bycollection(features_collection):
    cmvns = apply_cmvn(features_collection, by_collection=True)
    cmvns = np.concatenate([f.data for f in cmvns.values()], axis=0)
    assert cmvns.shape == (
        sum(f.nframes for f in features_collection.values()),
        features_collection['0'].ndims)
    assert cmvns.mean(axis=0) == pytest.approx(0, abs=1e-6)
    assert cmvns.var(axis=0) == pytest.approx(1, abs=1e-6)


def test_apply_cmvn_byfeatures(features_collection):
    cmvns = apply_cmvn(features_collection, by_collection=False)
    for feat in cmvns.values():
        assert feat.data.mean(axis=0) == pytest.approx(0, abs=1e-6)
        assert feat.data.var(axis=0) == pytest.approx(1, abs=1e-6)
