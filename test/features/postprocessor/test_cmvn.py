"""Test of the shennong.features.cmvn module"""

import pytest
import numpy as np

from shennong.features import Features, FeaturesCollection
from shennong.features.postprocessor.cmvn import CmvnPostProcessor, apply_cmvn


def test_params():
    p = {'dim': 1, 'stats': None}
    c = CmvnPostProcessor(**p)
    assert c.get_params()['dim'] == 1
    assert c.get_params()['stats'].shape == (2, 2)
    assert c.get_params()['stats'].dtype == np.float64
    assert c.get_params()['stats'].sum() == 0.0

    with pytest.raises(ValueError) as err:
        c.set_params(**{'dim': None})
    assert 'cannot set attribute dim for CmvnPostProcessor' in str(err)

    with pytest.raises(ValueError) as err:
        c.set_params(**{'stats': None})
    assert 'cannot set attribute stats for CmvnPostProcessor' in str(err)


@pytest.mark.parametrize('dim', [-2, 0, 1, 3, 2.54, 'a'])
def test_dim(dim):
    if dim in (1, 3):
        p = CmvnPostProcessor(dim)
        assert p.dim == dim
    else:
        with pytest.raises(ValueError) as err:
            CmvnPostProcessor(dim)
        assert 'dimension must be a strictly positive integer' in str(err)


@pytest.mark.parametrize('norm_vars', [True, False])
def test_cmvn(mfcc, norm_vars):
    backup = mfcc.data.copy()

    proc = CmvnPostProcessor(mfcc.ndims)
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
    assert cmvn1.data.mean() == pytest.approx(0, abs=1e-6)
    if norm_vars is True:
        assert cmvn1.data.var(axis=0) == pytest.approx(np.ones(cmvn1.ndims))
    else:
        assert cmvn1.data.var(axis=0) == pytest.approx(mfcc.data.var(axis=0))
    assert mfcc.ndims == proc.dim == proc.ndims == cmvn1.ndims

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
    assert 'cmvn' not in mfcc.properties
    assert 'cmvn' in cmvn2.properties
    assert cmvn2.properties['cmvn']['stats'].shape == (2, 14)


def test_pre_stats(mfcc):
    with pytest.raises(ValueError) as err:
        CmvnPostProcessor(mfcc.ndims, stats=1)
    assert 'shape (2, 14), but is shaped as ()' in str(err)

    with pytest.raises(ValueError) as err:
        CmvnPostProcessor(mfcc.ndims, stats=np.random.random((2, mfcc.ndims)))
    assert 'shape (2, 14), but is shaped as (2, 13)' in str(err)

    stats = np.random.random((2, mfcc.ndims+1))
    proc = CmvnPostProcessor(mfcc.ndims, stats=stats.copy())
    assert stats == pytest.approx(proc.stats)


def test_weights(mfcc):
    weights = np.zeros(mfcc.nframes)
    proc = CmvnPostProcessor(dim=mfcc.ndims)
    proc.accumulate(mfcc, weights=weights)
    assert proc.count == 0

    weights = np.ones(mfcc.nframes)
    proc = CmvnPostProcessor(dim=mfcc.ndims)
    proc.accumulate(mfcc, weights=weights)
    assert proc.count == mfcc.nframes

    weights = np.ones(mfcc.nframes) * 0.5
    proc = CmvnPostProcessor(dim=mfcc.ndims)
    proc.accumulate(mfcc, weights=weights)
    assert proc.count == mfcc.nframes * 0.5

    weights = np.zeros(mfcc.nframes)
    weights[:2] = 0.1
    proc = CmvnPostProcessor(dim=mfcc.ndims)
    proc.accumulate(mfcc, weights=weights)
    assert proc.count == pytest.approx(0.2)


def test_bad_weights(mfcc):
    proc = CmvnPostProcessor(dim=mfcc.ndims)

    with pytest.raises(ValueError) as err:
        proc.accumulate(mfcc, weights=np.asarray([[1, 2], [3, 4]]))
    assert 'weights must have a single dimension' in str(err)

    with pytest.raises(ValueError) as err:
        proc.accumulate(mfcc, weights=np.asarray([]))
    assert 'there is 0 weights but {} feature frames'.format(
        mfcc.nframes) in str(err)


def test_skip_dims(mfcc):
    proc = CmvnPostProcessor(mfcc.ndims)
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
    del cmvn5.properties['cmvn']
    del cmvn5.properties['pipeline']
    del mfcc.properties['pipeline']
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
    assert cmvns.mean(axis=0) == pytest.approx(0, abs=1e-5)
    assert cmvns.var(axis=0) == pytest.approx(1, abs=1e-5)


@pytest.mark.parametrize('skip_dims', [[0, 1], [-1], [13]])
def test_apply_cmvn_skipdims(features_collection, skip_dims):
    if skip_dims in ([-1], [13]):
        with pytest.raises(ValueError) as err:
            apply_cmvn(features_collection, skip_dims=skip_dims)
        assert 'out of bounds dimensions' in str(err)
    else:
        cmvns = apply_cmvn(
            features_collection, skip_dims=skip_dims, by_collection=False)
        for feats in cmvns.values():
            assert feats.data[:, 2:].mean(axis=0) == pytest.approx(0, abs=1e-5)
            assert feats.data[:, 2:].var(axis=0) == pytest.approx(1, abs=1e-5)

            assert feats.data[:, :2].mean(axis=0) != pytest.approx(0, abs=1e-5)
            assert feats.data[:, :2].var(axis=0) != pytest.approx(1, abs=1e-5)


def test_apply_cmvn_byfeatures(features_collection):
    cmvns = apply_cmvn(features_collection, by_collection=False)
    for feat in cmvns.values():
        assert feat.data.mean(axis=0) == pytest.approx(0, abs=1e-5)
        assert feat.data.var(axis=0) == pytest.approx(1, abs=1e-5)
