"""Test of the module shennong.processor.ubm"""

import pytest
import os
import numpy as np

import kaldi.gmm
import kaldi.matrix
import shennong.pipeline as pipeline
from shennong import Features, FeaturesCollection
from shennong.postprocessor.vad import VadPostProcessor
from shennong.processor.ubm import DiagUbmProcessor


def test_params():
    assert len(DiagUbmProcessor(2).get_params()) == 12

    params = {'num_gauss': 16, 'num_iters': 3, 'initial_gauss_proportion': 0.7}
    p = DiagUbmProcessor(**params)

    params_out = p.get_params()
    assert len(params_out) == 12
    for k, v in params.items():
        assert params_out[k] == v
    assert p.get_params() == params_out

    p = DiagUbmProcessor(2)
    assert p.name == 'ubm'
    p.set_params(**params_out)
    assert params_out == p.get_params()
    assert len(params_out) == 12
    for k, v in params.items():
        assert params_out[k] == v
    assert p.get_params() == params_out

    with pytest.raises(ValueError) as err:
        p = DiagUbmProcessor(1)
    assert 'Number of gaussians must be at least 2' in str(err.value)

    with pytest.raises(TypeError) as err:
        p = DiagUbmProcessor(2, features=0)
    assert 'Features configuration must be a dict' in str(err.value)

    wrong_config = DiagUbmProcessor(2).get_params()['features']
    del wrong_config['mfcc']
    with pytest.raises(ValueError) as err:
        p.features = wrong_config
    assert 'Need mfcc features to train UBM-GMM' in str(err.value)

    with pytest.raises(TypeError) as err:
        p = DiagUbmProcessor(2, vad=0)
    assert 'VAD configuration must be a dict' in str(err.value)

    wrong_config = VadPostProcessor().get_params()
    wrong_config['wrong'] = 0
    with pytest.raises(ValueError) as err:
        p.vad = wrong_config
    assert 'Unknown parameters given for VAD config' in str(err.value)


def test_load_save(tmpdir):
    num_gauss = 8
    dim = 13
    p = DiagUbmProcessor(num_gauss)
    with pytest.raises(TypeError) as err:
        p.save(str(tmpdir.join('foo.dubm')))
    assert 'GMM not initialized' in str(err.value)

    p.gmm = kaldi.gmm.DiagGmm(num_gauss, dim)
    assert isinstance(p.gmm, kaldi.gmm.DiagGmm)

    random_means = np.random.rand(num_gauss, dim)
    random_vars = np.random.rand(num_gauss, dim)
    p.gmm.set_means(kaldi.matrix.Matrix(random_means))
    p.gmm.set_inv_vars(kaldi.matrix.Matrix(1/random_vars))

    p.save(str(tmpdir.join('foo.dubm')))
    p = DiagUbmProcessor.load(str(tmpdir.join('foo.dubm')))

    assert isinstance(p.gmm, kaldi.gmm.DiagGmm)
    assert p.gmm.get_means().numpy() == pytest.approx(random_means, abs=1e-6)
    assert p.gmm.get_vars().numpy() == pytest.approx(random_vars, abs=1e-6)

    with pytest.raises(OSError) as err:
        p.save(str(tmpdir.join('foo.dubm')))
    assert 'file already exists' in str(err.value)

    os.remove(str(tmpdir.join('foo.dubm')))
    with pytest.raises(OSError) as err:
        p = DiagUbmProcessor.load(str(tmpdir.join('foo.dubm')))
    assert 'file not found' in str(err.value)


def test_initialize(features_collection):
    ubm = DiagUbmProcessor(2048)
    with pytest.raises(ValueError) as err:
        ubm.initialize_gmm(features_collection)
    assert 'Too few frames to train on' in str(err.value)

    f1 = Features(np.random.random((10, 2)), np.ones((10,)))
    f2 = Features(np.random.random((10, 4)), np.ones((10,)))
    fc = FeaturesCollection(f1=f1, f2=f2)

    ubm = DiagUbmProcessor(2, num_iters_init=4, num_iters=1)
    with pytest.raises(ValueError) as err:
        ubm.initialize_gmm(fc)
    assert 'Features have unconsistent dims' in str(err.value)

    fc = FeaturesCollection(f1=Features(np.zeros((10, 2)), np.ones((10,))))
    with pytest.raises(ValueError) as err:
        ubm.initialize_gmm(fc)
    assert 'Features do not have positive variance' in str(err.value)

    dim = features_collection[list(features_collection.keys())[0]].ndims
    ubm.initialize_gmm(features_collection)
    assert isinstance(ubm.gmm, kaldi.gmm.DiagGmm)
    assert ubm.num_gauss == 2
    assert ubm.gmm.num_gauss() == ubm.num_gauss
    assert ubm.gmm.get_means().shape == (ubm.num_gauss, dim)
    ubm.gmm.gconsts()


def test_gaussian_selection():
    ubm = DiagUbmProcessor(8, num_iters_init=1, num_iters=1)
    fc = FeaturesCollection(f1=Features(
        np.random.random((50, 2)), np.arange(50)))

    with pytest.raises(TypeError) as err:
        ubm.gaussian_selection(fc)
    assert 'GMM not initialized' in str(err.value)

    ubm.initialize_gmm(fc)
    ubm.gaussian_selection(fc)
    del ubm.selection['f1']
    with pytest.raises(ValueError) as err:
        ubm.gaussian_selection(fc)
    assert 'No gselect information for utterance' in str(err.value)

    ubm.selection = None
    ubm.gaussian_selection(fc)
    ubm.selection['f1'] = [ubm.selection['f1'][0]]
    with pytest.raises(ValueError) as err:
        ubm.gaussian_selection(fc)
    assert 'Input gselect utterance f1 has wrong size' in str(err.value)

    ubm.selection = None
    ubm.gaussian_selection(fc)
    ubm.gaussian_selection(fc)

    assert ubm.selection.keys() == fc.keys()
    assert len(ubm.selection['f1']) == fc['f1'].nframes
    assert len(ubm.selection['f1'][0]) == ubm.num_gauss
    for gselect in ubm.selection['f1']:
        assert sorted(gselect) == list(range(ubm.num_gauss))


def test_gaussian_selection_to_post():
    ubm = DiagUbmProcessor(8, num_iters_init=1, num_iters=1)
    fc = FeaturesCollection(f1=Features(
        np.random.random((50, 2)), np.arange(50)))
    ubm.initialize_gmm(fc)
    with pytest.raises(ValueError) as err:
        ubm.gaussian_selection_to_post(fc)
    assert 'Gaussian selection has not been done' in str(err.value)

    ubm.selection = {}
    with pytest.raises(ValueError) as err:
        ubm.gaussian_selection_to_post(fc)
    assert 'No gselect information for utterance f1' in str(err.value)

    ubm.selection = None
    ubm.gaussian_selection(fc)
    ubm.selection['f1'] = [ubm.selection['f1'][0]]
    with pytest.raises(ValueError) as err:
        ubm.gaussian_selection_to_post(fc)
    assert 'Input gselect utterance f1 has wrong size' in str(err.value)

    ubm.selection = None
    ubm.gaussian_selection(fc)
    posteriors = ubm.gaussian_selection_to_post(
        fc, min_post=10)
    assert posteriors.keys() == fc.keys()
    assert len(posteriors['f1']) == fc['f1'].nframes
    for frame in range(fc['f1'].nframes):
        gaussians, loglikes = zip(*posteriors['f1'][frame])
        assert len(gaussians) == len(set(gaussians))
        assert set(gaussians) <= set(ubm.selection['f1'][frame])
        assert sum(loglikes) == pytest.approx(1, abs=1e-4)


def test_accumulate(features_collection):
    ubm = DiagUbmProcessor(2, num_iters_init=1, num_iters=1)
    with pytest.raises(TypeError) as err:
        ubm.accumulate(features_collection)
    assert 'GMM not initialized' in str(err.value)

    ubm.initialize_gmm(features_collection)
    fc = FeaturesCollection(f1=Features(
        np.random.random((10, 2)), np.ones((10,))))

    with pytest.raises(ValueError) as err:
        ubm.accumulate(fc)
    assert 'Features from utterance f1 have wrong dims' in str(err.value)

    ubm.initialize_gmm(fc)
    weights = {'f2': np.random.rand(10)}
    with pytest.raises(ValueError) as err:
        ubm.accumulate(fc, weights)
    assert 'Keys differ between weights and features' in str(err.value)

    weights = {'f1': np.array([0.5])}
    with pytest.raises(ValueError) as err:
        ubm.accumulate(fc, weights)
    assert 'Wrong size for weights' in str(err.value)

    weights = {'f1': np.random.rand(10)}
    gmm_accs = ubm.accumulate(fc, weights)
    assert isinstance(gmm_accs, kaldi.gmm.AccumDiagGmm)


def test_estimate():
    ubm = DiagUbmProcessor(4, num_iters_init=1, num_iters=1)
    fc = FeaturesCollection(f1=Features(
        np.random.random((20, 2)), np.arange(20)))
    ubm.initialize_gmm(fc)
    gmm_accs = ubm.accumulate(fc)

    ubm = DiagUbmProcessor(4, num_iters_init=1, num_iters=1)
    with pytest.raises(TypeError) as err:
        ubm.estimate(gmm_accs)
    assert 'GMM not initialized' in str(err.value)

    ubm.initialize_gmm(fc)
    with pytest.raises(ValueError) as err:
        ubm.estimate(gmm_accs, mixup=2)

    message = 'Mixup parameter must be greater than the number of gaussians'
    assert message in str(err.value)
    ubm.estimate(gmm_accs, mixup=8)


def test_process(wav_file, wav_file_float32, wav_file_8k):
    utterances = [
        ('u1', wav_file, 's1', 0, 1),
        ('u2', wav_file_float32, 's2', 1, 1.2),
        ('u3', wav_file_8k, 's1', 1, 3)]

    config = {'num_iters_init': 1, 'num_iters': 1, 'num_frames': 100,
              'vad': {'energy_threshold': 0}}
    ubm = DiagUbmProcessor(2, **config)
    ubm.process(utterances)
    config['features'] = pipeline.get_default_config(
        'mfcc', with_pitch=False, with_cmvn=False,
        with_sliding_window_cmvn=False, with_delta=False)
    ubm = DiagUbmProcessor(2, **config)
    ubm.process(utterances)
