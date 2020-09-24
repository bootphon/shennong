import pytest
import os
import numpy as np

from shennong.features.postprocessor.vad import VadPostProcessor
from shennong.features.processor.diagubm import DiagUbmProcessor
from shennong.features.features import Features, FeaturesCollection
import kaldi.gmm
import kaldi.matrix


def test_params():
    assert len(DiagUbmProcessor(2).get_params()) == 13

    params = {'num_gauss': 16, 'num_iters': 3, 'initial_gauss_proportion': 0.7}
    p = DiagUbmProcessor(**params)

    params_out = p.get_params()
    assert len(params_out) == 13
    for k, v in params.items():
        assert params_out[k] == v
    assert p.get_params() == params_out

    p = DiagUbmProcessor(2)
    p.set_params(**params_out)
    params_out == p.get_params()
    assert len(params_out) == 13
    for k, v in params.items():
        assert params_out[k] == v
    assert p.get_params() == params_out

    with pytest.raises(ValueError) as err:
        p = DiagUbmProcessor(1)
    assert 'Number of gaussians must be at least 2' in str(err.value)

    with pytest.raises(TypeError) as err:
        p = DiagUbmProcessor(2, extract_config=0)
    assert 'Features configuration must be a dict' in str(err.value)

    wrong_config = DiagUbmProcessor(2).get_params()['extract_config']
    del wrong_config['mfcc']
    with pytest.raises(ValueError) as err:
        p.extract_config = wrong_config
    assert 'Need mfcc features to train UBM-GMM' in str(err.value)

    with pytest.raises(TypeError) as err:
        p = DiagUbmProcessor(2, vad_config=0)
    assert 'VAD configuration must be a dict' in str(err.value)

    wrong_config = VadPostProcessor().get_params()
    wrong_config['wrong'] = 0
    with pytest.raises(ValueError) as err:
        p.vad_config = wrong_config
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

    ubm = DiagUbmProcessor(2, num_iters_init=4, num_iters=1,
                           num_frames=10)
    with pytest.raises(ValueError) as err:
        ubm.initialize_gmm(fc)
    assert 'Features have unconsistent dims' in str(err.value)

    fc = FeaturesCollection(f1=Features(np.zeros((10, 2)), np.ones((10,))))
    with pytest.raises(ValueError) as err:
        ubm.initialize_gmm(fc)
    assert 'Features do not have positive variance' in str(err.value)

    ubm.initialize_gmm(features_collection)


def test_gaussian_selection():
    ubm = DiagUbmProcessor(2, num_iters_init=1, num_iters=1,
                           num_frames=100)
    fc = FeaturesCollection(f1=Features(
        np.random.random((10, 2)), np.ones((10,))))

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


def test_gaussian_selection_to_post():
    ubm = DiagUbmProcessor(2, num_iters_init=1, num_iters=1,
                           num_frames=100)
    fc = FeaturesCollection(f1=Features(
        np.random.random((10, 2)), np.ones((10,))))
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
    ubm.gaussian_selection_to_post(fc, min_post=10)


def test_accumulate(features_collection):
    ubm = DiagUbmProcessor(2, num_iters_init=1, num_iters=1,
                           num_frames=100)
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
    ubm.accumulate(fc, weights)


def test_estimate():
    ubm = DiagUbmProcessor(2, num_iters_init=1, num_iters=1,
                           num_frames=100)
    fc = FeaturesCollection(f1=Features(
        np.random.random((10, 2)), np.ones((10,))))
    ubm.initialize_gmm(fc)
    gmm_accs = ubm.accumulate(fc)

    ubm = DiagUbmProcessor(2, num_iters_init=1, num_iters=1,
                           num_frames=100)
    with pytest.raises(TypeError) as err:
        ubm.estimate(gmm_accs)
    assert 'GMM not initialized' in str(err.value)

    ubm.initialize_gmm(fc)
    ubm.estimate(gmm_accs, mixup=8)


def test_process(wav_file, wav_file_float32, wav_file_8k):
    utterances = [
        ('u1', wav_file, 's1', 0, 1),
        ('u2', wav_file_float32, 's2', 1, 1.2),
        ('u3', wav_file_8k, 's1', 1, 3)]

    ubm = DiagUbmProcessor(2, num_iters_init=1, num_iters=1,
                           num_frames=100, vad_config={
                               'energy_threshold': 0})
    ubm.process(utterances)
