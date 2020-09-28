import pytest
import os
import numpy as np

from shennong.features.processor.vtln import VtlnProcessor
from shennong.features.processor.diagubm import DiagUbmProcessor
from shennong.features.features import Features, FeaturesCollection
from shennong.features.pipeline import extract_features, get_default_config
import kaldi.transform.lvtln
import kaldi.matrix


def test_params():
    assert len(VtlnProcessor().get_params()) == 12

    params = {'by_speaker': False, 'num_iters': 3, 'warp_step': 0.5}
    p = VtlnProcessor(**params)

    params_out = p.get_params()
    assert len(params_out) == 12
    for k, v in params.items():
        assert params_out[k] == v
    assert p.get_params() == params_out

    p = VtlnProcessor()
    p.set_params(**params_out)
    params_out == p.get_params()
    assert len(params_out) == 12
    for k, v in params.items():
        assert params_out[k] == v
    assert p.get_params() == params_out

    with pytest.raises(ValueError) as err:
        p = VtlnProcessor(norm_type='wrong')
    assert 'Invalid norm type' in str(err.value)

    wrong_config = VtlnProcessor().get_params()['extract_config']
    del wrong_config['mfcc']
    with pytest.raises(ValueError) as err:
        p.extract_config = wrong_config
    assert 'Need mfcc features to train VTLN model' in str(err.value)

    with pytest.raises(TypeError) as err:
        p = VtlnProcessor(extract_config=0)
    assert 'Features extraction configuration must be a dict' in str(err.value)

    with pytest.raises(TypeError) as err:
        p = VtlnProcessor(1, ubm_config=0)
    assert 'UBM configuration must be a dict' in str(err.value)

    wrong_config = DiagUbmProcessor(2).get_params()
    wrong_config['wrong'] = 0
    with pytest.raises(ValueError) as err:
        p.ubm_config = wrong_config
    assert 'Unknown parameters given for UBM config' in str(err.value)


@pytest.mark.parametrize('chosen_class', [0, 1])
def test_load_save_model(tmpdir, chosen_class):
    dim = 4
    num_classes = 2
    default_class = 1
    p = VtlnProcessor()
    with pytest.raises(TypeError) as err:
        p.save(str(tmpdir.join('foo.vtln')))
    assert 'VTLN not initialized' in str(err.value)

    p.lvtln = kaldi.transform.lvtln.LinearVtln.new(
        dim, num_classes, default_class)
    assert isinstance(p.lvtln, kaldi.transform.lvtln.LinearVtln)

    random_transform = np.random.rand(dim, dim)
    random_warp = np.random.rand()
    p.lvtln.set_transform(chosen_class, kaldi.matrix.Matrix(random_transform))
    p.lvtln.set_warp(chosen_class, random_warp)

    p.save(str(tmpdir.join('foo.vtln')))
    p = VtlnProcessor.load(str(tmpdir.join('foo.vtln')))

    assert isinstance(p.lvtln, kaldi.transform.lvtln.LinearVtln)
    new_transform = kaldi.matrix.Matrix(dim, dim)
    p.lvtln.get_transform(chosen_class, new_transform)
    assert new_transform.numpy() == pytest.approx(random_transform, abs=1e-6)
    assert p.lvtln.get_warp(chosen_class) == pytest.approx(
        random_warp, abs=1e-6)

    with pytest.raises(OSError) as err:
        p.save(str(tmpdir.join('foo.vtln')))
    assert 'file already exists' in str(err.value)

    os.remove(str(tmpdir.join('foo.vtln')))
    with pytest.raises(OSError) as err:
        p = VtlnProcessor.load(str(tmpdir.join('foo.vtln')))
    assert 'file not found' in str(err.value)


def test_load_save_warps(tmpdir):
    p = VtlnProcessor()
    with pytest.raises(TypeError) as err:
        p.save_warps(str(tmpdir.join('warps.yml')))
    assert 'Warps not computed' in str(err.value)

    p.warps = {'u1': 1.05}
    p.save_warps(str(tmpdir.join('warps.yml')))
    p.warps = VtlnProcessor.load_warps(str(tmpdir.join('warps.yml')))

    assert isinstance(p.warps, dict)
    assert list(p.warps.keys()) == ['u1']
    assert p.warps['u1'] == 1.05

    with pytest.raises(OSError) as err:
        p.save_warps(str(tmpdir.join('warps.yml')))
    assert 'file already exists' in str(err.value)

    os.remove(str(tmpdir.join('warps.yml')))
    with pytest.raises(OSError) as err:
        p.warps = VtlnProcessor.load_warps(str(tmpdir.join('warps.yml')))
    assert 'file not found' in str(err.value)


def test_compute_mapping_transform():
    vtln = VtlnProcessor()
    fu = FeaturesCollection(f1=Features(
        np.random.random((20, 2)), np.arange(20)))
    with pytest.raises(TypeError) as err:
        vtln.compute_mapping_transform(fu, fu, 0, 1)
    assert 'VTLN not initialized' in str(err.value)

    vtln.lvtln = kaldi.transform.lvtln.LinearVtln.new(2, 2, 0)
    fu = FeaturesCollection(f1=Features(
        np.random.random((20, 2)), np.arange(20)))
    with pytest.raises(ValueError) as err:
        vtln.compute_mapping_transform(fu, FeaturesCollection(), 0, 1)
    assert 'No transformed features for key f1' in str(err.value)

    ft = FeaturesCollection(f1=Features(
        np.random.random((10, 2)), np.arange(10)))
    with pytest.raises(ValueError) as err:
        vtln.compute_mapping_transform(fu, ft, 0, 1)
    assert 'Number of rows and/or columns differs' in str(err.value)

    ft = FeaturesCollection(f1=Features(
        np.random.random((20, 2)), np.arange(20)))
    with pytest.raises(ValueError) as err:
        vtln.compute_mapping_transform(fu, ft, 0, 1, weights={})
    assert 'No weights for utterance f1' in str(err.value)

    vtln.compute_mapping_transform(
        fu, ft, 0, 1, weights={'f1': np.random.rand(20)})


@pytest.mark.parametrize('utt2speak', [None, {'f1': '1'}])
def test_estimate(utt2speak):
    vtln = VtlnProcessor()
    ubm = DiagUbmProcessor(2, num_iters_init=1, num_iters=1,
                           vad_config={'energy_threshold': 0})
    fc = FeaturesCollection(f1=Features(
        np.random.random((20, 2)), np.arange(20)))
    ubm.initialize_gmm(fc)
    ubm.gaussian_selection(fc)
    posteriors = ubm.gaussian_selection_to_post(fc)

    with pytest.raises(TypeError) as err:
        vtln.estimate(ubm, fc, posteriors, utt2speak)
    assert 'VTLN not initialized' in str(err.value)
    vtln.lvtln = kaldi.transform.lvtln.LinearVtln.new(2, 2, 0)

    with pytest.raises(ValueError) as err:
        vtln.estimate(ubm, fc, {}, utt2speak)
    assert 'No posterior for utterance f1' in str(err.value)

    with pytest.raises(ValueError) as err:
        vtln.estimate(ubm, fc, {'f1': [[(0, 1)]]}, utt2speak)
    assert 'Posterior has wrong size' in str(err.value)


def test_train(wav_file, wav_file_float32, wav_file_8k):
    utterances = [
        ('s1a', wav_file, 's1', 0, 1),
        ('s2a', wav_file_float32, 's2', 1, 1.2),
        ('s1b', wav_file_8k, 's1', 1, 3)]
    ubm_config = DiagUbmProcessor(8).get_params()
    ubm_config['vad_config']['energy_threshold'] = 0
    ubm_config['num_iters_init'] = 1
    ubm_config['num_iters'] = 1

    config = get_default_config('mfcc', with_vtln=True)
    config['cmvn']['with_vad'] = False
    config['vtln']['ubm_config'] = ubm_config
    config['vtln']['min_warp'] = 0.99
    config['vtln']['max_warp'] = 1
    config['vtln']['num_iters'] = 1

    vtln = VtlnProcessor(**config['vtln'])
    ubm = DiagUbmProcessor(2, num_iters_init=1, num_iters=1,
                           num_frames=100, vad_config={
                               'energy_threshold': 0})
    with pytest.raises(ValueError) as err:
        vtln.process(utterances, ubm=ubm)
    assert 'Given UBM-GMM has not been trained' in str(err.value)

    ubm.process(utterances)
    vtln.process(utterances, ubm=ubm, utt2speak={
                 's1a': 's1', 's1b': 's1', 's2a': 's2'})
    ubm.process(utterances)
    config['vtln']['by_speaker'] = False
    del config['vtln']['extract_config']['sliding_window_cmvn']
    extract_features(config, utterances)
