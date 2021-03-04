import pytest
import os
import numpy as np
import kaldi.transform.lvtln
import kaldi.matrix

from shennong.features.processor.vtln import VtlnProcessor
from shennong.features.processor.ubm import DiagUbmProcessor
from shennong.features.features import Features, FeaturesCollection


def test_params():
    assert len(VtlnProcessor().get_params()) == 10

    params = {'by_speaker': False, 'num_iters': 3, 'warp_step': 0.5}
    p = VtlnProcessor(**params)

    params_out = p.get_params()
    assert len(params_out) == 10
    for k, v in params.items():
        assert params_out[k] == v
    assert p.get_params() == params_out

    p = VtlnProcessor()
    p.set_params(**params_out)
    assert params_out == p.get_params()
    assert len(params_out) == 10
    for k, v in params.items():
        assert params_out[k] == v
    assert p.get_params() == params_out

    with pytest.raises(ValueError) as err:
        p = VtlnProcessor(norm_type='wrong')
    assert 'Invalid norm type' in str(err.value)

    wrong_config = VtlnProcessor().get_params()['features']
    del wrong_config['mfcc']
    with pytest.raises(ValueError) as err:
        p.features = wrong_config
    assert 'Need mfcc features to train VTLN model' in str(err.value)

    with pytest.raises(TypeError) as err:
        p = VtlnProcessor(features=0)
    assert 'Features extraction configuration must be a dict' in str(err.value)

    with pytest.raises(TypeError) as err:
        p = VtlnProcessor(1, ubm=0)
    assert 'UBM configuration must be a dict' in str(err.value)

    wrong_config = DiagUbmProcessor(2).get_params()
    wrong_config['wrong'] = 0
    with pytest.raises(ValueError) as err:
        p.ubm = wrong_config
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
    p = VtlnProcessor()
    fu = FeaturesCollection(f1=Features(
        np.random.random((20, 2)), np.arange(20)))
    with pytest.raises(TypeError) as err:
        p.compute_mapping_transform(fu, fu, 0, 1)
    assert 'VTLN not initialized' in str(err.value)

    p.lvtln = kaldi.transform.lvtln.LinearVtln.new(2, 2, 0)
    fu = FeaturesCollection(f1=Features(
        np.random.random((20, 2)), np.arange(20)))
    with pytest.raises(ValueError) as err:
        p.compute_mapping_transform(fu, FeaturesCollection(), 0, 1)
    assert 'No transformed features for key f1' in str(err.value)

    ft = FeaturesCollection(f1=Features(
        np.random.random((10, 2)), np.arange(10)))
    with pytest.raises(ValueError) as err:
        p.compute_mapping_transform(fu, ft, 0, 1)
    assert 'Number of rows and/or columns differs' in str(err.value)

    ft = FeaturesCollection(f1=Features(
        np.random.random((20, 2)), np.arange(20)))
    with pytest.raises(ValueError) as err:
        p.compute_mapping_transform(fu, ft, 0, 1, weights={})
    assert 'No weights for utterance f1' in str(err.value)

    p.compute_mapping_transform(
        fu, ft, 0, 1.15, weights={'f1': np.random.rand(20)})
    t = kaldi.matrix.Matrix(2, 2)
    p.lvtln.get_transform(0, t)
    assert not np.array_equal(t.numpy(), np.eye(2))
    p.lvtln.get_transform(1, t)
    assert np.array_equal(t.numpy(), np.eye(2))
    assert p.lvtln.get_warp(0) == pytest.approx(1.15, abs=1e-6)
    assert p.lvtln.get_warp(1) == pytest.approx(1, abs=1e-6)


@pytest.mark.parametrize('by_speaker', [True, False])
def test_estimate(by_speaker):
    utt2speak = {'f1': '1'} if by_speaker else None
    p = VtlnProcessor()
    dim = 2
    ubm = DiagUbmProcessor(4, num_iters_init=1, num_iters=1,
                           vad={'energy_threshold': 0})
    fc = FeaturesCollection(f1=Features(
        np.random.random((20, dim)), np.arange(20)))
    ubm.initialize_gmm(fc)
    ubm.gaussian_selection(fc)
    posteriors = ubm.gaussian_selection_to_post(fc)

    with pytest.raises(TypeError) as err:
        p.estimate(ubm, fc, posteriors, utt2speak)
    assert 'VTLN not initialized' in str(err.value)
    p.lvtln = kaldi.transform.lvtln.LinearVtln.new(dim, 2, 0)

    with pytest.raises(ValueError) as err:
        p.estimate(ubm, fc, {}, utt2speak)
    assert 'No posterior for utterance f1' in str(err.value)

    with pytest.raises(ValueError) as err:
        p.estimate(ubm, fc, {'f1': [[(0, 1)]]}, utt2speak)
    assert 'Posterior has wrong size' in str(err.value)

    transforms, warps = p.estimate(ubm, fc, posteriors, utt2speak)
    assert isinstance(transforms, dict)
    key = '1' if by_speaker else 'f1'
    assert list(transforms.keys()) == [key]
    assert isinstance(transforms[key], kaldi.matrix.Matrix)
    assert transforms[key].shape == (dim, dim+1)
    assert isinstance(warps, dict)
    assert list(warps.keys()) == [key]
    assert isinstance(warps[key], float)
    assert p.min_warp <= warps[key] <= p.max_warp


def test_process(wav_file, wav_file_float32, wav_file_8k):

    ubm_config = DiagUbmProcessor(8).get_params()
    ubm_config['vad']['energy_threshold'] = 0
    ubm_config['num_iters_init'] = 1
    ubm_config['num_iters'] = 1

    vtln_config = {}
    vtln_config['ubm'] = ubm_config
    vtln_config['min_warp'] = 0.99
    vtln_config['max_warp'] = 1
    vtln_config['num_iters'] = 1

    vtln = VtlnProcessor(**vtln_config)
    ubm = DiagUbmProcessor(**ubm_config)

    with pytest.raises(TypeError) as err:
        vtln.process({'s1a': (wav_file, 's1', 0, 1)}, ubm=ubm)
    assert 'Invalid utterances format' in str(err.value)

    utterances = [('s1a', wav_file, 's1', 0, 1), ('s2a', wav_file_float32)]
    with pytest.raises(ValueError) as err:
        vtln.process(utterances, ubm=ubm)
    assert 'the wavs index is not homogeneous' in str(err.value)

    utterances = [('s1a', wav_file), ('s2a', wav_file_float32)]
    with pytest.raises(ValueError) as err:
        vtln.process(utterances, ubm=ubm)
    assert 'Requested speaker-adapted VTLN' in str(err.value)

    utterances = [
        ('s1a', wav_file, 's1', 0, 1),
        ('s2a', wav_file_float32, 's2', 1, 1.2),
        ('s1b', wav_file_8k, 's1', 1, 3)]

    with pytest.raises(ValueError) as err:
        vtln.process(utterances, ubm=ubm)
    assert 'Given UBM-GMM has not been trained' in str(err.value)

    vtln.min_warp = 10
    with pytest.raises(ValueError) as err:
        vtln.process(utterances, ubm=ubm)
    assert 'Min warp > max warp' in str(err.value)
    vtln.min_warp = 0.99

    ubm.process(utterances)
    warps = vtln.process(utterances, ubm=ubm)
    assert isinstance(warps, dict)
    assert set(warps.keys()) == set(['s1a', 's1b', 's2a'])
    assert warps['s1a'] == warps['s1b']

    vtln.by_speaker = False
    vtln.features.pop('sliding_window_cmvn', None)
    warps = vtln.process(utterances)
    assert isinstance(warps, dict)
    assert set(warps.keys()) == set(['s1a', 's1b', 's2a'])
