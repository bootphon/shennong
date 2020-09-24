import pytest
import os
import numpy as np

from shennong.features.processor.vtln import VtlnProcessor
from shennong.features.processor.diagubm import DiagUbmProcessor
from shennong.features.pipeline import extract_features, get_default_config
import kaldi.transform.lvtln
import kaldi.matrix


def test_params():
    assert len(VtlnProcessor().get_params()) == 13

    params = {'by_speaker': False, 'num_iters': 3, 'warp_step': 0.5}
    p = VtlnProcessor(**params)

    params_out = p.get_params()
    assert len(params_out) == 13
    for k, v in params.items():
        assert params_out[k] == v
    assert p.get_params() == params_out

    p = VtlnProcessor()
    p.set_params(**params_out)
    params_out == p.get_params()
    assert len(params_out) == 13
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
def test_load_save(tmpdir, chosen_class):
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


def test_train(wav_file, wav_file_float32, wav_file_8k):
    utterances = [
        ('u1', wav_file, 's1', 0, 1),
        ('u2', wav_file_float32, 's2', 1, 1.2),
        ('u3', wav_file_8k, 's1', 1, 3)]
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
    ubm.process(utterances)
    vtln.process(utterances, ubm=ubm)

    ubm.process(utterances)
    config['vtln']['by_speaker'] = False
    extract_features(config, utterances)
