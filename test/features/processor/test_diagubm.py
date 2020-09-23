import pytest
import os
import numpy as np

from shennong.features.postprocessor.vad import VadPostProcessor
from shennong.features.processor.diagubm import DiagUbmProcessor
import kaldi.gmm
import kaldi.matrix


def test_params():
    assert len(DiagUbmProcessor(1).get_params()) == 13

    params = {'num_gauss': 16, 'num_iters': 3, 'initial_gauss_proportion': 0.7}
    p = DiagUbmProcessor(**params)

    params_out = p.get_params()
    assert len(params_out) == 13
    for k, v in params.items():
        assert params_out[k] == v
    assert p.get_params() == params_out

    p = DiagUbmProcessor(1)
    p.set_params(**params_out)
    params_out == p.get_params()
    assert len(params_out) == 13
    for k, v in params.items():
        assert params_out[k] == v
    assert p.get_params() == params_out

    with pytest.raises(TypeError) as err:
        p = DiagUbmProcessor(1, extract_config=0)
    assert 'Features configuration must be a dict' in str(err.value)

    wrong_config = DiagUbmProcessor(1).get_params()['extract_config']
    del wrong_config['mfcc']
    with pytest.raises(ValueError) as err:
        p.extract_config = wrong_config
    assert 'Need mfcc features to train UBM-GMM' in str(err.value)

    with pytest.raises(TypeError) as err:
        p = DiagUbmProcessor(1, vad_config=0)
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
    with pytest.raises(ValueError) as err:
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

    with pytest.raises(ValueError) as err:
        p.save(str(tmpdir.join('foo.dubm')))
    assert 'file already exists' in str(err.value)

    os.remove(str(tmpdir.join('foo.dubm')))
    with pytest.raises(ValueError) as err:
        p = DiagUbmProcessor.load(str(tmpdir.join('foo.dubm')))
    assert 'file not found' in str(err.value)
