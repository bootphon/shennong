import pytest

from shennong.features.postprocessor.vad import VadPostProcessor
from shennong.features.processor.diagubm import DiagUbmProcessor


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
    assert 'Unknown parameters given' in str(err.value)
