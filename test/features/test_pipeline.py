"""Test of the module shennong.features.pipeline"""

import pytest
import yaml

import shennong.features.pipeline as pipeline


def equal_dict(d1, d2):
    assert 'htk_compat' not in d1.keys()
    assert 'sample_rate' not in d1.keys()
    if not d1.keys() == d2.keys():
        return False
    for k, v in d1.items():
        if isinstance(v, str):
            if not v == d2[k]:
                return False
        elif isinstance(v, dict):
            if not equal_dict(v, d2[k]):
                return False
        else:
            if v != pytest.approx(d2[k]):
                return False
    return True


@pytest.mark.parametrize(
    'features', pipeline._valid_features)
def test_config_good(features):
    c1 = pipeline.get_default_config(features, to_yaml=False)
    c2 = pipeline.get_default_config(
        features, to_yaml=True, yaml_commented=False)
    c3 = pipeline.get_default_config(
        features, to_yaml=True, yaml_commented=True)
    assert features in c1.keys()
    assert '#' not in c2
    assert '#' in c3
    assert equal_dict(c1, yaml.load(c2, Loader=yaml.FullLoader))
    assert equal_dict(c1, yaml.load(c3, Loader=yaml.FullLoader))


def test_config_bad():
    with pytest.raises(ValueError) as err:
        pipeline.get_default_config('bad')
    assert 'invalid features "bad"' in str(err)
