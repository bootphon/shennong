"""Test of the module shennong.features.pipeline"""

import numpy as np
import os
import pytest
import yaml

import shennong.features.pipeline as pipeline
import shennong.utils as utils
from shennong.audio import Audio


@pytest.fixture(scope='session')
def wavs_index(wav_file):
    return [('utt1', wav_file, 'speaker1')]


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


@pytest.mark.parametrize('kind', ['dict', 'file', 'str'])
def test_config_format(wavs_index, capsys, tmpdir, kind):
    config = pipeline.get_default_config('mfcc', to_yaml=kind != 'dict')

    if kind == 'file':
        tempfile = str(tmpdir.join('foo'))
        open(tempfile, 'w').write(config)
        config = tempfile

    if kind == 'str':
        config2 = 'a:\nb\n'
        with pytest.raises(ValueError) as err:
            pipeline._init_config(config2)
        assert 'error in configuration' in str(err)

    parsed = pipeline._init_config(config, log=utils.get_logger(level='info'))
    output = capsys.readouterr().err
    for word in ('mfcc', 'pitch', 'cmvn', 'delta'):
        assert word in output
        assert word in parsed


def test_config_bad(wavs_index):
    with pytest.raises(ValueError) as err:
        pipeline.get_default_config('bad')
    assert 'invalid features "bad"' in str(err)

    config = pipeline.get_default_config('mfcc')
    del config['mfcc']
    with pytest.raises(ValueError) as err:
        pipeline.extract_features(config, wavs_index)
    assert 'the configuration does not define any features' in str(err)

    config = pipeline.get_default_config('mfcc')
    config['plp'] = config['mfcc']
    with pytest.raises(ValueError) as err:
        pipeline.extract_features(config, wavs_index)
    assert 'more than one features extraction processor' in str(err)

    config = pipeline.get_default_config('mfcc')
    config['invalid'] = config['mfcc']
    with pytest.raises(ValueError) as err:
        pipeline.extract_features(config, wavs_index)
    assert 'invalid keys in configuration' in str(err)

    config = pipeline.get_default_config('mfcc')
    del config['cmvn']['with_vad']
    parsed = pipeline._init_config(config)
    assert 'cmvn' in parsed
    assert 'with_vad' not in parsed['cmvn']
    assert 'vad' not in parsed

    config = pipeline.get_default_config('mfcc')
    del config['pitch']
    with pytest.raises(ValueError) as err:
        pipeline.extract_features(config, wavs_index)
    assert 'configuration defines pitch_post but not pitch' in str(err)

    config = pipeline.get_default_config('mfcc')
    del config['pitch_post']
    with pytest.raises(ValueError) as err:
        pipeline.extract_features(config, wavs_index)
    assert 'configuration defines pitch but not pitch_post' in str(err)

    config = pipeline.get_default_config('mfcc')
    del config['cmvn']['by_speaker']
    c = pipeline._init_config(config)
    assert not c['cmvn']['by_speaker']


def test_check_speakers(wavs_index, capsys):
    log = utils.get_logger(level='info')

    config = pipeline.get_default_config('mfcc')
    with pytest.raises(ValueError) as err:
        pipeline.extract_features(config, [(wavs_index[0][1], )], log=log)
    assert 'no speaker information provided' in str(err)

    config = pipeline.get_default_config('mfcc', with_cmvn=False)
    pipeline.extract_features(config, wavs_index, log=log)
    log_out = capsys.readouterr()
    assert '(CMVN disabled)' in log_out.err

    config = pipeline.get_default_config('mfcc', with_cmvn=True)
    config['cmvn']['by_speaker'] = False
    pipeline.extract_features(config, wavs_index, log=log)
    assert '(CMVN by speaker disabled)' in capsys.readouterr().err


def test_check_environment(capsys):
    if 'OMP_NUM_THREADS' in os.environ:
        del os.environ['OMP_NUM_THREADS']
    pipeline._check_environment(2, log=utils.get_logger())
    out = capsys.readouterr().err
    assert 'working on 2 threads but implicit parallelism is active' in out


def test_wavs_bad(wav_file, wav_file_8k, tmpdir, capsys):
    fun = pipeline._init_wavs

    # ensure we catch basic errors
    with pytest.raises(ValueError) as err:
        fun([('a'), ('a', 'b')])
    assert 'entries have different lengths' in str(err)

    with pytest.raises(ValueError) as err:
        fun([('a', 'b', 'c', 'd', 'e', 'g')])
    assert 'unknown format for wavs index' in str(err)

    with pytest.raises(ValueError) as err:
        fun([('a'), ('a')])
    assert 'duplicates found in wavs index' in str(err)

    with pytest.raises(ValueError) as err:
        fun([('/foo/bar/a')])
    assert 'the following wav files are not found' in str(err)

    # build a stereo file and make sure it is not supported by the
    # pipeline
    audio = Audio.load(wav_file)
    stereo = Audio(
        np.asarray((audio.data, audio.data)).T, sample_rate=audio.sample_rate)
    assert stereo.nchannels == 2
    wav_file_2 = str(tmpdir.join('stereo.wav'))
    stereo.save(wav_file_2)
    with pytest.raises(ValueError) as err:
        fun([(wav_file_2)])
    assert 'all wav files are not mono' in str(err)

    # ensure we catch differences in sample rates
    w = [(wav_file, ), (wav_file_8k, )]
    out = fun(w)
    err = capsys.readouterr().err
    assert 'several sample rates found in wav files' in err
    assert sorted(out.keys()) == ['utt_1', 'utt_2']

    # make sure timestamps are ordered
    with pytest.raises(ValueError) as err:
        fun([('1', wav_file, 1, 0)])
    assert 'timestamps are not in increasing order for' in str(err)


def test_processor_bad():
    get = pipeline._get_processor
    with pytest.raises(ValueError) as err:
        get('bad')
    assert 'invalid processor "' in str(err)

    with pytest.raises(ValueError) as err:
        get(0)
    assert 'invalid processor "' in str(err)


@pytest.mark.parametrize('features', pipeline.valid_features())
def test_extract_features(wavs_index, features):
    config = pipeline.get_default_config(
        features, with_cmvn=False, with_pitch=False)
    feats = pipeline.extract_features(config, wavs_index)
    feat1 = feats[wavs_index[0][0]]
    assert feat1.is_valid()
    assert feat1.shape[0] == 140

    config = pipeline.get_default_config(
        features, with_cmvn=False, with_pitch=True)
    feats = pipeline.extract_features(config, wavs_index)
    feat2 = feats[wavs_index[0][0]]
    assert feat2.is_valid()
    assert feat2.shape[0] == 140
    assert feat2.shape[1] == feat1.shape[1] + 3

    wavs_index = [('u1', wavs_index[0][1], 0, 1)]
    config = pipeline.get_default_config(
        features, with_cmvn=False, with_pitch=False)
    feats = pipeline.extract_features(config, wavs_index)
    feat3 = feats[wavs_index[0][0]]
    assert feat3.is_valid()
    assert feat3.shape[0] == 98
    assert feat3.shape[1] == feat1.shape[1]


# def test_extract_features_full(wav_file, wav_file_8k, capsys):
#     # difficult case with different sampling rates, speakers and segments
#     index = [
#         ('u1', wav_file, 's1', 0, 1),
#         ('u2', wav_file, 's2', 1, 1.2),
#         ('u3', wav_file_8k, 's1', 0, 3)]
#     config = pipeline.get_default_config('mfcc')

#     feats = pipeline.extract_features(config, index, log=utils.get_logger())

#     # ensure we have the expected log messages
#     messages = capsys.readouterr().err
#     assert 'WARNING - several sample rates found in wav files' in messages

#     for utt in ('u1', 'u2', 'u3'):
#         assert utt in feats
