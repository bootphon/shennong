"""Test of the module shennong.features.pipeline"""

import numpy as np
import os
import pytest
import yaml

import shennong.logger as logger
import shennong.pipeline as pipeline
from shennong.pipeline_manager import PipelineManager
from shennong import Audio, FeaturesCollection
from shennong.serializers import supported_extensions


@pytest.fixture(scope='session')
def utterances_index(wav_file):
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
    'features, with_vtln',
    [(f, v) for f in pipeline.valid_features()
     for v in (False, 'simple', 'full')])
def test_config_good(features, with_vtln):
    c1 = pipeline.get_default_config(
        features, to_yaml=False, with_vtln=with_vtln)
    c2 = pipeline.get_default_config(
        features, to_yaml=True, yaml_commented=False, with_vtln=with_vtln)
    c3 = pipeline.get_default_config(
        features, to_yaml=True, yaml_commented=True, with_vtln=with_vtln)
    assert features in c1.keys()
    assert '#' not in c2
    assert '#' in c3
    assert equal_dict(c1, yaml.load(c2, Loader=yaml.FullLoader))
    assert equal_dict(c1, yaml.load(c3, Loader=yaml.FullLoader))


@pytest.mark.parametrize('kind', ['dict', 'file', 'str'])
def test_config_format(utterances_index, capsys, tmpdir, kind):
    config = pipeline.get_default_config('mfcc', to_yaml=kind != 'dict')

    if kind == 'file':
        tempfile = str(tmpdir.join('foo'))
        open(tempfile, 'w').write(config)
        config = tempfile

    if kind == 'str':
        config2 = 'a:\nb\n'
        with pytest.raises(ValueError) as err:
            pipeline._init_config(config2)
        assert 'error in configuration' in str(err.value)

    parsed = pipeline._init_config(config, log=logger.get_logger(
        'pipeline', level='info'))
    output = capsys.readouterr().err
    for word in ('mfcc', 'pitch', 'cmvn', 'delta'):
        assert word in output
        assert word in parsed


def test_config_bad(utterances_index):
    with pytest.raises(ValueError) as err:
        pipeline.get_default_config('bad')
    assert 'invalid features "bad"' in str(err.value)

    config = pipeline.get_default_config('mfcc')
    del config['mfcc']
    with pytest.raises(ValueError) as err:
        pipeline.extract_features(config, utterances_index)
    assert 'the configuration does not define any features' in str(err.value)

    config = pipeline.get_default_config('mfcc')
    config['plp'] = config['mfcc']
    with pytest.raises(ValueError) as err:
        pipeline.extract_features(config, utterances_index)
    assert 'more than one features extraction processor' in str(err.value)

    config = pipeline.get_default_config('mfcc')
    config['invalid'] = config['mfcc']
    with pytest.raises(ValueError) as err:
        pipeline.extract_features(config, utterances_index)
    assert 'invalid keys in configuration' in str(err.value)

    with pytest.raises(ValueError) as err:
        pipeline.get_default_config('mfcc', with_vtln=True)
    assert 'must be "simple" or "full" but is "True"' in str(err.value)

    config = pipeline.get_default_config(
        'mfcc', with_pitch=True, with_crepe_pitch=True)
    with pytest.raises(ValueError) as err:
        pipeline.extract_features(config, utterances_index)
    assert 'both default pitch and CREPE pitch are requested' in str(err.value)

    config = pipeline.get_default_config('mfcc')
    del config['cmvn']['with_vad']
    parsed = pipeline._init_config(config)
    assert 'cmvn' in parsed
    assert parsed['cmvn']['with_vad']

    config = pipeline.get_default_config('mfcc')
    del config['cmvn']['by_speaker']
    c = pipeline._init_config(config)
    assert not c['cmvn']['by_speaker']

    config = pipeline.get_default_config('mfcc')
    del config['pitch']['postprocessing']
    c = pipeline._init_config(config)
    assert c['pitch']['postprocessing'] == {}

    config = pipeline.get_default_config(
        'mfcc', with_pitch=False, with_crepe_pitch=True)
    del config['crepe_pitch']['postprocessing']
    c = pipeline._init_config(config)
    assert c['crepe_pitch']['postprocessing'] == {}


def test_check_speakers(utterances_index, capsys):
    log = logger.get_logger('test', 'info')

    config = pipeline.get_default_config('mfcc')
    with pytest.raises(ValueError) as err:
        pipeline.extract_features(
            config, [(utterances_index[0][1], )], log=log)
    assert 'no speaker information provided' in str(err.value)

    capsys.readouterr()  # clean the buffer
    config = pipeline.get_default_config('mfcc', with_cmvn=False)
    pipeline.extract_features(config, utterances_index, log=log)
    log_out = capsys.readouterr()
    assert 'cmvn' not in log_out.err
    assert '(CMVN disabled)' in log_out.err

    config = pipeline.get_default_config('mfcc', with_cmvn=True)
    config['cmvn']['by_speaker'] = False
    pipeline.extract_features(config, utterances_index, log=log)
    log_out = capsys.readouterr().err
    assert 'cmvn by utterance' in log_out
    assert '(CMVN by speaker disabled)' in log_out


def test_check_environment(capsys):
    if 'OMP_NUM_THREADS' in os.environ:
        del os.environ['OMP_NUM_THREADS']
    pipeline._check_environment(2, log=logger.get_logger('test', 'info'))
    out = capsys.readouterr().err
    assert 'working on 2 threads but implicit parallelism is active' in out


def test_utts_bad(wav_file, wav_file_8k, tmpdir, capsys):
    fun = pipeline._init_utterances

    # ensure we catch basic errors
    with pytest.raises(ValueError) as err:
        fun([('a'), ('a', 'b')])
    assert 'entries have different lengths' in str(err.value)

    with pytest.raises(ValueError) as err:
        fun([('a', 'b', 'c', 'd', 'e', 'g')])
    assert 'unknown format for utterances index' in str(err.value)

    with pytest.raises(ValueError) as err:
        fun([('a'), ('a')])
    assert 'duplicates found in utterances index' in str(err.value)

    with pytest.raises(ValueError) as err:
        fun([('/foo/bar/a')])
    assert 'the following wav files are not found' in str(err.value)


def test_check_wavs_bad(wav_file, wav_file_8k, tmpdir, capsys):
    def fun(utts):
        c = pipeline._init_config(pipeline.get_default_config(
            'mfcc', with_cmvn=False))
        u = pipeline._init_utterances(utts, log=logger.get_logger('test', 'info'))
        PipelineManager(c, u, log=logger.get_logger('test', 'info'))
        return u

    # build a stereo file and make sure it is not supported by the
    # pipeline
    audio = Audio.load(wav_file)
    stereo = Audio(
        np.asarray((audio.data, audio.data)).T, sample_rate=audio.sample_rate)
    assert stereo.nchannels == 2
    wav_file_2 = str(tmpdir.join('stereo.wav'))
    stereo.save(wav_file_2)
    with pytest.raises(ValueError) as err:
        fun([(wav_file_2,)])
    assert 'all wav files are not mono' in str(err.value)

    # ensure we catch differences in sample rates
    capsys.readouterr()  # clear buffer
    w = [(wav_file, ), (wav_file_8k, )]
    out = fun(w)
    err = capsys.readouterr().err
    assert 'several sample rates found in wav files' in err
    assert sorted(out.keys()) == ['utt_1', 'utt_2']

    # make sure timestamps are ordered
    with pytest.raises(ValueError) as err:
        fun([('1', wav_file, 1, 0)])
    assert 'timestamps are not in increasing order for' in str(err.value)


def test_processor_bad():
    get = PipelineManager.get_processor_class
    with pytest.raises(ValueError) as err:
        get('bad')
    assert 'invalid processor "bad"' in str(err.value)

    with pytest.raises(ValueError) as err:
        get(0)
    assert 'invalid processor "0"' in str(err.value)


@pytest.mark.parametrize('features', pipeline.valid_features())
def test_extract_features(utterances_index, features):
    config = pipeline.get_default_config(
        features, with_cmvn=False, with_pitch=False)
    feats = pipeline.extract_features(config, utterances_index)
    feat1 = feats[utterances_index[0][0]]
    assert feat1.is_valid()
    assert feat1.shape[0] == 140
    assert feat1.dtype == np.float32

    config = pipeline.get_default_config(
        features, with_cmvn=False, with_pitch=True)
    feats = pipeline.extract_features(config, utterances_index)
    feat2 = feats[utterances_index[0][0]]
    assert feat2.is_valid()
    assert feat2.shape[0] == 140
    assert feat2.shape[1] == feat1.shape[1] + 3

    config = pipeline.get_default_config(
        features, with_cmvn=False, with_pitch=False, with_crepe_pitch=True)
    config['crepe_pitch']['model_capacity'] = 'tiny'
    feats = pipeline.extract_features(config, utterances_index)
    feat2 = feats[utterances_index[0][0]]
    assert feat2.is_valid()
    assert feat2.shape[0] == 140
    assert feat2.shape[1] == feat1.shape[1] + 3

    utterances_index = [('u1', utterances_index[0][1], 0, 1)]
    config = pipeline.get_default_config(
        features, with_cmvn=False, with_pitch=False)
    feats = pipeline.extract_features(config, utterances_index)
    feat3 = feats[utterances_index[0][0]]
    assert feat3.is_valid()
    assert feat3.shape[0] == 98
    assert feat3.shape[1] == feat1.shape[1]


@pytest.mark.parametrize(
    'by_speaker, with_vad',
    [(s, v) for s in (True, False) for v in (True, False)])
def test_cmvn(utterances_index, by_speaker, with_vad):
    config = pipeline.get_default_config(
        'mfcc', with_cmvn=True, with_pitch=False, with_delta=False)
    config['cmvn']['by_speaker'] = by_speaker
    config['cmvn']['with_vad'] = with_vad
    feats = pipeline.extract_features(config, utterances_index)
    feat2 = feats[utterances_index[0][0]]
    assert feat2.is_valid()
    assert feat2.shape[0] == 140
    assert feat2.shape[1] == 13


def test_sliding_window_cmvn(utterances_index):
    config = pipeline.get_default_config(
        'mfcc', with_cmvn=False, with_pitch=False,
        with_sliding_window_cmvn=True, with_delta=False)
    feats = pipeline.extract_features(config, utterances_index)
    feat2 = feats[utterances_index[0][0]]
    assert feat2.is_valid()
    assert feat2.shape[0] == 140
    assert feat2.shape[1] == 13


@pytest.mark.parametrize('with_vtln', ['simple', 'full'])
def test_extract_features_with_vtln(utterances_index, with_vtln):
    config = pipeline.get_default_config(
        'mfcc', with_pitch=False, with_vtln=with_vtln, with_delta=False)
    config['vtln']['ubm']['num_gauss'] = 4
    config['vtln']['ubm']['num_iters'] = 1
    config['vtln']['ubm']['num_iters_init'] = 1
    config['vtln']['num_iters'] = 1
    feats = pipeline.extract_features(config, utterances_index)
    feat2 = feats[utterances_index[0][0]]
    assert feat2.is_valid()
    assert feat2.shape[0] == 140
    assert feat2.shape[1] == 13


@pytest.mark.parametrize('ext', supported_extensions().keys())
def test_extract_features_full(ext, wav_file, wav_file_8k, wav_file_float32,
                               capsys, tmpdir):
    # difficult case with parallel jobs, different sampling rates,
    # speakers and segments
    index = [
        ('u1', wav_file, 's1', 0, 1),
        ('u2', wav_file_float32, 's2', 1, 1.2),
        ('u3', wav_file_8k, 's1', 1, 3)]
    config = pipeline.get_default_config('mfcc')

    # disable VAD because it can alter the cmvn result (far from (0,
    # 1) when the signal includes non-voiced frames)
    config['cmvn']['with_vad'] = False

    feats = pipeline.extract_features(
        config, index, njobs=2, log=logger.get_logger('test', 'info'))

    # ensure we have the expected log messages
    messages = capsys.readouterr().err
    assert 'INFO - test - get 3 utterances from 2 speakers in 3 wavs' in messages
    assert 'WARNING - test - several sample rates found in wav files' in messages

    for utt in ('u1', 'u2', 'u3'):
        assert utt in feats
        assert feats[utt].dtype == np.float32

    # check properies
    p1 = feats['u1'].properties
    p2 = feats['u2'].properties
    p3 = feats['u3'].properties
    assert p1['audio']['file'] == wav_file
    assert p1['audio']['duration'] == 1.0
    assert p2['audio']['file'] == wav_file_float32
    assert p2['audio']['duration'] == pytest.approx(0.2)
    assert p3['audio']['file'] == wav_file_8k
    assert p3['audio']['duration'] < 0.5  # ask 3s but get duration-tstart
    assert p1['mfcc'] == p2['mfcc']
    assert p1['mfcc']['sample_rate'] != p3['mfcc']['sample_rate']
    assert p1.keys() == {
        'audio', 'mfcc', 'cmvn', 'pitch', 'delta', 'speaker', 'pipeline'}
    assert p1.keys() == p2.keys() == p3.keys()
    assert p1['pipeline'] == p2['pipeline'] == p3['pipeline']

    # check shape. mfcc*delta + pitch = 13 * 3 + 3 = 42
    assert feats['u1'].shape == (98, 42)
    assert feats['u2'].shape == (18, 42)
    assert feats['u3'].shape == (40, 42)

    # check cmvn
    assert feats['u2'].data[:, :13].mean() == pytest.approx(0.0, abs=1e-5)
    assert feats['u2'].data[:, :13].std() == pytest.approx(1.0, abs=1e-5)

    data = np.vstack((feats['u1'].data[:, :13], feats['u3'].data[:, :13]))
    assert data.mean() == pytest.approx(0.0, abs=1e-5)
    assert data.std() == pytest.approx(1.0, abs=1e-5)
    assert np.abs(data.mean()) <= np.abs(feats['u1'].data[:, :13].mean())
    assert np.abs(data.std() - 1.0) <= np.abs(
        feats['u1'].data[:, :13].std() - 1.0)
    assert np.abs(data.mean()) <= np.abs(feats['u3'].data[:, :13].mean())
    assert np.abs(data.std() - 1.0) <= np.abs(
        feats['u3'].data[:, :13].std() - 1.0)

    # save / load the features
    filename = str(tmpdir.join('feats' + ext))
    feats.save(filename)
    feats2 = FeaturesCollection.load(filename)
    assert feats2 == feats
