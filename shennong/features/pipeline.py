"""Features extraction pipeline

The pipeline takes as input a list of wav files, extracts the features
and do the postprocessing.

"""

import collections
import datetime
import numpy as np
import os
import re
import textwrap
import yaml

from shennong.audio import AudioData
from shennong.features import FeaturesCollection
from shennong.features.processor.bottleneck import BottleneckProcessor
from shennong.features.processor.energy import EnergyProcessor
from shennong.features.processor.filterbank import FilterbankProcessor
from shennong.features.processor.mfcc import MfccProcessor
from shennong.features.processor.plp import PlpProcessor
from shennong.features.processor.pitch import (
    PitchProcessor, PitchPostProcessor)
from shennong.features.postprocessor.cmvn import CmvnPostProcessor
from shennong.features.postprocessor.delta import DeltaPostProcessor
from shennong.features.postprocessor.vad import VadPostProcessor
from shennong.utils import get_logger, get_njobs


_valid_features = ['mfcc', 'filterbank', 'plp', 'bottleneck']

_valid_processors = {
    'bottleneck': BottleneckProcessor,
    'energy': EnergyProcessor,
    'filterbank': FilterbankProcessor,
    'mfcc': MfccProcessor,
    'pitch': PitchProcessor,
    'pitch_post': PitchPostProcessor,
    'plp': PlpProcessor,
    'cmvn': CmvnPostProcessor,
    'delta': DeltaPostProcessor,
    'vad': VadPostProcessor}


def extract_features(config, wavs_index, njobs=1, log=get_logger()):
    """Speech features extraction pipeline

    Format of each element in `wavs_index`:
    * 1-uple (or str): <wav-file>
    * 2-uple: <utterance-id> <wav-file>
    * 3-uple: <utterance-id> <wav-file> <speaker-id>
    * 4-uple: <utterance-id> <wav-file> <tstart> <tstop>
    * 5-uple: <utterance-id> <wav-file> <speaker-id> <tstart> <tstop>

    Parameters
    ----------
    config : dict or str
        The pipeline configuration, can be a dictionary, a path to a
        YAML file or a string formatted in YAML.
    wavs_index : sequence of tuples
        The list of wav file to extract the features on.
    njobs : int, optional
        The number to subprocesses to execute in parallel, use a
        single process by default.

    """
    # intialize the pipeline configuration and the list of wav files
    # to process, make all the checks to ensure all is correct
    njobs = get_njobs(njobs, log=log)
    config = _init_config(config, log=log)
    wavs_index = _init_wavs(wavs_index, log=log)

    # ensures speakers info provided if cmvn by speaker is requested
    utt_ids = list(wavs_index.keys())
    try:
        if config['cmvn']['by_speaker'] and not wavs_index[utt_ids[0]].speaker:
            raise ValueError(
                'cmvn normalization by speaker requested '
                'but no speaker information provided')
    except KeyError:
        pass

    # instanciate the features processor and postprocessors all in a
    # single dictionnary
    features = [k for k in config.keys() if k in _valid_features][0]
    processors = {}
    processors['features'] = _valid_processors[features](**config[features])
    if 'cmvn' in config:
        processors['cmvn'] = {**config['cmvn']}
        if config['cmvn']['by_speaker']:
            # instanciate a CMVN processor by speaker
            speakers = set(w.speaker for w in wavs_index.values())
            for speaker in speakers:
                processors['cmvn'][speaker] = CmvnPostProcessor(
                    processors['features'].ndims)
        else:
            processors['cmvn']['cmvn'] = CmvnPostProcessor(
                    processors['features'].ndims)
        if config['cmvn']['with_vad']:
            processors['cmvn']['vad'] = VadPostProcessor(**config['vad'])
    if 'delta' in config:
        processors['delta'] = DeltaPostProcessor(**config['delta'])
    if 'pitch' in config:
        processors['pitch'] = PitchProcessor(**config['pitch'])
        # forward framing parameters
        p = processors['pitch']
        p.frame_shift = processors['features'].frame_shift
        p.frame_length = processors['features'].frame_length
        processors['pitch_post'] = PitchPostProcessor(**config['pitch_post'])

    # TODO parallelize this 2 functions loop. Before: why several CPUs
    # are in use (only for pitch)? Because numpy?
    feats = FeaturesCollection()
    pitch = FeaturesCollection()
    final = FeaturesCollection()

    # extract features and pitch, accumulate cmvn stats
    for name, wav_entry in wavs_index.items():
        log.info('step 1: %s', name)
        feats[name], pitch[name] = _extract_one(wav_entry, processors, log=log)

    # apply cmvn and extract deltas
    for name, wav_entry in wavs_index.items():
        log.info('step 2: %s', name)
        final[name] = _extract_two(
            wav_entry, feats[name], pitch[name], processors, log=log)

    # TODO log a message with structure of the output features
    # (labels on columns)
    return final


def _init_config(config, log=get_logger()):
    try:
        if os.path.isfile(config):
            log.debug('loading configuration from %s', config)
            config = open(config, 'r')
    except TypeError:
        pass

    if isinstance(config, str):
        # the config is a string, try to load it as a YAML
        try:
            config = yaml.load(config, Loader=yaml.FullLoader)
        except yaml.YAMLError as err:
            raise ValueError('Error in configuration file: {}', str(err))

    # enumeratesures all the keys in config are known
    unknown_keys = [
        k for k in config.keys() if k not in _valid_processors.keys()]
    if unknown_keys:
        raise ValueError(
            'invalid keys in configuration: {}'.format(
                ', '.join(unknown_keys)))

    # ensure one and only one features processor is defined in the
    # configuration
    features = [k for k in config.keys() if k in _valid_features]
    if not features:
        raise ValueError(
            'the configuration do not define any features extraction, '
            'only post-processing (must have one and only one entry of {})'
            .format(', '.join(_valid_features)))
    if len(features) > 1:
        raise ValueError(
            'more than one features extraction processor is defined, '
            '(must have one and only one entry of {}): {}'
            .format(', '.join(_valid_features)), ', '.join(features))

    # if cmvn with vad, make sure vad options are defined
    try:
        if config['cmvn']['with_vad']:
            if 'vad' not in config:
                raise ValueError(
                    'requested cmvn with vad, but no vad options defined')
    except KeyError:
        pass

    # if we have pitch or pitch_post, make sure we have the other
    if 'pitch' in config and 'pitch_post' not in config:
        raise ValueError(
            'configuration defines pitch but not pitch_post')
    if 'pitch_post' in config and 'pitch' not in config:
        raise ValueError(
            'configuration defines pitch_post but not pitch')

    # log message describing the pipeline configuration
    post = []
    if 'pitch' in config:
        post.append('pitch')
    if 'cmvn' in config:
        post.append('cmvn')
    if 'delta' in config:
        post.append('delta')
    log.info(
        'pipeline configured for %s features extraction%s',
        features[0], ' with {}'.format(', '.join(post)) if post else '')

    return config


def _init_wavs(wavs, log=get_logger()):
    # guess the for format of `wavs` and ensure it is homogeneous
    wavs = list((w,) if isinstance(w, str) else w for w in wavs)
    format = set(len(w) for w in wavs)
    if not len(format) == 1:
        raise ValueError(
            'the wavs index is not homogeneous, entries have different '
            'length: {}'.format(', '.join(str(t) for t in format)))
    format = list(format)[0]

    # ensure the wavs index format is valid
    valid_formats = {
        1: '<wav-file>',
        2: '<utterance-id> <wav-file>',
        3: '<utterance-id> <wav-file> <speaker-id>',
        4: '<utterance-id> <wav-file> <tstart> <tstop>',
        5: '<utterance-id> <wav-file> <speaker-id> <tstart> <tstop>'}
    try:
        log.info(
            'detected format for wavs index is: %s',
            valid_formats[format])
    except KeyError:
        raise ValueError('unknown format for wavs index')

    # ensure 1st column has unique elements
    duplicates = [w for w, c in collections.Counter(
        w[0] for w in wavs).items() if c > 1]
    if duplicates:
        raise ValueError(
            'duplicates found in wavs index: %s', ', '.join(duplicates))

    # build a dict {utt_id: (wav_file, speaker_id, tstart, tstop)}
    wav_entry = collections.namedtuple(
        'wav_entry', ['file', 'speaker', 'tstart', 'tstop'])
    wavs_index = {}
    for n, wav in enumerate(wavs, start=1):
        if format == 1:
            utt_id = 'utt_{}'.format(str(n))
            wav_file = wav[0]
        else:
            utt_id = wav[0]
            wav_file = wav[1]

        wavs_index[utt_id] = wav_entry(
            file=wav_file,
            speaker=wav[2] if format in (3, 5) else None,
            tstart=(float(wav[2]) if format == 4
                    else float(wav[3]) if format == 5 else None),
            tstop=(float(wav[3]) if format == 4
                   else float(wav[4]) if format == 5 else None))

    # ensure all the wavs are here, log the total duration and the
    # number of speakers, make sure all wavs are mono abd check the
    # sample rate (warning if all the wavs are not at the same sample
    # rate)
    wavs = [w.file for w in wavs_index.values()]
    not_found = [w for w in wavs if not os.path.isfile(w)]
    if not_found:
        raise ValueError(
            'the following wav files are not found: {}'
            .format(', '.join(not_found)))
    wavs_metadata = [AudioData.scan(w) for w in wavs]
    log.info(
        'get %s wav files%s, total duration: %s',
        len(wavs),
        '' if format not in (3, 5) else ' from {} speakers'.format(
            len(set(w.speaker for w in wavs_index.values()))),
        datetime.timedelta(seconds=sum(w.duration for w in wavs_metadata)))
    if not all(w.nchannels == 1 for w in wavs_metadata):
        raise ValueError('all wavs are not mono, exiting')
    samplerates = set(w.sample_rate for w in wavs_metadata)
    if len(samplerates) > 1:
        log.warning(
            'several sample rates found in wav files: %s, features extraction '
            'pipeline will work but this may not be a good idea to work on '
            'heterogeneous data',
            ', '.join(str(s) + 'Hz' for s in samplerates))

    # ensure all (tstart, tstop) pairs are valid (numbers and
    # tstart < tstop)
    if format in (4, 5):
        tstamps = [(w.tstart, w.tstop, w.file) for w in wavs_index.values()]
        for (tstart, tstop, wfile) in tstamps:
            if not tstart < tstop:
                raise ValueError(
                    'timestamps are not in increasong order for {}: {} >= {}'
                    .format(wfile, tstart, tstop))

    return wavs_index


def default_config(features, to_yaml=False, yaml_commented=True,
                   with_pitch=True, with_cmvn=True, with_delta=True):
    """Returns the default configuration for the specified pipeline

    The pipeline is specified with the main `features` it computes and
    the postprocessing steps it includes. The returned configuration
    can be a dictionnay or a YAML formatted string.

    Parameters
    ----------
    features : str
        The features extracted by the pipeline, must be 'mfcc',
        'filterbank', 'plp' or 'bottleneck'.
    to_yaml : bool, optional
        If False the result configuration is a dict, if True this is a
        YAML formatted string ready to be written to a file. Default
        to False.
    yaml_commented : bool, optional
        If True add the docstring of each parameter as a comment in
        the YAML string, if False do nothing. This option has an
        effect only if `to_yaml` is True. Default to True.
    with_pitch : bool, optional
        Configure the pipeline for pitch extraction, default to True
    with_cmvn : bool, optional
        Configure the pipeline for CMVN normalization of the features,
        default to True.
    with_delta : bool, optional
        Configure the pipeline for features's delta extraction,
        default to True.

    Returns
    -------
    config : dict or str
        If `to_yaml` is True returns a YAML formatted string ready to
        be written to a file, else returns a dictionary.

    Raises
    ------
    ValueError
        If `features` are not 'mfcc', 'filterbank', 'plp' or
        'bottleneck'.

    """
    # check features are correct
    if features not in _valid_features:
        raise ValueError('invalid features "{}", must be in {}'.format(
            features, ', '.join(_valid_features)))

    config = {}

    # filter out sample rate parameter because it is dependent of
    # the input wav file
    config[features] = {
        k: v for k, v in
        _valid_processors[features]().get_params().items()
        if k not in ('sample_rate', 'htk_compat')}

    if with_pitch:
        # filter out the frame parameters, already specified for
        # the features, and sample rate
        config['pitch'] = {
            k: v for k, v
            in _valid_processors['pitch']().get_params().items()
            if k not in ('frame_length', 'frame_shift', 'sample_rate')}
        config['pitch_post'] = _valid_processors['pitch_post']().get_params()

    if with_cmvn:
        config['cmvn'] = {
            'by_speaker': True, 'with_vad': True}
        config['vad'] = _valid_processors['vad']().get_params()

    if with_delta:
        config['delta'] = _valid_processors['delta']().get_params()

    if to_yaml:
        return _get_config_to_yaml(config, comments=yaml_commented)
    return config


def _get_config_to_yaml(config, comments=True):
    """Converts a configuration from dict to a yaml string

    Auxiliary method to :func:`default_config`.

    Parameters
    ----------
    config : dict
        A dict of parameters, one key per processor and
        postprocessors. For each key, the value is a dict as well,
        with a mapping (parameter name: parameter value).
    comments : bool, optional
        When True, include the parameters docstrings as comments
        in the yaml string, default to True

    Returns
    -------
    yaml : str
        A string formatted to the YAML format, ready to be written
        to a file

    """
    # do not sort keys by alphabetical order
    yaml.add_representer(
        dict, lambda self, data:
        yaml.representer.SafeRepresenter.represent_dict(
            self, data.items()))

    # inform yaml to represent numpy floats as standard floats
    yaml.add_representer(
        np.float32, yaml.representer.Representer.represent_float)

    # build the yaml formated multiline string
    config = yaml.dump(config).strip()

    if not comments:
        return config

    # incrust the parameters docstings as comments in the yaml
    config_commented = []
    processor = None
    for line in config.split('\n'):
        if not line.startswith(' '):
            processor = line[:-1]
            config_commented.append(line)
        else:
            param = line.split(': ')[0].strip()
            default = line.split(': ')[1].strip()

            if processor == 'cmvn' and param == 'by_speaker':
                docstring = (
                    'If false, do normalization by wav file, '
                    'if true do normalization by speaker')
            elif processor == 'cmvn' and param == 'with_vad':
                docstring = (
                    'If true do normalization only on frames where '
                    'voice activity has been detected, if false do not '
                    'consider voice activity for normalization')
            else:
                docstring = getattr(
                    _valid_processors[processor], param).__doc__ or ''
                docstring = re.sub(r'\n\n', '. ', docstring)
                docstring = re.sub(r'\n', ' ', docstring)
                docstring = re.sub(r' +', ' ', docstring).strip()

            docstring += '. Default is {}.'.format(default)
            docstring = re.sub(r'\.+', '.', docstring)

            config_commented += [
                '  # ' + w for w in textwrap.wrap(docstring, width=66)]
            config_commented.append(line)

    return '\n'.join(config_commented)


def _extract_one(wav_entry, processors, log=get_logger()):
    audio = AudioData.load(wav_entry.file)

    # features extraction
    p = processors['features']
    p.sample_rate = audio.sample_rate
    feats = p.process(audio)

    # TODO vad
    if 'cmvn' in processors:
        if processors['cmvn']['by_speaker']:
            p = processors['cmvn'][wav_entry.speaker]
        else:
            p = processors['cmvn']['cmvn']
        p.accumulate(feats)

    if 'pitch' in processors:
        p1 = processors['pitch']
        p1.sample_rate = audio.sample_rate
        # p2 = processors['pitch_post']
        # pitch = p2.process(p1.process(audio))
        pitch = p1.process(audio)
    else:
        pitch = None

    return feats, pitch


def _extract_two(wav_entry, feats, pitch, processors, log=get_logger()):
    # apply cmvn
    if 'cmvn' in processors:
        if processors['cmvn']['by_speaker']:
            p = processors['cmvn'][wav_entry.speaker]
        else:
            p = processors['cmvn']['cmvn']
        feats = p.process(feats)

    if 'delta' in processors:
        feats = processors['delta'].process(feats)

    if pitch:
        return feats.concatenate(pitch)

    # TODO clean properties (with reference to wav, speaker, all
    # processors, etc...)
    return feats
