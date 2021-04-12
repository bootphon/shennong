# coding: utf-8
"""High-level functions for a complete features extraction pipeline

This module exposes two main functions :func:`get_default_config` that
generates a configuration for the pipeline given some arguments, and
:func:`extract_features` which takes as input a configuration and a
list of utterances, extracts the features, do the postprocessing and
returns the extracted features as an instance of
:class:`~shennong.features.features.FeaturesCollection`.

Examples
--------

>>> from shennong.pipeline import get_default_config, extract_features

Generates a configuration for MFCC extraction (including CMVN
normalization by speaker, delta / delta-delta and pitch). The
configuration is a dictionary:

>>> config = get_default_config(
...     'mfcc', with_pitch='kaldi', with_cmvn=True, with_delta=True)
>>> config.keys()
dict_keys(['mfcc', 'pitch', 'cmvn', 'delta'])

Generates the same configuration, but without CMVN and without
delta:

>>> config = get_default_config('mfcc', with_pitch='kaldi')
>>> config.keys()
dict_keys(['mfcc', 'pitch'])

The returned configuration is intialized with default parameters. This
is suitable for most usages, but you change them if you want. Here we
are using a blackman windox for frame extraction, and we are changing
the min/max F0 frequency for pitch extraction:

>>> config['mfcc']['window_type'] = 'blackman'
>>> config['mfcc']['blackman_coeff'] = 0.42
>>> config['pitch']['min_f0'] = 25
>>> config['pitch']['max_f0'] = 400

Generates a list of utterances to extract the features on (here we
have 2 utterances from the same speaker and same file):

>>> from shennong import Utterances
>>> wav = './test/data/test.wav'
>>> utterances = Utterances([
...     ('utt1', wav, 'spk1', 0, 1), ('utt2', wav, 'spk1', 1, 1.4)])

Extract the features:

>>> features = extract_features(config, utterances, njobs=1)
>>> features.keys()
dict_keys(['utt1', 'utt2'])
>>> type(features['utt1'])
<class 'shennong.features.Features'>
>>> features['utt1'].shape
(98, 16)

The extracted features embed a ``property`` dictionnary with
information on the input audio, pipeline parameters, etc. The field
'pipeline' describe as a list the processing steps being executed, as
well as the columns of the resulting features matrix (here MFCCs are
on columns 0 to 12 and pitch on columns 13 to 15):

>>> p = features['utt1'].properties
>>> p.keys()
dict_keys(['pipeline', 'mfcc', 'speaker', 'audio', 'pitch'])
>>> p['pipeline']
[{'name': 'mfcc', 'columns': [0, 12]}, {'name': 'pitch', 'columns': [13, 15]}]

"""

import os
import textwrap

import numpy as np
import joblib
import yaml

from shennong import FeaturesCollection
from shennong.logger import get_logger
from shennong.utils import get_njobs
from shennong.pipeline_manager import PipelineManager


def valid_features():
    """Returns the list of features that can be extracted by the pipeline.Audio

    This list only includes main features extraction algorithms and
    excludes postprocessing. See also :func:`get_default_config`.

    """
    return PipelineManager.valid_features


def get_default_config(
        features,
        to_yaml=False,
        yaml_commented=True,
        with_pitch=False,
        with_cmvn=False,
        with_sliding_window_cmvn=False,
        with_delta=False,
        with_vtln=False):
    """Returns the default configuration for the specified pipeline

    The pipeline is specified with the main `features` it computes and
    the postprocessing steps it includes. The returned configuration
    can be a dictionnay or a YAML formatted string.

    Parameters
    ----------
    features : str
        The features extracted by the pipeline, must be 'mfcc',
        'filterbank', 'plp' or 'bottleneck'. See also
        :func:`valid_features`.
    to_yaml : bool, optional
        If False the result configuration is a dict, if True this is a
        YAML formatted string ready to be written to a file. Default
        to False.
    yaml_commented : bool, optional
        If True add the docstring of each parameter as a comment in
        the YAML string, if False do nothing. This option has an
        effect only if ``to_yaml`` is True. Default to True.
    with_pitch : False, 'kaldi' or 'crepe', optional
        Configure the pipeline for pitch extraction using Kaldi or CREPE,
        default to False.
    with_cmvn : bool, optional
        Configure the pipeline for CMVN normalization of the features,
        default to False.
    with_sliding_window_cmvn: bool, optional
        Configure the pipeline for sliding window CMVN normalization
        of the features, default to False.
    with_delta : bool, optional
        Configure the pipeline for features's delta extraction,
        default to False.
    with_vtln : bool or str, optional
        Configure the pipeline for VTLN normalization, default to False. Must
        be False, 'simple' or 'full'. When 'simple' the features default to
        MFCC with default values. When 'full' all features parameters are
        exposed.

    Returns
    -------
    config : dict or str
        If ``to_yaml`` is True returns a YAML formatted string ready
        to be written to a file, else returns a dictionary.

    Raises
    ------
    ValueError
        If ``features`` is not in :func:`valid_features` or if ``with_pitch``
        is not valid.

    """
    # check features are correct
    if features not in valid_features():
        raise ValueError('invalid features "{}", must be in {}'.format(
            features, ', '.join(valid_features())))

    if with_pitch not in (False, 'kaldi', 'crepe'):
        raise ValueError(
            f'with_pitch argument must be False, "kaldi" or "crepe" '
            f'but is "{with_pitch}"')

    config = {}

    # filter out sample rate parameter because it is dependent of
    # the input wav file
    config[features] = {
        k: v for k, v in
        PipelineManager.get_processor_params(features).items()
        if k not in ('sample_rate', 'htk_compat')}

    if with_pitch:  # 'kaldi' or 'crepe'
        # filter out the frame parameters, already specified for
        # the features, and sample rate
        config['pitch'] = {'processor': with_pitch}
        for key, value in (
                PipelineManager.get_processor_params(
                    f'{with_pitch}_pitch').items()):
            if key not in ('frame_length', 'frame_shift', 'sample_rate'):
                config['pitch'][key] = value
        config['pitch']['postprocessing'] = (
            PipelineManager.get_processor_params(f'{with_pitch}_pitch_post'))

    if with_cmvn:
        config['cmvn'] = {'by_speaker': True, 'with_vad': True}
        config['cmvn']['vad'] = PipelineManager.get_processor_params('vad')

    if with_sliding_window_cmvn:
        config['sliding_window_cmvn'] = PipelineManager.get_processor_params(
            'sliding_window_cmvn')

    if with_delta:
        config['delta'] = PipelineManager.get_processor_params('delta')

    if with_vtln:
        if with_vtln not in ('simple', 'full'):
            raise ValueError(
                f'invalid value for "with_vtln", must be "simple" '
                f'or "full" but is "{with_vtln}"')

        config['vtln'] = PipelineManager.get_processor_params('vtln')

        if with_vtln == 'simple':
            config['vtln']['features'] = 'default'
            config['vtln']['ubm']['features'] = 'default'

    if to_yaml:
        return _get_config_to_yaml(config, comments=yaml_commented)
    return config


def extract_features(configuration, utterances,
                     njobs=1, log=get_logger('pipeline', 'warning')):
    """Speech features extraction pipeline

    Given a pipeline ``configuration`` and ``utterances`` defining a list of
    utterances on which to extract features, this function applies the whole
    pipeline and returns the extracted features as an instance of
    :class:`~shennong.features.features.FeaturesCollection`. It uses ``njobs``
    parallel subprocesses.

    Parameters
    ----------
    config : dict or str
        The pipeline configuration, can be a dictionary, a path to a
        YAML file or a string formatted in YAML. To get a
        configuration example, see :func:`get_default_config`
    utterances : :class:`~shennong.utterances.Utterances`
        The list of utterances to extract the features on.
    njobs : int, optional
        The number to subprocesses to execute in parallel, use a
        single process by default.
    log : logging.Logger
        A logger to display messages during pipeline execution

    Returns
    -------
    features : :class:`~shennong.features.features.FeaturesCollection`
       The extracted speech features

    Raises
    ------
    ValueError
        If the ``configuration`` or the ``utterances`` are invalid, or if
        something goes wrong during features extraction.

    """
    # intialize the pipeline configuration, the list of wav files to
    # process, instanciate the pipeline processors and make all the
    # checks to ensure all is correct
    njobs = get_njobs(njobs, log=log)
    config = _init_config(configuration, log=log)

    log.info(
        'detected format for utterances index is: %s',
        utterances.format(type=str))

    # check the OMP_NUM_THREADS variable for parallel computations
    _check_environment(njobs, log=log)

    # do all the computations
    return _extract_features(config, utterances, njobs=njobs, log=log)


# a little tweak to change the &log message in joblib parallel loops
class _Parallel(joblib.Parallel):
    def __init__(self, name, log, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.log = log

    def _print(self, msg, msg_args):
        if not self.verbose:  # pragma: nocover
            return
        msg = (msg % msg_args).replace('Done', 'done')
        self.log.info('%s: %s', self, msg)

    def __repr__(self):
        return self.name


def _check_environment(njobs, log=get_logger('pipeline', 'warning')):
    if njobs == 1:
        return

    try:
        nthreads = int(os.environ['OMP_NUM_THREADS'])
    except KeyError:
        nthreads = None

    if not nthreads or nthreads > 1:
        log.warning(
            'working on %s threads but implicit parallelism is active, '
            'this may slow down the processing. Set the environment variable '
            'OMP_NUM_THREADS=1 to disable this warning', njobs)


def _get_config_to_yaml(config, comments=True):
    """Converts a configuration from dict to a yaml string

    Auxiliary method to :func:`get_default_config`.

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
    # inform yaml to not sort keys by alphabetical order
    yaml.add_representer(
        dict, lambda self, data:
        yaml.representer.SafeRepresenter.represent_dict(self, data.items()))

    # inform yaml to represent numpy floats as standard floats
    yaml.add_representer(
        np.float32, yaml.representer.Representer.represent_float)

    # store the pitch processor (if any)
    try:
        pitch_processor = config['pitch']['processor']
    except KeyError:
        pitch_processor = None

    # build the yaml formated multiline string
    config = yaml.dump(config).strip()

    if not comments:
        return config + '\n'

    # incrust the parameters docstrings as comments in the yaml
    config_commented = []
    processors = []
    prev_offset = 0
    for line in config.split('\n'):
        offset = len(line.split(': ')[0]) - len(line.split(': ')[0].strip())
        for _ in range((prev_offset - offset)//2):
            processors.pop()
        if line.endswith(':'):
            processor = line[:-1].strip()
            # special case of pitch_postprocessor
            if processor == 'postprocessing':
                processor = f'{processors[-1]}_post'
            processors.append(processor)

            # special case here when '   vad:' we are in the ubm section of
            # vtln: no need to append this comment
            if processor == 'vad' and offset != 4:
                config_commented.append(
                    "  # The vad options are not used if 'with_vad' is false")

            config_commented.append(line)
        else:
            param = line.split(': ')[0].strip()
            default = line.split(': ')[1].strip()
            processor = processors[-1]

            if processor == 'cmvn' and param == 'by_speaker':
                docstring = (
                    'If false, do normalization by utterance, '
                    'if true do normalization by speaker.')
            elif processor == 'cmvn' and param == 'with_vad':
                docstring = (
                    'If true do normalization only on frames where '
                    'voice activity has been detected, if false do not '
                    'consider voice activity for normalization.')
            elif param == 'features' and default == 'default':
                # custom docstring when using VTLN with default features
                docstring = (
                    'Features extraction configuration. Default is to use '
                    'MFCCs with default parameters. Regenerate this '
                    'configuration file with "speech-features config" using '
                    'the "--vtln-full" option to expose all the parameters.')
            elif processor == 'pitch' and param == 'processor':
                docstring = f'Computing pitch using {pitch_processor}'
            elif 'pitch' in processor and param != 'processor':
                docstring = PipelineManager.get_docstring(
                    pitch_processor + '_' + processor, param, default)
            else:
                docstring = PipelineManager.get_docstring(
                    processor, param, default)

            config_commented += [
                ' ' * offset + '# ' + w
                for w in textwrap.wrap(docstring, width=68 - offset)]
            config_commented.append(line)
        prev_offset = offset

    return '\n'.join(config_commented) + '\n'


def _init_config(config, log=get_logger('pipeline', 'warning')):
    try:
        if os.path.isfile(config):
            log.debug('loading configuration from %s', config)
            config = open(config, 'r').read()
    except TypeError:
        pass

    if isinstance(config, str):
        # the config is a string, try to load it as a YAML
        try:
            config = yaml.load(config, Loader=yaml.FullLoader)
        except yaml.YAMLError as err:
            raise ValueError(f'error in configuration: {err}')

    # ensure all the keys in config are known
    unknown_keys = [
        k for k in config.keys()
        if k not in list(PipelineManager.valid_processors) + ['pitch']]
    if unknown_keys:
        raise ValueError(
            'invalid keys in configuration: {}'.format(
                ', '.join(unknown_keys)))

    # ensure one and only one features processor is defined in the
    # configuration
    features = [k for k in config.keys() if k in valid_features()]
    if not features:
        raise ValueError(
            'the configuration does not define any features extraction '
            '(must have one and only one entry of {})'
            .format(', '.join(valid_features())))
    if len(features) > 1:
        raise ValueError(
            'more than one features extraction processors are defined, '
            '(must have one and only one entry of {}): {}'
            .format(', '.join(valid_features()), ', '.join(features)))

    if 'cmvn' in config:
        # force by_speaker to False if not existing
        if 'by_speaker' not in config['cmvn']:
            log.warning(
                'by_speaker option not specified for cmvn, '
                'assuming it is false and doing cmvn by utterance')
            config['cmvn']['by_speaker'] = False
        # force with_vad to True if not existing
        if 'with_vad' not in config['cmvn']:
            config['cmvn']['with_vad'] = True

    # on pitch, make sure we have a 'postprocessing' entry
    if 'pitch' in config and 'postprocessing' not in config['pitch']:
        config['pitch']['postprocessing'] = {}

    # log message describing the pipeline configuration
    msg = []
    if 'pitch' in config:
        msg.append(f'{config["pitch"]["processor"]} pitch')
    if 'delta' in config:
        msg.append('delta')
    if 'cmvn' in config:
        msg.append('cmvn by {}{}'.format(
            'speaker' if config['cmvn']['by_speaker'] else 'utterance',
            ' with vad' if config['cmvn']['with_vad'] else ''))
    if 'vtln' in config:
        msg.append('vtln by {}'.format(
            'speaker' if config['vtln']['by_speaker'] else 'utterance'))

    log.info(
        'pipeline configured for %s features extraction%s',
        features[0], ' with {}'.format(', '.join(msg)) if msg else '')

    return config


def _extract_features(config, utterances, log, njobs=1):
    # the manager will instanciate the pipeline components
    manager = PipelineManager(config, utterances, log=log)

    # verbosity level for joblib (no joblib verbosity on debug level
    # (level <= 10) because each step is already detailed in inner
    # loops
    verbose = 8 if log.getEffectiveLevel() > 10 else 0

    # vtln : compute vtln warps or load pre-computed warps
    if 'vtln' in config:
        manager.warps = manager.get_vtln_processor(
            'vtln').process(utterances, njobs=njobs)

    # cmvn : two passes. 1st with features pitch and cmvn
    # accumulation, 2nd with cmvn application and delta
    if 'cmvn' in config:
        # extract features and pitch, accumulate cmvn stats
        pass_one = _Parallel(
            'features extraction, pass 1', log,
            n_jobs=njobs, verbose=verbose, prefer='threads')(
                joblib.delayed(_extract_pass_one)(
                    utterance, manager, log=log) for utterance in utterances)

        # apply cmvn and extract deltas
        features = FeaturesCollection(_Parallel(
            'features extraction, pass 2', log,
            n_jobs=njobs, verbose=verbose, prefer='threads')(
                joblib.delayed(_extract_pass_two)(
                    utterance, manager, features, pitch, log=log)
                for utterance, features, pitch in pass_one))

    # no cmvn: single pass
    else:
        features = FeaturesCollection(_Parallel(
            'features extraction', log,
            n_jobs=njobs, verbose=verbose, prefer='threads')(
                joblib.delayed(_extract_single_pass)(
                    utterance, manager, log=log) for utterance in utterances))

    return features


def _extract_pass_one(utterance, manager, log):
    # load audio signal of the utterance
    log.debug('%s: load audio', utterance.audio_file)
    audio = manager.get_audio(utterance)

    # main features extraction
    log.debug('%s: extract %s', utterance.name, manager.features)
    if 'vtln' in manager.config:
        features = manager.get_features_processor(utterance).process(
            audio, vtln_warp=manager.get_warp(utterance))
    else:
        features = manager.get_features_processor(utterance).process(audio)

    # cmvn accumulation
    if 'cmvn' in manager.config:
        log.debug('%s: accumulate cmvn', utterance.name)
        # weight CMVN by voice activity detection (null weights on
        # non-voiced frames)
        if manager.config['cmvn']['with_vad']:
            energy = manager.get_energy_processor(utterance).process(audio)
            vad = manager.get_vad_processor(utterance).process(energy)
            vad = vad.data.reshape((vad.shape[0], ))  # reshape as 1d array
        else:
            vad = None

        manager.get_cmvn_processor(utterance).accumulate(
            features, weights=vad)

    # pitch extraction
    if 'pitch' in manager.config:
        processor = manager.config['pitch']['processor']
        log.debug('%s: extract %s pitch', utterance.name, processor)
        p1 = manager.get_pitch_processor(utterance)
        p2 = manager.get_pitch_post_processor(utterance)
        pitch = p2.process(p1.process(audio))
    else:
        pitch = None

    # add info on speaker and audio input on the features properties
    speaker = utterance.speaker
    if speaker:
        features.properties['speaker'] = speaker

    features.properties['audio'] = {
        'file': os.path.abspath(utterance.audio_file),
        'sample_rate': manager.audio_metadata[
            utterance.audio_file].sample_rate}
    if utterance.tstart is not None:
        features.properties['audio']['tstart'] = utterance.tstart
        features.properties['audio']['tstop'] = utterance.tstop
    features.properties['audio']['duration'] = utterance.duration

    return utterance, features, pitch


def _extract_pass_two(utterance, manager, features, pitch, log, tolerance=2):
    # apply cmvn
    if 'cmvn' in manager.config:
        log.debug('%s: apply cmvn', utterance.name)
        features = manager.get_cmvn_processor(utterance).process(features)

    # apply sliding window cmvn
    if 'sliding_window_cmvn' in manager.config:
        log.debug('%s: apply sliding window cmvn', utterance.name)
        features = manager.get_sliding_window_cmvn_processor(
            utterance).process(features)

    # apply delta
    if 'delta' in manager.config:
        log.debug('%s: apply delta', utterance.name)
        features = manager.get_delta_processor(utterance).process(features)

    # concatenate the pitch features to the main ones. because of
    # downsampling in pitch processing the resulting number of frames
    # can differ (the same tolerance is applied in Kaldi, see
    # the paste-feats binary)
    if pitch:
        log.debug('%s: concatenate pitch', utterance.name)
        features = features.concatenate(pitch, tolerance=tolerance, log=log)

    return utterance.name, features


def _extract_single_pass(utterance, manager, log):
    _, features, pitch = _extract_pass_one(utterance, manager, log=log)
    return _extract_pass_two(utterance, manager, features, pitch, log=log)


def _extract_single_pass_warp(utterance, manager, warp, log):
    # load audio signal of the utterance
    log.debug('%s: load audio', utterance.name)
    audio = manager.get_audio(utterance)

    # main features extraction
    log.debug('%s: extract %s', utterance.name, manager.features)
    features = manager.get_features_processor(utterance).process(
        audio, vtln_warp=warp)

    # apply delta
    if 'delta' in manager.config:
        log.debug('%s: apply delta', utterance.name)
        features = manager.get_delta_processor(utterance).process(features)

    return utterance.name, features


def extract_features_warp(configuration, utterances, warp, log, njobs=1):
    """Speech features extraction pipeline when all features are warped
    by the same factor. Used in the
    :func:`~shennong.features.processor.vtln.VtlnProcessor.process`
    method of the :class:`~shennong.features.processor.vtln.VtlnProcessor`.
    """
    # intialize the pipeline configuration, the list of wav files to
    # process, instanciate the pipeline processors and make all the
    # checks to ensure all is correct
    njobs = get_njobs(njobs, log=log)
    config = _init_config(configuration, log=log)

    # check the OMP_NUM_THREADS variable for parallel computations
    _check_environment(njobs, log=log)

    manager = PipelineManager(config, utterances, log=log)

    # verbosity level for joblib (no joblib verbosity on debug level
    # (level <= 10) because each step is already detailed in inner
    # loops
    verbose = 8 if log.getEffectiveLevel() > 10 else 0

    return FeaturesCollection(_Parallel(
        f'features extraction with warp {warp}', log,
        n_jobs=njobs, verbose=verbose, prefer='threads')(
            joblib.delayed(_extract_single_pass_warp)(
                utterance, manager, warp, log=log)
            for utterance in utterances))
