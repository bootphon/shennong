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

>>> config = get_default_config('mfcc')
>>> config.keys()
dict_keys(['mfcc', 'pitch', 'cmvn', 'delta'])

Generates the same configuration, but without CMVN and without
delta:

>>> config = get_default_config('mfcc', with_cmvn=False, with_delta=False)
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

>>> wav = './test/data/test.wav'
>>> utterances = [('utt1', wav, 'spk1', 0, 1), ('utt2', wav, 'spk1', 1, 1.5)]

Extract the features:

>>> features = extract_features(config, utterances, njobs=1)
>>> features.keys()
dict_keys(['utt1', 'utt2'])
>>> type(features['utt1'])
<class 'shennong.features.features.Features'>
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

import collections
import datetime
import importlib
import joblib
import numpy as np
import os
import re
import textwrap
import yaml

from shennong.audio import Audio
from shennong.features import FeaturesCollection
from shennong.utils import get_logger, get_njobs


def valid_features():
    """Returns the list of features that can be extracted by the pipeline.Audio

    This list only includes main features extraction algorithms and
    excludes postprocessing. See also :func:`get_default_config`.

    """
    return _Manager._valid_features


def get_default_config(features, to_yaml=False, yaml_commented=True,
                       with_pitch=True, with_cmvn=True,
                       with_sliding_window_cmvn=False, with_delta=True,
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
    with_pitch : bool, optional
        Configure the pipeline for pitch extraction, default to True
    with_cmvn : bool, optional
        Configure the pipeline for CMVN normalization of the features,
        default to True.
    with_sliding_window_cmvn: bool, optional
        Configure the pipeline for sliding window CMVN normalization
        of the features, default to False.
    with_delta : bool, optional
        Configure the pipeline for features's delta extraction,
        default to True.
    with_vtln:
    with_vad_trimming: bool, optional
        Configure the pipeline for removing silent frames, default to False.

    Returns
    -------
    config : dict or str
        If ``to_yaml`` is True returns a YAML formatted string ready
        to be written to a file, else returns a dictionary.

    Raises
    ------
    ValueError
        If ``features`` is not in :func:`valid_features`.

    """
    # check features are correct
    if features not in valid_features():
        raise ValueError('invalid features "{}", must be in {}'.format(
            features, ', '.join(valid_features())))

    config = {}

    # filter out sample rate parameter because it is dependent of
    # the input wav file
    config[features] = {
        k: v for k, v in
        _Manager.get_processor_params(features).items()
        if k not in ('sample_rate', 'htk_compat')}

    if with_pitch:
        # filter out the frame parameters, already specified for
        # the features, and sample rate
        config['pitch'] = {
            k: v for k, v
            in _Manager.get_processor_params('pitch').items()
            if k not in ('frame_length', 'frame_shift', 'sample_rate')}
        config['pitch']['postprocessing'] = (
            _Manager.get_processor_params('pitch_post'))

    if with_cmvn:
        config['cmvn'] = {'by_speaker': True, 'with_vad': True}
        config['cmvn']['vad'] = _Manager.get_processor_params('vad')

    if with_sliding_window_cmvn:
        config['sliding_window_cmvn'] = _Manager.get_processor_params(
            'sliding_window_cmvn')

    if with_delta:
        config['delta'] = _Manager.get_processor_params('delta')

    if with_vtln:
        if isinstance(with_vtln, str):
            config['vtln'] = {'warps_path': with_vtln}
        else:
            config['vtln'] = _Manager.get_processor_params('vtln')

    if to_yaml:
        return _get_config_to_yaml(config, comments=yaml_commented)
    return config


def extract_features(configuration, utterances_index,
                     njobs=1, log=get_logger()):
    """Speech features extraction pipeline

    Given a pipeline ``configuration`` and an ``utterances_index``
    defining a list of utterances on which to extract features, this
    function applies the whole pipeline and returns the extracted
    features as an instance of
    :class:`~shennong.features.features.FeaturesCollection`. It uses
    ``njobs`` parallel subprocesses.

    The utterances in the ``utterances_index`` can be defined in one
    of the following format (the format must be homogoneous across the
    index, i.e. only one format can be used):

    * 1-uple (or str): ``<wav-file>``
    * 2-uple: ``<utterance-id> <wav-file>``
    * 3-uple: ``<utterance-id> <wav-file> <speaker-id>``
    * 4-uple: ``<utterance-id> <wav-file> <tstart> <tstop>``
    * 5-uple: ``<utterance-id> <wav-file> <speaker-id> <tstart> <tstop>``

    Parameters
    ----------
    config : dict or str
        The pipeline configuration, can be a dictionary, a path to a
        YAML file or a string formatted in YAML. To get a
        configuration example, see :func:`get_default_config`
    utterances_index : sequence of tuples
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
        If the ``configuration`` or the ``utterances_index`` are
        invalid, or if something goes wrong during features
        extraction.

    """
    # intialize the pipeline configuration, the list of wav files to
    # process, instanciate the pipeline processors and make all the
    # checks to ensure all is correct
    njobs = get_njobs(njobs, log=log)
    config = _init_config(configuration, log=log)
    utterances = _init_utterances(utterances_index, log=log)

    # check the OMP_NUM_THREADS variable for parallel computations
    _check_environment(njobs, log=log)

    # do all the computations
    return _extract_features(
        config, utterances, njobs=njobs, log=log)


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


def _check_environment(njobs, log=get_logger()):
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


def _get_config_to_yaml(config, comments=True):  # TODO: adapt with VTLN
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
        yaml.representer.SafeRepresenter.represent_dict(
            self, data.items()))

    # inform yaml to represent numpy floats as standard floats
    yaml.add_representer(
        np.float32, yaml.representer.Representer.represent_float)

    # build the yaml formated multiline string
    config = yaml.dump(config).strip()

    if not comments:
        return config + '\n'

    # incrust the parameters docstrings as comments in the yaml
    config_commented = []
    processor = None
    for line in config.split('\n'):
        if line.endswith(':'):
            processor = line[:-1].strip()
            # special case of pitch_postprocessor
            if processor == 'postprocessing':
                processor = 'pitch_post'

            if processor == 'vad':
                config_commented.append(
                    "  # The vad options are not used if 'with_vad' is false")
            config_commented.append(line)
        else:
            param = line.split(': ')[0].strip()
            default = line.split(': ')[1].strip()

            if processor == 'cmvn' and param == 'by_speaker':
                docstring = (
                    'If false, do normalization by utterance, '
                    'if true do normalization by speaker')
            elif processor == 'cmvn' and param == 'with_vad':
                docstring = (
                    'If true do normalization only on frames where '
                    'voice activity has been detected, if false do not '
                    'consider voice activity for normalization')
            else:
                docstring = _Manager.get_docstring(
                    processor, param, default)

            offset = 4 if processor in ('vad', 'pitch_post') else 2
            config_commented += [
                ' ' * offset + '# ' + w
                for w in textwrap.wrap(docstring, width=68 - offset)]
            config_commented.append(line)

    return '\n'.join(config_commented) + '\n'


def _init_config(config, log=get_logger()):
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
            raise ValueError('error in configuration: {}', str(err))

    # ensure all the keys in config are known
    unknown_keys = [
        k for k in config.keys()
        if k not in _Manager._valid_processors]
    if unknown_keys:
        raise ValueError(
            'invalid keys in configuration: {}'.format(
                ', '.join(unknown_keys)))

    # ensure one and only one features processor is defined in the
    # configuration
    features = [k for k in config.keys() if k in valid_features()]
    if not features:
        raise ValueError(
            'the configuration does not define any features extraction, '
            'only post-processing (must have one and only one entry of {})'
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

    # if pitch, make sure we have a 'postprocessing' entry
    if 'pitch' in config and 'postprocessing' not in config['pitch']:
        config['pitch']['postprocessing'] = {}

    # log message describing the pipeline configuration
    msg = []
    if 'pitch' in config:
        msg.append('pitch')
    if 'delta' in config:
        msg.append('delta')
    if 'cmvn' in config:
        by = 'speaker' if config['cmvn']['by_speaker'] else 'utterance'
        vad = ' with vad' if config['cmvn']['with_vad'] else ''
        msg.append('cmvn by {}{}'.format(by, vad))
    log.info(
        'pipeline configured for %s features extraction%s',
        features[0], ' with {}'.format(', '.join(msg)) if msg else '')

    return config


_Utterance = collections.namedtuple(
    '_Utterance', ['file', 'speaker', 'tstart', 'tstop'])


def _init_utterances(utts_index, log=get_logger()):
    """Returns a dict {utt_id: (wav_file, speaker_id, tstart, tstop)}

    Raises on any error, log a warning on strange but non-critical
    issues.

    """
    # guess the for format of `wavs` and ensure it is homogeneous
    utts = list((u,) if isinstance(u, str) else u for u in utts_index)
    index_format = set(len(u) for u in utts)
    if not len(index_format) == 1:
        raise ValueError(
            'the wavs index is not homogeneous, entries have different '
            'lengths: {}'.format(', '.join(str(t) for t in index_format)))
    index_format = list(index_format)[0]

    # ensure the utterances index format is valid
    valid_formats = {
        1: '<wav-file>',
        2: '<utterance-id> <wav-file>',
        3: '<utterance-id> <wav-file> <speaker-id>',
        4: '<utterance-id> <wav-file> <tstart> <tstop>',
        5: '<utterance-id> <wav-file> <speaker-id> <tstart> <tstop>'}
    try:
        log.info(
            'detected format for utterances index is: %s',
            valid_formats[index_format])
    except KeyError:
        raise ValueError('unknown format for utterances index')

    # ensure 1st column has unique elements
    duplicates = [u for u, c in collections.Counter(
        u[0] for u in utts).items() if c > 1]
    if duplicates:
        raise ValueError(
            'duplicates found in utterances index: {}'.format(
                ', '.join(duplicates)))

    # sort the utterances by wav_file (and then by utt_id), this
    # is a minor optimization to use the cache system of Audio.load(),
    # ie this avoids to reload several times the same wav when using
    # tstart/tstop segments.
    utts = sorted(utts, key=lambda u: u if index_format == 1 else (u[1], u[0]))

    # build the utterances collection as a dict
    # {utt_id: (wav_file, speaker_id, tstart, tstop)}
    utterances = {}
    for n, utt in enumerate(utts, start=1):
        if index_format == 1:
            utt_id = 'utt_{}'.format(str(n))
            wav_file = utt[0]
        else:
            utt_id = utt[0]
            wav_file = utt[1]

        utterances[utt_id] = _Utterance(
            file=wav_file,
            speaker=utt[2] if index_format in (3, 5) else None,
            tstart=(float(utt[2]) if index_format == 4
                    else float(utt[3]) if index_format == 5 else None),
            tstop=(float(utt[3]) if index_format == 4
                   else float(utt[4]) if index_format == 5 else None))

    # ensure all the wavs are here
    wavs = [w.file for w in utterances.values()]
    not_found = [w for w in wavs if not os.path.isfile(w)]
    if not_found:
        raise ValueError(
            'the following wav files are not found: {}'
            .format(', '.join(not_found)))

    return utterances


def _extract_features(config, utterances, njobs=1, log=get_logger()):
    # the manager will instanciate the pipeline components
    manager = _Manager(config, utterances, log=log)

    # verbosity level for joblib (no joblib verbosity on debug level
    # (level <= 10) because each step is already detailed in inner
    # loops
    verbose = 8 if log.getEffectiveLevel() > 10 else 0

    # vtln : compute vtln warps or load pre-computed warps
    if 'vtln' in config:
        if 'warps_path' in config['vtln']:
            # TODO load warps
            log.debug('Loading pre-computed VTLN warps')
            manager.warps = {}
        else:
            log.debug('Computing VTLN warps')
            manager.warps = manager.get_vtln_processor(
                'vtln').process(utterances)

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
        features = FeaturesCollection(**{k: v for k, v in _Parallel(
            'features extraction, pass 2', log,
            n_jobs=njobs, verbose=verbose, prefer='threads')(
                joblib.delayed(_extract_pass_two)(
                    utterance, manager, features, pitch, log=log)
                for utterance, features, pitch in pass_one)})

    # no cmvn: single pass
    else:
        features = FeaturesCollection(**{k: v for k, v in _Parallel(
            'features extraction', log,
            n_jobs=njobs, verbose=verbose, prefer='threads')(
                joblib.delayed(_extract_single_pass)(
                    utterance, manager, log=log) for utterance in utterances)})

    return features


def _extract_pass_one(utt_name, manager, log=get_logger()):
    # load audio signal of the utterance
    log.debug('%s: load audio', utt_name)
    audio = manager.get_audio(utt_name)

    # main features extraction
    log.debug('%s: extract %s', utt_name, manager.features)
    features = manager.get_features_processor(utt_name).process(
        audio, vtln_warp=manager.get_warp(utt_name))

    # cmvn accumulation
    if 'cmvn' in manager.config:
        log.debug('%s: accumulate cmvn', utt_name)
        # weight CMVN by voice activity detection (null weights on
        # non-voiced frames)
        if manager.config['cmvn']['with_vad']:
            energy = manager.get_energy_processor(utt_name).process(audio)
            vad = manager.get_vad_processor(utt_name).process(energy)
            vad = vad.data.reshape((vad.shape[0], ))  # reshape as 1d array
        else:
            vad = None

        manager.get_cmvn_processor(utt_name).accumulate(features, weights=vad)

    # pitch extraction
    if 'pitch' in manager.config:
        log.debug('%s: extract pitch', utt_name)
        p1 = manager.get_pitch_processor(utt_name)
        p2 = manager.get_pitch_post_processor(utt_name)
        pitch = p2.process(p1.process(audio))
    else:
        pitch = None

    # add info on speaker and audio input on the features properties
    speaker = manager.utterances[utt_name].speaker
    if speaker:
        features.properties['speaker'] = speaker

    utterance = manager.utterances[utt_name]
    features.properties['audio'] = {
        'file': os.path.abspath(utterance.file),
        'sample_rate': manager._wavs_metadata[utterance.file].sample_rate}
    if utterance.tstart is not None:
        features.properties['audio']['tstart'] = utterance.tstart
        features.properties['audio']['tstop'] = utterance.tstop
        features.properties['audio']['duration'] = min(
            utterance.tstop - utterance.tstart,
            manager._wavs_metadata[utterance.file].duration - utterance.tstart)
    else:
        features.properties['audio']['duration'] = (
            manager._wavs_metadata[utterance.file].duration)

    return utt_name, features, pitch


def _extract_pass_two(utt_name, manager, features, pitch,
                      tolerance=2, log=get_logger()):
    # apply cmvn
    if 'cmvn' in manager.config:
        log.debug('%s: apply cmvn', utt_name)
        features = manager.get_cmvn_processor(utt_name).process(features)

    # apply sliding window cmvn
    if 'sliding_window_cmvn' in manager.config:
        log.debug('%s: apply sliding window cmvn', utt_name)
        features = manager.get_sliding_window_cmvn_processor(
            utt_name).process(features)

    # apply delta
    if 'delta' in manager.config:
        log.debug('%s: apply delta', utt_name)
        features = manager.get_delta_processor(utt_name).process(features)

    # concatenate the pitch features to the main ones. because of
    # downsampling in pitch processing the resulting number of frames
    # can differ (the same tolerance is applied in Kaldi, see
    # the paste-feats binary)
    if pitch:
        log.debug('%s: concatenate pitch', utt_name)
        features._log = log
        features = features.concatenate(pitch, tolerance=tolerance)

    return utt_name, features


def _extract_single_pass(utt_name, manager, log=get_logger()):
    _, features, pitch = _extract_pass_one(utt_name, manager, log=log)
    return _extract_pass_two(utt_name, manager, features, pitch, log=log)


def _extract_single_pass_warp(utt_name, manager, warp, log=get_logger()):
    # load audio signal of the utterance
    log.debug('%s: load audio', utt_name)
    audio = manager.get_audio(utt_name)

    # main features extraction
    log.debug('%s: extract %s', utt_name, manager.features)
    features = manager.get_features_processor(utt_name).process(
        audio, vtln_warp=warp)

    # apply delta
    if 'delta' in manager.config:
        log.debug('%s: apply delta', utt_name)
        features = manager.get_delta_processor(utt_name).process(features)

    # apply sliding window cmvn
    if 'sliding_window_cmvn' in manager.config:
        log.debug('%s: apply sliding window cmvn', utt_name)
        features = manager.get_sliding_window_cmvn_processor(
            utt_name).process(features)

    return utt_name, features


def _extract_features_warp(configuration, utterances_index, warp,
                           njobs=1, log=get_logger()):
    # TODO: documentation fonction
    # intialize the pipeline configuration, the list of wav files to
    # process, instanciate the pipeline processors and make all the
    # checks to ensure all is correct
    njobs = get_njobs(njobs, log=log)
    config = _init_config(configuration, log=log)
    utterances = _init_utterances(utterances_index, log=log)

    # check the OMP_NUM_THREADS variable for parallel computations
    _check_environment(njobs, log=log)

    manager = _Manager(config, utterances, log=log)

    # verbosity level for joblib (no joblib verbosity on debug level
    # (level <= 10) because each step is already detailed in inner
    # loops
    verbose = 8 if log.getEffectiveLevel() > 10 else 0

    return FeaturesCollection(**{k: v for k, v in _Parallel(
        'features extraction with warp {}'.format(warp), log,
        n_jobs=njobs, verbose=verbose, prefer='threads')(
        joblib.delayed(_extract_single_pass_warp)(
            utterance, manager, warp, log=log) for utterance in utterances)})


class _Manager:
    """This class handles the instanciation of processors for the pipeline

    Instanciation is the "hard part" because it relies on several
    parameters (CMVN or not, by speaker or not, at which sample rate,
    etc...). All this mechanics is abstracted by this class.

    """
    _valid_features = [
        'mfcc', 'plp', 'filterbank', 'bottleneck', 'rastaplp', 'spectrogram']
    """The main features available in shennong, excluding post-processing"""

    _valid_processors = {
        'bottleneck': ('processor', 'BottleneckProcessor'),
        'energy': ('processor', 'EnergyProcessor'),
        'filterbank': ('processor', 'FilterbankProcessor'),
        'mfcc': ('processor', 'MfccProcessor'),
        'pitch': ('processor', 'PitchProcessor'),
        'pitch_post': ('processor', 'PitchPostProcessor'),
        'plp': ('processor', 'PlpProcessor'),
        'rastaplp': ('processor', 'RastaPlpProcessor'),
        'spectrogram': ('processor', 'SpectrogramProcessor'),
        'vtln': ('processor', 'VtlnProcessor'),
        'cmvn': ('postprocessor', 'CmvnPostProcessor'),
        'delta': ('postprocessor', 'DeltaPostProcessor'),
        'sliding_window_cmvn':
            ('postprocessor', 'SlidingWindowCmvnPostProcessor'),
        'vad': ('postprocessor', 'VadPostProcessor')}
    """The features processors as a dict {name: (module, class)}"""

    def __init__(self, config, utterances, log=get_logger()):
        self._config = config
        self._utterances = utterances
        self._warps = {}
        self.log = log

        # the list of speakers
        self._speakers = set(u.speaker for u in self.utterances.values())
        if self._speakers == {None}:
            self._speakers = None
        self._check_speakers()

        # store the metadata because we need to access the sample rate
        # for processors instanciation
        wavs = set(u.file for u in utterances.values())
        self._wavs_metadata = {w: Audio.scan(w) for w in wavs}

        # make sure all the wavs are compatible with the pipeline
        log.info(f'scanning {len(self._utterances)} utterances...')
        self._check_wavs()

        # the features type to be extracted
        self.features = [
            k for k in self.config.keys() if k in self._valid_features][0]

        # get some framing parameters constant for all processors
        # (retrieve them from a features processor instance)
        p = self.get_features_processor(next(iter(self.utterances.keys())))
        self.frame_length = p.frame_length
        self.frame_shift = p.frame_shift

        # if CMVN by speaker, instanciate a CMVN processor by speaker
        # here, else instanciate a processor per utterance
        if 'cmvn' in self.config:
            if self.config['cmvn']['by_speaker']:
                self._cmvn_processors = {
                    spk: self.get_processor_class('cmvn')(p.ndims)
                    for spk in self.speakers}
            else:
                self._cmvn_processors = {
                    utt: self.get_processor_class('cmvn')(p.ndims)
                    for utt in self.utterances}

    @property
    def config(self):
        return self._config

    @property
    def utterances(self):
        return self._utterances

    @property
    def speakers(self):
        return self._speakers

    @property
    def warps(self):
        return self._warps

    @warps.setter
    def warps(self, value):
        if value.keys() != self.utterances.keys():
            raise ValueError('Given warps do not match utterances')
        self._warps = value

    def _check_speakers(self):
        """Ensures the configuration is compatible with speakers information

        On any error raises a ValueError. Logs a warning if speakers
        information is provided but not used by the pipeline. If all is
        good, silently returns None.

        """
        # ensures speakers info provided if cmvn by speaker is requested
        if 'cmvn' not in self.config or not self.config['cmvn']['by_speaker']:
            cmvn_by_speaker = False
        else:  # config['cmvn']['by_speaker'] exists and is True
            assert self.config['cmvn']['by_speaker']
            cmvn_by_speaker = True

        if cmvn_by_speaker and not self.speakers:
            raise ValueError(
                'cmvn normalization by speaker requested '
                'but no speaker information provided')
        if not cmvn_by_speaker and self.speakers:
            self.log.warning(
                'speakers information is provided but will not be used '
                '(CMVN%s disabled)',
                ' by speaker' if 'cmvn' in self.config else '')

    def _check_wavs(self):
        """Ensures all the wav files are compatible with the pipeline"""
        # log the total duration and the number of speakers
        total_duration = sum(w.duration for w in self._wavs_metadata.values())
        # nspeakers = len(self.speakers or self.utterances)
        speakers = ('' if not self.speakers
                    else ' from {} speakers'.format(len(self.speakers)))
        self.log.info(
            'get %s utterances%s in %s wavs, total wavs duration: %s',
            len(self.utterances), speakers, len(self._wavs_metadata),
            datetime.timedelta(seconds=total_duration))

        # make sure all wavs are mono
        if not all(w.nchannels == 1 for w in self._wavs_metadata.values()):
            raise ValueError('all wav files are not mono')

        # check the sample rate (warning if all the wavs are not at the
        # same sample rate)
        samplerates = set(w.sample_rate for w in self._wavs_metadata.values())
        if len(samplerates) > 1:
            self.log.warning(
                'several sample rates found in wav files: %s, features '
                'extraction pipeline will work but this may not be a good '
                'idea to work on heterogeneous data',
                ', '.join(str(s) + 'Hz' for s in samplerates))

        # ensure all (tstart, tstop) pairs are valid (numbers and
        # tstart < tstop)
        tstamps = [
            (w.tstart, w.tstop, w.file) for w in self.utterances.values()]
        for (tstart, tstop, wfile) in tstamps:
            if tstart is not None and tstart > tstop:
                raise ValueError(
                    'timestamps are not in increasing order for {}: '
                    '{} >= {}'.format(wfile, tstart, tstop))

    @classmethod
    def get_processor_class(cls, name):
        """Returns the (post)processor class given its `name`

        This function enables dynamic import of processors classes to
        avoid a big list of useless imports. Raises a ValueError if
        the `name` is not valid or the module/class cannot be
        imported.

        """
        try:
            _module, _class = cls._valid_processors[name]
        except KeyError:
            raise ValueError('invalid processor "{}"'.format(name))
        if name == 'pitch_post':
            name = 'pitch'
        if name == 'sliding_window_cmvn':
            name = 'cmvn'

        module = 'shennong.features.{}.{}'.format(_module, name)
        try:
            module = importlib.import_module(module)
        except ModuleNotFoundError:  # pragma: nocover
            raise ValueError('cannot import module "{}"'.format(module))

        try:
            return module.__dict__[_class]
        except KeyError:    # pragma: nocover
            raise ValueError(
                'cannot find class "{}" in module {}'.format(_class, module))

    @classmethod
    def get_processor_params(cls, name):
        """Returns the processors parameters given its `name`

        Get the processors class using :func:`get_processor_class`,
        instanciate it and returns its parameters with default values
        as a dict {param: value}.

        """
        return cls.get_processor_class(name)().get_params()

    @classmethod
    def get_docstring(cls, processor, param, default):
        """Returns the docstring of a given processor's parameter"""
        # extract the raw docstring
        docstring = getattr(
            cls.get_processor_class(processor), param).__doc__ or ''

        # postprocess to adapt Python docstring to the YAML output
        # (also adds default value)
        docstring = re.sub(r'\n\n', '. ', docstring)
        docstring = re.sub(r'\n', ' ', docstring)
        docstring = re.sub(r'`', '', docstring)
        docstring = re.sub(':func:', '', docstring)
        docstring += '. Default is {}.'.format(default)
        docstring = re.sub(r'\.+', '.', docstring)
        docstring = re.sub(r' +', ' ', docstring)
        docstring = re.sub(r'\. \.', '.', docstring)

        return docstring.strip()

    def get_audio(self, utterance):
        """Returns the audio data for that `utterance`"""
        utt = self.utterances[utterance]
        audio = Audio.load(utt.file)
        if utt.tstart is not None:
            assert utt.tstop > utt.tstart
            audio = audio.segment([(utt.tstart, utt.tstop)])[0]

        if self.features == 'bottleneck':
            # resample here the signal (this avoid bugs if one part of
            # the pipeline on 8k and the other on 16k), then update
            # the metadata for the wav to be used by the rest of the
            # pipeline
            self.log.debug(
                'resampling audio from %dHz@%db to %dHz@%db',
                audio.sample_rate, audio.dtype.itemsize * 8, 8000, 16)

            audio = audio.resample(8000).astype(np.int16)
            self._wavs_metadata[self.utterances[utterance].file] = (
                Audio._metawav(
                    audio.nchannels, audio.sample_rate,
                    audio.nsamples, audio.duration))
        return audio

    def get_features_processor(self, utterance):
        """Instanciates and returns a features extraction processor"""
        wav = self.utterances[utterance].file
        proc = self.get_processor_class(self.features)(
            **self.config[self.features])
        proc._log = self.log
        try:
            proc.sample_rate = self._wavs_metadata[wav].sample_rate
        except AttributeError:
            # bottleneck does not support changing sample rate
            pass
        return proc

    def get_energy_processor(self, utterance):
        """Instanciates and returns an energy processor"""
        wav = self.utterances[utterance].file
        proc = self.get_processor_class('energy')()
        proc.frame_length = self.frame_length
        proc.frame_shift = self.frame_shift
        proc.sample_rate = self._wavs_metadata[wav].sample_rate
        return proc

    def get_vad_processor(self, utterance):
        """Instanciates and returns a VAD processor"""
        return self.get_processor_class('vad')(
            **self.config['cmvn']['vad'])

    def get_cmvn_processor(self, utterance):
        """Instanciates and returns a CMVN processor"""
        if self.config['cmvn']['by_speaker']:
            speaker = self.utterances[utterance].speaker
            return self._cmvn_processors[speaker]
        else:
            return self._cmvn_processors[utterance]

    def get_sliding_window_cmvn_processor(self, utterrance):
        """Instanciates and returns a sliding-window CMVN processor"""
        return self.get_processor_class('sliding_window_cmvn')(
            **self.config['sliding_window_cmvn'])

    def get_pitch_processor(self, utterance):
        """Instanciates and returns a pitch processor"""
        wav = self.utterances[utterance].file
        params = {k: v for k, v in self.config['pitch'].items()
                  if k != 'postprocessing'}
        params['sample_rate'] = self._wavs_metadata[wav].sample_rate
        params['frame_shift'] = self.frame_shift
        params['frame_length'] = self.frame_length
        return self.get_processor_class('pitch')(**params)

    def get_pitch_post_processor(self, utterance):
        """Instanciates and returns a pitch post-processor"""
        return self.get_processor_class('pitch_post')(
            **self.config['pitch']['postprocessing'])

    def get_delta_processor(self, utterance):
        """Instanciates and returns a delta processor"""
        return self.get_processor_class('delta')(**self.config['delta'])

    def get_vtln_processor(self, utterance):
        """Instanciates and returns a VTLN processor"""
        return self.get_processor_class('vtln')(**self.config['vtln'])

    def get_warp(self, utterance):
        """Returns the VTLN warp associated to this utterance"""
        return 1 if utterance not in self.warps else self.warps[utterance]
