"""Features extraction pipeline

The pipeline takes as input a list of wav files, extracts the features
and do the postprocessing.

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


_valid_features = ['mfcc', 'plp', 'filterbank', 'bottleneck']

_valid_processors = {
    'bottleneck': ('processor', 'BottleneckProcessor'),
    'energy': ('processor', 'EnergyProcessor'),
    'filterbank': ('processor', 'FilterbankProcessor'),
    'mfcc': ('processor', 'MfccProcessor'),
    'pitch': ('processor', 'PitchProcessor'),
    'pitch_post': ('processor', 'PitchPostProcessor'),
    'plp': ('processor', 'PlpProcessor'),
    'cmvn': ('postprocessor', 'CmvnPostProcessor'),
    'delta': ('postprocessor', 'DeltaPostProcessor'),
    'vad': ('postprocessor', 'VadPostProcessor')}


def _get_processor(name):
    """Returns the (post)processor class `name`

    This function enables dynamic import of processors classes to
    avoid a big list of useless imports

    Raises a ValueError if the `name` is not valid or the module/class
    cannot be imported

    """
    try:
        _module, _class = _valid_processors[name]
    except KeyError:
        raise ValueError('invalid processor "{}"'.format(name))
    if name == 'pitch_post':
        name = 'pitch'

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


def valid_features():
    """Returns the list of features that can be extracted by he pipeline"""
    return _valid_features


def get_default_config(features, to_yaml=False, yaml_commented=True,
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
        _get_processor(features)().get_params().items()
        if k not in ('sample_rate', 'htk_compat')}

    if with_pitch:
        # filter out the frame parameters, already specified for
        # the features, and sample rate
        config['pitch'] = {
            k: v for k, v
            in _get_processor('pitch')().get_params().items()
            if k not in ('frame_length', 'frame_shift', 'sample_rate')}
        config['pitch_post'] = _get_processor('pitch_post')().get_params()

    if with_cmvn:
        config['cmvn'] = {
            'by_speaker': True, 'with_vad': True}
        vad_opts = _get_processor('vad')().get_params()
        for k, v in vad_opts.items():
            config['cmvn']['vad_' + k] = v

    if with_delta:
        config['delta'] = _get_processor('delta')().get_params()

    if to_yaml:
        return _get_config_to_yaml(config, comments=yaml_commented)
    return config


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
        YAML file or a string formatted in YAML. To get a
        configuration example, see :func:`get_default_config`
    wavs_index : sequence of tuples
        The list of wav file to extract the features on.
    njobs : int, optional
        The number to subprocesses to execute in parallel, use a
        single process by default.

    """
    # intialize the pipeline configuration, the list of wav files to
    # process, instanciate the pipeline processors and make all the
    # checks to ensure all is correct
    njobs = get_njobs(njobs, log=log)
    config = _init_config(config, log=log)
    wavs_index = _init_wavs(wavs_index, log=log)

    # the list of speakers
    speakers = set(w.speaker for w in wavs_index.values())
    if speakers == {None}:
        speakers = None
    _check_speakers(config, speakers, log)

    # do all the computations
    return _extract_features(
        config, speakers, wavs_index, njobs=njobs, log=log)


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
                    'If false, do normalization by utterance, '
                    'if true do normalization by speaker')
            elif processor == 'cmvn' and param == 'with_vad':
                docstring = (
                    'If true do normalization only on frames where '
                    'voice activity has been detected, if false do not '
                    'consider voice activity for normalization')

            # special case 'vad_xxx' -> 'vad', 'xxx'
            elif processor == 'cmvn' and param.startswith('vad_'):
                docstring = _get_docstring('vad', param.replace('vad_', ''))
                docstring += ". This has no effect if 'with_vad' is false"
            else:
                docstring = _get_docstring(processor, param)

            docstring += '. Default is {}.'.format(default)
            docstring = re.sub(r'\.+', '.', docstring)

            config_commented += [
                '  # ' + w for w in textwrap.wrap(docstring, width=66)]
            config_commented.append(line)

    return '\n'.join(config_commented) + '\n'


def _get_docstring(processor, param):
    """Return sthe docstring of a given processor's parameter"""
    docstring = getattr(
        _get_processor(processor), param).__doc__ or ''
    docstring = re.sub(r'\n\n', '. ', docstring)
    docstring = re.sub(r'\n', ' ', docstring)
    docstring = re.sub(r' +', ' ', docstring).strip()
    return docstring


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
        k for k in config.keys() if k not in _valid_processors]
    if unknown_keys:
        raise ValueError(
            'invalid keys in configuration: {}'.format(
                ', '.join(unknown_keys)))

    # ensure one and only one features processor is defined in the
    # configuration
    features = [k for k in config.keys() if k in _valid_features]
    if not features:
        raise ValueError(
            'the configuration does not define any features extraction, '
            'only post-processing (must have one and only one entry of {})'
            .format(', '.join(_valid_features)))
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

    # if we have pitch or pitch_post, make sure we have the other
    if 'pitch' in config and 'pitch_post' not in config:
        raise ValueError(
            'configuration defines pitch but not pitch_post')
    if 'pitch_post' in config and 'pitch' not in config:
        raise ValueError(
            'configuration defines pitch_post but not pitch')

    # log message describing the pipeline configuration
    msg = []
    if 'pitch' in config:
        msg.append('pitch')
    if 'cmvn' in config:
        msg.append('cmvn')
    if 'delta' in config:
        msg.append('delta')
    log.info(
        'pipeline configured for %s features extraction%s',
        features[0], ' with {}'.format(', '.join(msg)) if msg else '')

    return config


wav_entry = collections.namedtuple(
    'wav_entry', ['file', 'speaker', 'tstart', 'tstop'])


def _init_wavs(wavs, log=get_logger()):
    """Returns a dict {utt_id: (wav_file, speaker_id, tstart, tstop)}

    Raises on any error, log a warning on strange but non-critical
    issues.

    """
    # guess the for format of `wavs` and ensure it is homogeneous
    wavs = list((w,) if isinstance(w, str) else w for w in wavs)
    format = set(len(w) for w in wavs)
    if not len(format) == 1:
        raise ValueError(
            'the wavs index is not homogeneous, entries have different '
            'lengths: {}'.format(', '.join(str(t) for t in format)))
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
            'duplicates found in wavs index: {}'.format(', '.join(duplicates)))

    # build a dict {utt_id: (wav_file, speaker_id, tstart, tstop)}
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
    wavs_metadata = [Audio.scan(w) for w in wavs]
    log.info(
        'get %s wav files%s, total duration: %s',
        len(wavs),
        '' if format not in (3, 5) else ' from {} speakers'.format(
            len(set(w.speaker for w in wavs_index.values()))),
        datetime.timedelta(seconds=sum(w.duration for w in wavs_metadata)))
    if not all(w.nchannels == 1 for w in wavs_metadata):
        raise ValueError('all wav files are not mono')
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
                    'timestamps are not in increasing order for {}: {} >= {}'
                    .format(wfile, tstart, tstop))

    return wavs_index


def _check_speakers(config, speakers, log):
    """Ensures the configuration is compatible with speakers information

    On any error raises a ValueError. Logs a warning if speakers
    information is provided but not used by the pipeline. If all is
    good, silently returns None.

    """
    # ensures speakers info provided if cmvn by speaker is requested
    if 'cmvn' not in config:
        cmvn_by_speaker = False
    elif not config['cmvn']['by_speaker']:
        cmvn_by_speaker = False
    else:  # config['cmvn']['by_speaker'] exists and is True
        assert config['cmvn']['by_speaker']
        cmvn_by_speaker = True

    if cmvn_by_speaker and not speakers:
        raise ValueError(
            'cmvn normalization by speaker requested '
            'but no speaker information provided')
    if not cmvn_by_speaker and speakers:
        log.warning(
            'speakers information is provided but will not be used '
            '(CMVN%s disabled)', ' by speaker' if 'cmvn' in config else '')


def _init_pipeline(config, speakers, log=get_logger()):
    """Instanciates the processors required for the pipeline"""
    # instanciate the features processor and postpipeline all in a
    # single dictionnary
    pipeline = {}

    # main features extraction
    features = [k for k in config.keys() if k in _valid_features][0]
    pipeline['features'] = _get_processor(features)(**config[features])

    # cmvn, if by speaker instanciate a cmvn processr per speaker,
    # else a single one, if vad build a energy+vad processor (mfcc and
    # plp already embeed energy as 1st column but this depends on
    # parameters raw_energy/use_energy/htk_compat, so it is safer and
    # simpler to recompute the energy from scratch anyway)
    if 'cmvn' in config:
        pipeline['cmvn'] = {**config['cmvn']}
        if config['cmvn']['by_speaker']:
            # instanciate a CMVN processor by speaker
            for speaker in speakers:
                pipeline['cmvn'][speaker] = _get_processor('cmvn')(
                    pipeline['features'].ndims)
        else:
            pipeline['cmvn']['cmvn'] = _get_processor('cmvn')(
                    pipeline['features'].ndims)
        if config['cmvn']['with_vad']:
            pipeline['cmvn']['energy'] = _get_processor('energy')(
                frame_length=pipeline['features'].frame_length,
                frame_shift=pipeline['features'].frame_shift)

            # init vad from parameters in 'cmvn' group starting with 'vad_'
            vad_params = {
                k.replace('vad_', ''): v for k, v in config['cmvn'].items()
                if k.startswith('vad_')}
            pipeline['cmvn']['vad'] = _get_processor('vad')(**vad_params)

    # delta is straightforward
    if 'delta' in config:
        pipeline['delta'] = _get_processor('delta')(**config['delta'])

    # frames for pitch are the same are those for features
    if 'pitch' in config:
        pipeline['pitch'] = _get_processor('pitch')(**config['pitch'])
        # forward framing parameters
        pipeline['pitch'].frame_shift = pipeline['features'].frame_shift
        pipeline['pitch'].frame_length = pipeline['features'].frame_length
        pipeline['pitch_post'] = _get_processor('pitch_post')(
            **config['pitch_post'])

    return pipeline


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


def _extract_features(config, speakers, wavs_index, njobs=1, log=get_logger()):
    # instanciate the pipeline components
    pipeline = _init_pipeline(config, wavs_index, log=log)

    # check the OMP_NUM_THREADS variable for parallel computations
    _check_environment(njobs, log=log)

    # verbosity level for joblib
    verbose = 8

    # cmvn : two passes. 1st with features pitch and cmvn
    # accumulation, 2nd with cmvn application and delta
    if 'cmvn' in pipeline:
        # extract features and pitch, accumulate cmvn stats
        feats = _Parallel(
            'features extraction, pass 1', log,
            n_jobs=njobs, verbose=verbose, prefer='threads')(
                joblib.delayed(_extract_pass_one)(
                    name, wav_entry, pipeline, log=log)
                for name, wav_entry in wavs_index.items())

        # apply cmvn and extract deltas
        feats = FeaturesCollection(**{k: v for k, v in _Parallel(
            'features extraction, pass 2', log,
            n_jobs=njobs, verbose=verbose, prefer='threads')(
                joblib.delayed(_extract_pass_two)(
                    name, wav_entry, feats, pitch, pipeline, log=log)
                for name, wav_entry, feats, pitch in feats)})

    # no cmvn: single pass
    else:
        feats = FeaturesCollection(**{k: v for k, v in _Parallel(
            'features extraction', log,
            n_jobs=njobs, verbose=verbose, prefer='threads')(
                joblib.delayed(_extract_single_pass)(
                    name, wav_entry, pipeline, log=log)
                for name, wav_entry in list(wavs_index.items()))})

    # TODO log a message with structure of the output features
    # (labels on columns)
    return feats


def _extract_pass_one(name, wav_entry, pipeline, log=get_logger()):
    audio = Audio.load(wav_entry.file)
    if wav_entry.tstart is not None:
        assert wav_entry.tstop > wav_entry.tstart
        audio = audio.segment([(wav_entry.tstart, wav_entry.tstop)])[0]

    # features extraction
    p = pipeline['features']
    try:
        p.sample_rate = audio.sample_rate
    except AttributeError:  # bottleneck processor use fixed sample rate
        pass
    feats = p.process(audio)

    if 'cmvn' in pipeline:
        if pipeline['cmvn']['by_speaker']:
            cmvn = pipeline['cmvn'][wav_entry.speaker]
        else:
            cmvn = pipeline['cmvn']['cmvn']

        # weight CMVN by voice activity detection (null weights on
        # non-voiced frames)
        if 'vad' in pipeline['cmvn']:
            pipeline['cmvn']['energy'].sample_rate = audio.sample_rate
            energy = pipeline['cmvn']['energy'].process(audio)
            vad = pipeline['cmvn']['vad'].process(energy)
            vad = vad.data.reshape((vad.shape[0], ))  # reshape as 1d array
        else:
            vad = None
        cmvn.accumulate(feats, weights=vad)

    if 'pitch' in pipeline:
        p1 = pipeline['pitch']
        p1.sample_rate = audio.sample_rate
        p2 = pipeline['pitch_post']
        pitch = p2.process(p1.process(audio))
    else:
        pitch = None

    return name, wav_entry, feats, pitch


def _extract_pass_two(name, wav_entry, feats, pitch, pipeline,
                      tolerance=2, log=get_logger()):
    # apply cmvn
    if 'cmvn' in pipeline:
        if pipeline['cmvn']['by_speaker']:
            p = pipeline['cmvn'][wav_entry.speaker]
        else:
            p = pipeline['cmvn']['cmvn']
        feats = p.process(feats)

    if 'delta' in pipeline:
        feats = pipeline['delta'].process(feats)

    # concatenate the pitch features to the main ones. because of
    # downsampling in pitch processing the resulting number of frames
    # can differ (the same tolerance is applied in Kaldi, see
    # the paste-feats binary)
    if pitch:
        feats._log = log
        feats = feats.concatenate(pitch, tolerance=tolerance)

    # TODO clean properties (with reference to wav, speaker, all
    # pipeline, etc...)
    return name, feats


def _extract_single_pass(name, wav_entry, pipeline, log=get_logger()):
    _, _, feats, pitch = _extract_pass_one(name, wav_entry, pipeline, log=log)
    return _extract_pass_two(
        name, wav_entry, feats, pitch, pipeline, log=log)
