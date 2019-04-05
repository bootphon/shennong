"""Features extraction pipeline

The pipeline takes as input a list of wav files, extracts the features
and do the postprocessing.

"""

import numpy as np
import os
import re
import textwrap
import yaml

from shennong.audio import AudioData
from shennong.features.processor.bottleneck import BottleneckProcessor
from shennong.features.processor.filterbank import FilterbankProcessor
from shennong.features.processor.mfcc import MfccProcessor
from shennong.features.processor.plp import PlpProcessor
from shennong.features.processor.pitch import (
    PitchProcessor, PitchPostProcessor)
from shennong.features.postprocessor.cmvn import CmvnPostProcessor
from shennong.features.postprocessor.delta import DeltaPostProcessor
from shennong.utils import get_logger, get_njobs


_log = None

_valid_features = ['mfcc', 'filterbank', 'plp', 'bottleneck']

_valid_processors = {
    'bottleneck': BottleneckProcessor,
    'filterbank': FilterbankProcessor,
    'mfcc': MfccProcessor,
    'pitch': PitchProcessor,
    'pitch_post': PitchPostProcessor,
    'plp': PlpProcessor,
    'cmvn': CmvnPostProcessor,
    'delta': DeltaPostProcessor}


def extract_features(config, wavs, wav2spk=None, njobs=1, log=get_logger()):
    _log = log
    njobs = get_njobs(njobs, log=_log)
    config = _init_config(config)
    wavs = _init_wavs(wavs, wav2spk)  # dict {name: (file, speaker)}


def _init_config(config):
    if os.path.isfile(config):
        _log.debug('loading configuration from %s', config)
        config = open(config, 'r')

    if isinstance(config, str):
        # the config is a string, try to load it as a YAML
        try:
            config = yaml.load(config)
        except yaml.YAMLError as err:
            raise ValueError('Error in configuration file: {}', str(err))

        # ensures all the keys in config are known
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

        return config


def default_config(cls, features, to_yaml=False, yaml_commented=True,
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
        the YAML string, if False do nothing. Yhis option has an
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
    if features not in cls.valid_features():
        raise ValueError('invalid features "{}", must be in {}'.format(
            features, ', '.join(cls.valid_features())))

    config = {}

    # filter out sample rate parameter because it is dependent of
    # the input wav file
    config[features] = {
        k: v for k, v in
        cls._processors[features]().get_params().items()
        if k not in ('sample_rate', 'htk_compat')}

    if with_pitch:
        # filter out the frame parameters, already specified for
        # the features, and sample rate
        config['pitch'] = {
            k: v for k, v
            in cls._processors['pitch']().get_params().items()
            if k not in ('frame_length', 'frame_shift', 'sample_rate')}
        config['pitch_post'] = cls._processors['pitch_post']().get_params()

    if with_cmvn:
        config['cmvn'] = {'by_speaker': True, 'with_vad': True}

    if with_delta:
        config['delta'] = cls._processors['delta']().get_params()

    if to_yaml:
        return cls._get_config_to_yaml(config, comments=yaml_commented)
    return config


def _get_config_to_yaml(cls, config, comments=True):
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
    config = yaml.dump(config).strip().split('\n')

    if not comments:
        return config

    # incrust the parameters docstings as comments in the yaml
    config_commented = []
    processor = None
    for line in config:
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
                    cls._processors[processor], param).__doc__ or ''
                docstring = re.sub(r'\n\n', '. ', docstring)
                docstring = re.sub(r'\n', ' ', docstring)
                docstring = re.sub(r' +', ' ', docstring).strip()

            docstring += '. Default is {}'.format(default)
            docstring = re.sub(r'\.+', '.', docstring)

            config_commented += [
                '  # ' + w for w in textwrap.wrap(docstring, width=66)]
            config_commented.append(line)

    return '\n'.join(config_commented)
