#!/usr/bin/env python

import argparse
import os
import tempfile
import urllib.request
import shennong.features.pipeline as pipeline
from shennong.utils import list_files_with_extension, get_logger


ENGLISH_ITEM = (
    'https://raw.githubusercontent.com/bootphon/ABXpy/'
    'zerospeech2015/resources/english.item')

XITSONGA_ITEM = (
    'https://raw.githubusercontent.com/bootphon/ABXpy/'
    'zerospeech2015/resources/xitsonga.item')

ENGLISH_FILES_LIST = (
    'https://raw.githubusercontent.com/bootphon/'
    'Zerospeech2015/master/english_files.txt')

XITSONGA_FILES_LIST = (
    'https://raw.githubusercontent.com/bootphon/'
    'Zerospeech2015/master/xitsonga_files.txt')

log = get_logger(level='info')


def setup_data(data_directory, buckeye_directory, xitsonga_directory):
    """Setup a data directory with all input data required

    * creates the ``data_directory``
    * make a symlink to ``buckeye_directory`` and ``xitsonga_directory`` in it
    * download the ABX item files for buckeye and xitsonga
    * create the list of utterances both corpora
    * create the configuration files  for features extraction

    """
    # basic checks
    if not os.path.isdir(buckeye_directory):
        raise ValueError(f'directory does not exists: {buckeye_directory}')
    if not os.path.isdir(xitsonga_directory):
        raise ValueError(f'directory does not exists: {xitsonga_directory}')

    # create the data directory
    if os.path.exists(data_directory):
        raise ValueError(f'directory already exists: {data_directory}')
    os.makedirs(data_directory)

    # create symlinks to speech corpora
    log.info('symlinking corpora directory...')
    os.symlink(buckeye_directory, os.path.join(data_directory, 'english'))
    os.symlink(xitsonga_directory, os.path.join(data_directory, 'xitsonga'))

    # download ABX item files
    log.info('downloading ABX item files...')
    urllib.request.urlretrieve(
        ENGLISH_ITEM, os.path.join(data_directory, 'english.item'))
    urllib.request.urlretrieve(
        XITSONGA_ITEM, os.path.join(data_directory, 'xitsonga.item'))

    # prepare utterances lists
    log.info('creating utterances lists...')
    prepare_utterances_english(data_directory)
    prepare_utterances_xitsonga(data_directory)

    log.info('generating configuration files for features extraction...')
    generate_configurations(os.path.join(data_directory, 'config'))


def prepare_utterances_english(data_directory):
    with tempfile.NamedTemporaryFile() as temp:
        urllib.request.urlretrieve(ENGLISH_FILES_LIST, temp.name)
        files_list = {f.strip().decode() for f in temp}

    wavs = [wav for wav in list_files_with_extension(
        os.path.join(data_directory, 'english'), '.wav', abspath=True)
            if os.path.basename(wav) in files_list]
    assert len(wavs) == len(files_list)

    utts = [os.path.splitext(os.path.basename(wav))[0] for wav in wavs]
    spks = [utt[:3] for utt in utts]

    open(os.path.join(data_directory, 'english.utts'), 'w').write(
        '\n'.join(f'{u} {w} {s}' for u, w, s in zip(utts, wavs, spks))
        + '\n')


def prepare_utterances_xitsonga(data_directory):
    with tempfile.NamedTemporaryFile() as temp:
        urllib.request.urlretrieve(XITSONGA_FILES_LIST, temp.name)
        files_list = {f.strip().decode() for f in temp}

    wavs = [wav for wav in list_files_with_extension(
        os.path.join(data_directory, 'xitsonga', 'audio'),
        '.wav', abspath=True) if os.path.basename(wav) in files_list]
    assert len(wavs) == len(files_list)

    utts = [os.path.splitext(os.path.basename(wav))[0] for wav in wavs]
    spks = [utt.split('_')[2].lstrip('0').replace('m', '').replace('f', '')
            for utt in utts]

    open(os.path.join(data_directory, 'xitsonga.utts'), 'w').write(
        '\n'.join(f'{u} {w} {s}' for u, w, s in zip(utts, wavs, spks))
        + '\n')


def generate_configurations(conf_directory):
    try:
        os.makedirs(conf_directory)
    except FileExistsError:
        pass

    for features in pipeline.valid_features():
        conf = os.path.join(conf_directory, f'{features}_only.yaml')
        yaml = pipeline.get_default_config(
            features, to_yaml=True, yaml_commented=False,
            with_cmvn=False, with_delta=False, with_pitch=False)
        open(conf, 'w').write(yaml)

        conf = os.path.join(conf_directory, f'{features}_no_cmvn.yaml')
        yaml = pipeline.get_default_config(
            features, to_yaml=True, yaml_commented=False,
            with_cmvn=False, with_delta=True, with_pitch=True)
        open(conf, 'w').write(yaml)

        conf = os.path.join(conf_directory, f'{features}_full.yaml')
        yaml = pipeline.get_default_config(
            features, to_yaml=True, yaml_commented=False,
            with_cmvn=True, with_delta=True, with_pitch=True)
        open(conf, 'w').write(yaml)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', help='directory being created')
    parser.add_argument('buckeye_directory', help='path to Buckeye corpus')
    parser.add_argument('xitsonga_directory', help='path to Xitsonga corpus')
    args = parser.parse_args()

    setup_data(
        args.data_directory, args.buckeye_directory, args.xitsonga_directory)


if __name__ == '__main__':
    main()
