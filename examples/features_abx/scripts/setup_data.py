#!/usr/bin/env python
"""Data preparation for phone discrimination experiment"""

import argparse
import pathlib
import tempfile
import urllib.request
import yaml as pyyaml

import shennong.pipeline as pipeline
from shennong.logger import get_logger


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

log = get_logger('data setup', 'info')


def setup_data(data_directory, buckeye_directory, xitsonga_directory):
    """Setup a data directory with all input data required

    * creates the ``data_directory``
    * make a symlink to ``buckeye_directory`` and ``xitsonga_directory`` in it
    * download the ABX item files for buckeye and xitsonga
    * create the list of utterances for both corpora
    * create the configuration files for features extraction

    """
    # basic checks
    if not buckeye_directory.is_dir():
        raise ValueError(f'directory does not exists: {buckeye_directory}')
    if not xitsonga_directory.is_dir():
        raise ValueError(f'directory does not exists: {xitsonga_directory}')

    # resolve directories to absolute paths
    buckeye_directory = buckeye_directory.resolve()
    xitsonga_directory = xitsonga_directory.resolve()

    # create the data directory
    data_directory.mkdir(parents=True, exist_ok=True)

    # create symlinks to speech corpora
    log.info('symlinking corpora directory...')
    (data_directory / 'english').symlink_to(buckeye_directory)
    (data_directory / 'xitsonga').symlink_to(xitsonga_directory)

    # download ABX item files
    log.info('downloading ABX item files...')
    urllib.request.urlretrieve(ENGLISH_ITEM, data_directory / 'english.item')
    urllib.request.urlretrieve(XITSONGA_ITEM, data_directory / 'xitsonga.item')

    # prepare utterances lists
    log.info('creating utterances lists...')
    prepare_utterances_english(data_directory)
    prepare_utterances_xitsonga(data_directory)

    log.info('generating configuration files for features extraction...')
    generate_configurations(data_directory / 'config')


def prepare_utterances_english(data_directory):
    with tempfile.NamedTemporaryFile() as temp:
        urllib.request.urlretrieve(ENGLISH_FILES_LIST, temp.name)
        files_list = {f.strip().decode() for f in temp}

    wavs = [
        wav.resolve() for wav in (data_directory / 'english').glob('**/*.wav')
        if wav.name in files_list]
    assert len(wavs) == len(files_list)

    utts = [wav.stem for wav in wavs]
    spks = [utt[:3] for utt in utts]

    open(data_directory / 'english.utts', 'w').write(
        '\n'.join(f'{u} {w} {s}' for u, w, s in zip(utts, wavs, spks)) + '\n')


def prepare_utterances_xitsonga(data_directory):
    with tempfile.NamedTemporaryFile() as temp:
        urllib.request.urlretrieve(XITSONGA_FILES_LIST, temp.name)
        files_list = {f.strip().decode() for f in temp}

    wavs = [
        wav.resolve() for wav in (
            data_directory / 'xitsonga' / 'audio').glob('**/*.wav')
        if wav.name in files_list]
    assert len(wavs) == len(files_list)

    utts = [wav.stem for wav in wavs]
    spks = [utt.split('_')[2].lstrip('0').replace('m', '').replace('f', '')
            for utt in utts]

    open(data_directory / 'xitsonga.utts', 'w').write(
        '\n'.join(f'{u} {w} {s}' for u, w, s in zip(utts, wavs, spks)) + '\n')


def generate_configurations(conf_directory):
    conf_directory.mkdir(parents=True, exist_ok=True)

    for features in pipeline.valid_features():
        yaml = pipeline.get_default_config(
            features, to_yaml=True, yaml_commented=False,
            with_cmvn=False, with_delta=False, with_pitch=False)
        open(conf_directory / f'{features}_only.yaml', 'w').write(yaml)

        yaml = pipeline.get_default_config(
            features, to_yaml=True, yaml_commented=False,
            with_cmvn=False, with_delta=True, with_pitch='kaldi')
        open(conf_directory / f'{features}_nocmvn.yaml', 'w').write(yaml)

        yaml = pipeline.get_default_config(
            features, to_yaml=True, yaml_commented=False,
            with_cmvn=True, with_delta=True, with_pitch='kaldi')
        open(conf_directory / f'{features}_full.yaml', 'w').write(yaml)

    # rastaplp
    for conf in ('only', 'nocmvn', 'full'):
        filename = conf_directory / f'plp_{conf}.yaml'
        yaml = pyyaml.safe_load(open(filename, 'r'))
        yaml['plp']['rasta'] = True
        open(str(filename).replace('plp', 'rastaplp'), 'w').write(
            pyyaml.safe_dump(yaml))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_directory', type=pathlib.Path,
        help='directory being created')
    parser.add_argument(
        'buckeye_directory', type=pathlib.Path,
        help='path to Buckeye corpus')
    parser.add_argument(
        'xitsonga_directory', type=pathlib.Path,
        help='path to Xitsonga corpus')
    args = parser.parse_args()

    setup_data(
        args.data_directory,
        args.buckeye_directory,
        args.xitsonga_directory)


if __name__ == '__main__':
    main()
