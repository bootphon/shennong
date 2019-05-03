#!/usr/bin/env python

import os
import urllib.request
import shennong.features.pipeline as pipeline
from shennong.utils import list_files_with_extension


ENGLISH_ITEM = (
    'https://raw.githubusercontent.com/bootphon/ABXpy/'
    'zerospeech2015/resources/english.item')

XITSONGA_ITEM = (
    'https://raw.githubusercontent.com/bootphon/ABXpy/'
    'zerospeech2015/resources/xitsonga.item')


def setup_data(data_directory, buckeye_directory, xitsonga_directory):
    """Setup a data directory with all input data required

    Creates the ``data_directory`` and make a symlink to
    ``buckeye_directory`` and ``xitsonga_directory`` in it. It also
    download the ABX item files and create the list of utterances for
    features extraction.

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
    os.symlink(buckeye_directory, os.path.join(data_directory, 'english'))
    os.symlink(xitsonga_directory, os.path.join(data_directory, 'xitsonga'))

    # download ABX item files
    urllib.request.urlretrieve(
        ENGLISH_ITEM, os.path.join(data_directory, 'english.item'))
    urllib.request.urlretrieve(
        XITSONGA_ITEM, os.path.join(data_directory, 'xitsonga.item'))

    # prepare utterances lists
    prepare_utterances_english(data_directory)
    prepare_utterances_xitsonga(data_directory)


def prepare_utterances_english(data_directory):
    wavs = list_files_with_extension(
        os.path.join(data_directory, 'english'), '.wav', abspath=True)
    utts = [os.path.splitext(os.path.basename(wav))[0] for wav in wavs]
    spks = [utt[:3] for utt in utts]

    open(os.path.join(data_directory, 'english.utts'), 'w').write(
        '\n'.join(f'{u} {w} {s}' for u, w, s in zip(utts, wavs, spks))
        + '\n')


def prepare_utterances_xitsonga(data_directory):
    wavs = list_files_with_extension(
        os.path.join(data_directory, 'xitsonga', 'audio'),
        '.wav', abspath=True)
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
        conf1 = os.path.join(conf_directory, f'{features}_only.yaml')
        yaml = pipeline.get_default_config(
            features, to_yaml=True, yaml_commented=False,
            with_cmvn=False, with_delta=False, with_pitch=False)
        open(conf1, 'w').write(yaml)
        yield conf1

        conf2 = os.path.join(conf_directory, f'{features}_no_cmvn.yaml')
        yaml = pipeline.get_default_config(
            features, to_yaml=True, yaml_commented=False,
            with_cmvn=False, with_delta=True, with_pitch=True)
        open(conf2, 'w').write(yaml)
        yield conf2

        conf3 = os.path.join(conf_directory, f'{features}_full.yaml')
        yaml = pipeline.get_default_config(
            features, to_yaml=True, yaml_commented=False,
            with_cmvn=True, with_delta=True, with_pitch=True)
        open(conf3, 'w').write(yaml)
        yield conf3


def main():
    njobs = 4
    data_directory = os.path.abspath('./data')

    buckeye_directory = '/mnt/data/databases/BUCKEYE_revised_bootphon'
    xitsonga_directory = '/mnt/data/databases/nchlt_Xitsonga'
    try:
        setup_data(data_directory, buckeye_directory, xitsonga_directory)
    except ValueError:
        pass

    configs = list(generate_configurations(
        os.path.join(data_directory, 'config')))

    try:
        os.makedirs(os.path.join(data_directory, 'features'))
    except FileExistsError:
        pass

    for corpus in ('xitsonga', ):  # ('english', 'xitsonga'):
        for config in configs:
            print(f'{corpus} {os.path.basename(config)}')
            utterances = [line.split(' ') for line in open(os.path.join(
                data_directory, f'{corpus}.utts'), 'r')]
            features = pipeline.extract_features(
                config, utterances, njobs=njobs)
            h5f_file = os.path.join(
                data_directory, 'features',
                f'{corpus}_{os.path.basename(config)}'.replace(
                    '.yaml', '.h5f'))
            features.save(h5f_file)
            del features


if __name__ == '__main__':
    main()
