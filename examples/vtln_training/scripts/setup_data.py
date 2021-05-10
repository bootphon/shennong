#!/usr/bin/env python
"""Data preparation for the VTLN training experiment"""

import argparse
import pathlib
import random
import tempfile
import urllib.request

import shennong.logger
import shennong.pipeline
from shennong import Utterance, Utterances


DURATIONS = [5, 10, 20, 30, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600]
"""Duration per speaker in seconds for VTLN training"""


ENGLISH_ITEM = (
    'https://raw.githubusercontent.com/bootphon/ABXpy/'
    'zerospeech2015/resources/english.item')

ENGLISH_FILES_LIST = (
    'https://raw.githubusercontent.com/bootphon/'
    'Zerospeech2015/master/english_files.txt')


log = shennong.logger.get_logger('vtln training', level='info')


def setup_data(data_directory, buckeye_directory, slurm_jobs=10):
    """Setup a data directory with all input data required

    * creates the ``data_directory``
    * make a symlink to ``buckeye_directory`` in it
    * download the ABX item file
    * create the list of utterances
    * prepare utterances segments for VTLN training
    * generates ``slurm_jobs`` chunks of those segments for VTLN training
    * create the configuration files for features extraction

    """
    # basic checks
    if not buckeye_directory.is_dir():
        raise ValueError(f'directory does not exists: {buckeye_directory}')

    data_directory.mkdir(exist_ok=True, parents=True)

    log.info('symlinking corpora directory...')
    (data_directory / 'english').symlink_to(buckeye_directory)

    log.info('downloading ABX item files...')
    urllib.request.urlretrieve(ENGLISH_ITEM, data_directory / 'english.item')

    log.info('creating utterances lists...')
    prepare_utterances(data_directory)

    log.info('creating utterances segments...')
    prepare_segments(data_directory)

    log.info('prepare VTLN jobs')
    prepare_jobs(data_directory, slurm_jobs)

    log.info('generating configuration files for features extraction...')
    prepare_configurations(data_directory / 'config')


def trim_utterance(utt, segment, duration):
    if utt.name in segment.by_name():
        # part of the utterance has been consumed
        if utt.tstart + duration < utt.tstop:
            return Utterance(
                utt.name, utt.audio_file, utt.speaker,
                utt.tstart + duration, utt.tstop)
        # the entire utterance has been consumed
        return None
    # the utterance has not been consumed
    return utt


def segment_utterances(utterances, duration):
    """Yields Utterances with `duration` seconds per speaker

    All the consecutive segments of equal `duration` are considered until there
    is no data left for at least one speaker.

    """
    while True:
        try:
            segment = utterances.fit_to_duration(
                duration, truncate=False, shuffle=False)
        except ValueError:
            return

        utterances = Utterances(utt for utt in (
            trim_utterance(utt, segment, duration)
            for utt in utterances) if utt)

        yield segment


def prepare_utterances(data_directory):
    with tempfile.NamedTemporaryFile() as temp:
        urllib.request.urlretrieve(ENGLISH_FILES_LIST, temp.name)
        files_list = {f.strip().decode() for f in temp}

    wavs = [
        wav for wav in (data_directory / 'english').glob('**/s*.wav')
        if wav.name in files_list]
    assert len(wavs) == len(files_list)

    utts = [wav.stem for wav in wavs]
    spks = [utt[:3] for utt in utts]

    open(data_directory / 'english.utts', 'w').write(
        '\n'.join(f'{u} {w} {s}' for u, w, s in zip(utts, wavs, spks)) + '\n')


def prepare_segments(data_directory):
    # load input utterances
    utterances = Utterances([
        Utterance(u.name, u.audio_file, u.speaker, 0, u.duration)
        for u in Utterances.load(data_directory / 'english.utts')])

    # prepare output directory
    (data_directory / 'segments').mkdir(exist_ok=True)

    # generate all the segments (about 1000)
    utterances.save(data_directory / 'segments' / 'all.utt')
    for duration in DURATIONS:
        for i, segment in enumerate(
                segment_utterances(utterances, duration), start=1):
            segment.save(
                data_directory / 'segments' /
                f'{str(duration).rjust(3, "0")}_{str(i).rjust(3, "0")}.utt')


def prepare_jobs(data_directory, slurm_jobs):
    # group the 1000 and so segments into `slurm_njobs` jobs
    segments = list(
        f.resolve() for f in (data_directory / 'segments').glob('*.utt'))
    random.shuffle(segments)

    size = int(len(segments) / (slurm_jobs - 1))
    jobs = [segments[pos:pos+size] for pos in range(0, len(segments), size)]
    with open(data_directory / 'vtln_jobs.txt', 'w') as fout:
        for i, job in enumerate(jobs, start=1):
            for line in sorted(job):
                fout.write(f'{i} {line}\n')


def prepare_configurations(conf_directory):
    conf_directory.mkdir(exist_ok=True)

    yaml = shennong.pipeline.get_default_config(
        'mfcc', to_yaml=True, yaml_commented=False,
        with_cmvn=False, with_delta=False, with_pitch=False)
    open(conf_directory / 'mfcc_only.yaml', 'w').write(yaml)

    yaml = shennong.pipeline.get_default_config(
        'mfcc', to_yaml=True, yaml_commented=False,
        with_cmvn=False, with_delta=True, with_pitch='kaldi')
    open(conf_directory / 'mfcc_nocmvn.yaml', 'w').write(yaml)

    yaml = shennong.pipeline.get_default_config(
        'mfcc', to_yaml=True, yaml_commented=False,
        with_cmvn=True, with_delta=True, with_pitch='kaldi')
    open(conf_directory / 'mfcc_full.yaml', 'w').write(yaml)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_directory', type=pathlib.Path, help='directory being created')
    parser.add_argument(
        'buckeye_directory', type=pathlib.Path, help='path to Buckeye corpus')
    parser.add_argument(
        '-n', '--slurm-jobs', type=int, default=10,
        help='number of VTLN training jobs to generate')
    args = parser.parse_args()

    setup_data(args.data_directory, args.buckeye_directory, args.slurm_jobs)


if __name__ == '__main__':
    main()
