#!/usr/bin/env python
"""Computes MFCC features from wavs in a directory"""

import argparse
import joblib
import os

from shennong.audio import AudioData
from shennong.features.features import FeaturesCollection
from shennong.features.mfcc import MfccProcessor
from shennong.utils import list_files_with_extension, get_logger


def list_wav(data_dir):
    wavs = list_files_with_extension(data_dir, '.wav', abspath=True)
    return {os.path.splitext(os.path.basename(wav))[0]: wav for wav in wavs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_dir', help='directory with wavs')
    parser.add_argument(
        'out_file', nargs='?', default='./features.h5f',
        help='file to save the computed features, default to %(default)s')
    parser.add_argument(
        '-j', '--njobs', type=int, default=1,
        help='number of parallel jobs to use, default to %(default)s')
    parser.add_argument(
        '--sample-rate', type=int, default=16000,
        help='sample rate of the wav files (must all be the same), '
        'default to %(default)s')
    args = parser.parse_args()

    log = get_logger(os.path.basename(__file__))

    # compute the list of wav files on which to estimate speech features
    wavs = list_wav(args.data_dir)
    log.info('found %s wav files', len(wavs))
    if not wavs:
        return

    if os.path.exists(args.out_file):
        log.error('output file already exist: %s', args.out_file)
        return

    # computes MFCC with default arguments and save them to disk
    processor = MfccProcessor(sample_rate=args.sample_rate)
    features = processor.process_all(wavs, njobs=args.njobs)
    features.save(args.out_file)


if __name__ == '__main__':
    main()
