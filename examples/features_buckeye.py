#!/usr/bin/env python
"""Computes some features on the Buckeye corpus

Buckeye corpus is available for free at https://buckeyecorpus.osu.edu/

"""

import argparse
import os

from shennong.audio import AudioData
from shennong.features.features import FeaturesCollection
from shennong.features.mfcc import MfccProcessor
from shennong.utils import list_files_with_extension


def list_wav(data_dir):
    wavs = list_files_with_extension(data_dir, '.wav', abspath=True)
    return {os.path.splitext(os.path.basename(wav))[0]: wav for wav in wavs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_dir', help='Directory with the Buckeye corpus distribution')
    args = parser.parse_args()

    processor = MfccProcessor()
    features = FeaturesCollection()

    wavs = list_wav(args.data_dir)
    for wav_name, wav_file in wavs.items():
        print('processing {}'.format(wav_name))
        audio = AudioData.load(wav_file)
        feat = processor.process(audio)
        features[wav_name] = feat


if __name__ == '__main__':
    main()
