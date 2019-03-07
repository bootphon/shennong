#!/usr/bin/env python
"""Extract features from a wav file and save them to disk"""

import argparse

from shennong.audio import AudioData
from shennong.features import FeaturesCollection
from shennong.features.mfcc import MfccProcessor
from shennong.features.handlers import NumpyHandler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wav', help='wav file to compute features on')
    parser.add_argument('out', help='features file to create')
    args = parser.parse_args()

    # load the wav file
    audio = AudioData.load(args.wav)

    # compute the features for all processors
    features = FeaturesCollection(
        {'test': MfccProcessor(
            sample_rate=audio.sample_rate).process(audio),
         'test2': MfccProcessor(
             use_energy=False, window_type='hanning',
             sample_rate=audio.sample_rate).process(audio)})

    features.save(args.out)


if __name__ == '__main__':
    main()
