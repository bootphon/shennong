#!/usr/bin/env python
"""Extract features from a wav file and plot them"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from shennong.audio import AudioData
from shennong.features.filterbank import FilterbankProcessor
from shennong.features.mfcc import MfccProcessor
from shennong.features.plp import PlpProcessor
from shennong.features.bottleneck import BottleneckProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wav', help='wav file to compute features on')

    # load the WAV file
    wav_file = parser.parse_args().wav
    audio = AudioData.load(wav_file)

    # initialize features processors
    processors = {
        'mfcc': MfccProcessor(
            frame_shift=0.015, frame_length=0.025),
        'plp': PlpProcessor(
            frame_shift=0.015, frame_length=0.025),
        'filterbank': FilterbankProcessor(
            frame_shift=0.015, frame_length=0.025),
        'bottleneck': BottleneckProcessor(
            weights='FisherTri')}

    # compute the features for all processors
    features = {k: v.process(audio) for k, v in processors.items()}

    # plot the results
    fig, axes = plt.subplots(nrows=len(processors)+1)

    axes[0].plot(
        np.arange(0.0, audio.nsamples) / audio.sample_rate, audio.data)
    axes[0].set_title('audio')

    for n, (k, v) in enumerate(features.items(), start=1):
        axes[n].imshow(v.data.T, aspect='auto')
        axes[n].set_title(k)

    fig.suptitle(wav_file)
    plt.show()


if __name__ == '__main__':
    main()
