#!/usr/bin/env python
"""Extract features from a wav file and plot them"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from shennong import Audio
from shennong.processor import (
    FilterbankProcessor, MfccProcessor, PlpProcessor,
    BottleneckProcessor, SpectrogramProcessor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'wav', help='wav file to compute features on')
    parser.add_argument(
        '-o', '--output-file',
        help='output file, display on screen if not specified')
    args = parser.parse_args()

    # load the wav file
    audio = Audio.load(args.wav)

    # initialize features processors
    processors = {
        'spectrogram': SpectrogramProcessor(sample_rate=audio.sample_rate),
        'filterbank': FilterbankProcessor(sample_rate=audio.sample_rate),
        'mfcc': MfccProcessor(sample_rate=audio.sample_rate),
        'plp': PlpProcessor(sample_rate=audio.sample_rate, rasta=False),
        'rastaplp': PlpProcessor(sample_rate=audio.sample_rate, rasta=True),
        'bottleneck': BottleneckProcessor(weights='BabelMulti')}

    # compute the features for all processors
    features = {k: v.process(audio) for k, v in processors.items()}

    # plot the audio signal and the resulting features
    fig, axes = plt.subplots(
        nrows=len(processors)+1,
        gridspec_kw={'top': 0.95, 'bottom': 0.05, 'hspace': 0},
        subplot_kw={'xticks': [], 'yticks': []})
    time = np.arange(0.0, audio.nsamples) / audio.sample_rate
    axes[0].plot(time, audio.astype(np.float32).data)
    axes[0].set_xlim(0.0, audio.duration)
    axes[0].text(
        0.02, 0.8, 'audio',
        bbox={'boxstyle': 'round', 'alpha': 0.5, 'color': 'white'},
        transform=axes[0].transAxes)

    for n, (k, v) in enumerate(features.items(), start=1):
        axes[n].imshow(v.data.T, aspect='auto')
        axes[n].text(
            0.02, 0.8, k,
            bbox={'boxstyle': 'round', 'alpha': 0.5, 'color': 'white'},
            transform=axes[n].transAxes)

    axes[0].set_ylabel('amplitude')
    axes[1].set_ylabel('frequency')
    for i in range(2, 6):
        axes[i].set_ylabel('mel')
    axes[-1].set_ylabel('dim')
    axes[-1].set_xlabel('time')

    fig.set_size_inches(7, 9)
    if args.output_file:
        fig.savefig(args.output_file)
    else:
        plt.show()


if __name__ == '__main__':
    main()
