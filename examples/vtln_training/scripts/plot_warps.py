#!/usr/bin/env python
"""Plots warps on various training durations for multiple speakers"""

import argparse
import collections
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas


def parse_warps(warps):
    """Yields (speaker, duration, warp) from a collection of warp files"""
    for warp in warps:
        duration = warp.stem.split('_')[0]
        for line in (line.strip() for line in open(warp, 'r')):
            if line:
                line = line.split(' ')
                yield (line[0], duration, float(line[1]))


def parse_rows(data):
    """Yields (speaker, duration, mean, std, size)

    From a dict {speaker: {duration: list}}

    """
    for speaker, subdata in data.items():
        for duration, warps in subdata.items():
            warps = np.asarray(warps)
            yield speaker, duration, warps.mean(), warps.std(), warps.shape[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_directory', type=pathlib.Path, help='data directory')

    data_directory = parser.parse_args().data_directory
    assert data_directory.is_dir()

    # first ensure all the expected warps are here. Some segments are not
    # computed because there is not enough data on some utterances, yielding
    # the VTLN processor to exit with an error message. We could have fixed
    # that bug by constraining a minimal utterance duration in
    # setup_data.py::prepare_segments but due to huge number of warps already
    # available this is not a real issue here.
    warps = sorted((data_directory / 'warps').glob('*.warp'))
    missing = sorted(
        {s.stem for s in (data_directory / 'segments').glob('*.utt')} -
        {w.stem for w in warps})
    assert missing == [
        '005_234', '005_471', '005_472', '005_473', '005_474',
        '005_475', '005_476', '005_477', '005_478', '005_479',
        '010_236', '010_237', '010_238', '010_239', '020_118']

    # load all the warps and compute mean and std per (speaker, duration)
    # combination, save it as a csv
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    for speaker, duration, warp in parse_warps(warps):
        data[speaker][duration].append(warp)
    data = pandas.DataFrame(
        columns=('speaker', 'duration', 'mean', 'std', 'size'),
        data=parse_rows(data))
    data.to_csv(data_directory / 'warps.csv', index=False)

    durations = [d.lstrip('0') for d in data['duration'].unique()]
    xticks = [int(d) for d in durations[:-1]] + [660]

    # keep a representative subset of speakers
    speakers = [s for s in data['speaker'].unique()
                if s in ('s20', 's23', 's24', 's27', 's30', 's31', 's33')]

    plt.style.use(
        data_directory.resolve().parent.parent / 'plot.style')
    plt.figure(figsize=(6, 4))
    plt.grid(axis='both')

    for speaker in speakers:
        mean = data[data['speaker'] == speaker]['mean']
        hstd = data[data['speaker'] == speaker]['std'] / 2
        plt.plot(xticks, mean, marker='.')
        plt.fill_between(xticks, mean - hstd, mean + hstd, alpha=0.15)

    plt.xticks(
        [0, 100, 200, 300, 400, 500, 600, 660],
        [0, 100, 200, 300, 400, 500, 600, 'all'])
    plt.xlabel('duration per speaker (s)')
    plt.ylabel('VTLN warp')

    (data_directory / 'plots').mkdir(exist_ok=True)
    plt.savefig(data_directory / 'plots' / 'vtln_coefficients.pdf')
    # plt.show()


if __name__ == '__main__':
    main()
