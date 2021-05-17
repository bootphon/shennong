#!/usr/bin/env python
"""Plots ABX scores for various VTLN training durations"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas


LABEL = {
    'only': 'raw',
    'nocmvn': r'+$\Delta$/F0',
    'full': '+CMVN'}


def update_duration(value):
    if value == 'all':
        return 660
    if value == 'off':
        return 0
    return float(value)


def get_curve(data, conf):
    data = data[data.conf == conf].groupby('duration')
    return (
        data['score'].mean().to_numpy(),
        data['score'].std().fillna(0).to_numpy() / 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_directory', type=pathlib.Path, help='data directory')

    data_directory = parser.parse_args().data_directory
    assert data_directory.is_dir()

    plt.style.use(
        data_directory.resolve().parent.parent / 'plot.style')
    plt.figure(figsize=(6, 4))
    plt.grid(axis='both')

    data = pandas.read_csv(data_directory / 'abx.csv')
    data['duration'] = data['duration'].apply(update_duration)
    xticks = sorted(data['duration'].unique())

    for conf in 'only', 'nocmvn', 'full':
        mean, hstd = get_curve(data, conf)
        plt.plot(xticks, mean, marker='.', label=LABEL[conf])
        plt.fill_between(xticks, mean - hstd, mean + hstd, alpha=0.15)

    plt.xticks(
        [0, 100, 200, 300, 400, 500, 600, 660],
        [0, 100, 200, 300, 400, 500, 600, 'all'])
    plt.xlabel('duration per speaker (s)')
    plt.ylabel(r'ABX error rate (\%)')
    plt.legend()

    (data_directory / 'plots').mkdir(exist_ok=True)
    plt.savefig(data_directory / 'plots' / 'vtln_abx.pdf')
    # plt.show()


if __name__ == '__main__':
    main()
