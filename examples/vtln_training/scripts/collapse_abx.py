#!/usr/bin/env python

import argparse
import pathlib
import pandas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('score_file', type=pathlib.Path)

    # load data from csv
    csv = parser.parse_args().score_file
    data = pandas.read_csv(csv, sep='\t')

    # aggregate on talkers
    groups = data.groupby(['by', 'phone_1', 'phone_2'], as_index=False)
    data = groups['score'].mean()

    # aggregate on contexts
    groups = data.groupby(['phone_1', 'phone_2'], as_index=False)
    data = groups['score'].mean()

    # final ABX score in %
    print((1 - data.mean()[0]) * 100)


if __name__ == '__main__':
    main()
