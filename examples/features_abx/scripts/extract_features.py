#!/usr/bin/env python

import argparse
import os
import shennong.pipeline as pipeline
from shennong.utils import get_logger


def main():
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', help='input/output data directory')
    parser.add_argument('config_file', help='YAML configuration file')
    parser.add_argument(
        'corpus', choices=['english', 'xitsonga'], help='corpus to process')
    parser.add_argument(
        '-j', '--njobs', type=int, default=4, metavar='<int>',
        help='number of parallel jobs (default to %(default)s)')
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='increase log level')
    args = parser.parse_args()

    # check and setup arguments
    data_directory = args.data_directory
    if not os.path.isdir(data_directory):
        raise ValueError(f'directory not found: {data_directory}')
    config = args.config_file
    if not os.path.isfile(config):
        raise ValueError(f'file not found: {config}')
    try:
        os.makedirs(os.path.join(data_directory, 'features'))
    except FileExistsError:
        pass
    log = get_logger(level='debug' if args.verbose else 'info')

    # load input utterances
    utterances = [line.split(' ') for line in open(os.path.join(
        data_directory, f'{args.corpus}.utts'), 'r')]

    # extract the features
    features = pipeline.extract_features(
        config, utterances, njobs=args.njobs, log=log)

    # save them
    h5f_file = os.path.join(
        data_directory, 'features', f'{args.corpus}_{os.path.basename(config)}'
        .replace('.yaml', '.h5f'))
    features.save(h5f_file)


if __name__ == '__main__':
    main()
