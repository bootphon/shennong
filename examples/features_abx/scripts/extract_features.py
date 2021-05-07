#!/usr/bin/env python

import argparse
import pathlib
from shennong import pipeline, Utterances
from shennong.logger import get_logger


def main():
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_directory', help='input/output data directory', type=pathlib.Path)
    parser.add_argument(
        'config_file', help='YAML configuration file', type=pathlib.Path)
    parser.add_argument(
        'corpus', choices=['english', 'xitsonga'], help='corpus to process')
    parser.add_argument(
        '--do-vtln', action='store_true',
        help='extract warped features from pre-trained VTLN')
    parser.add_argument(
        '-j', '--njobs', type=int, default=4, metavar='<int>',
        help='number of parallel jobs (default to %(default)s)')
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='increase log level')
    args = parser.parse_args()

    # check and setup arguments
    data_directory = args.data_directory
    if not data_directory.is_dir():
        raise ValueError(f'directory not found: {data_directory}')

    config = args.config_file
    if not config.is_file():
        raise ValueError(f'file not found: {config}')

    warps = None
    if args.do_vtln:
        warps_file = data_directory / f'{args.corpus}.warps'
        if not warps_file.is_file():
            raise ValueError(f'file not found: {warps_file}')
        warps = {spk: float(warp) for spk, warp in (
            line.strip().split() for line in open(warps_file, 'r'))}

    (data_directory / 'features').mkdir(exist_ok=True)

    log = get_logger('extraction', 'debug' if args.verbose else 'info')

    # load input utterances
    log.info('loading utterances...')
    utterances = Utterances(
        [line.strip().split(' ') for line in open(
            data_directory / f'{args.corpus}.utts', 'r')])

    # extract the features
    features = pipeline.extract_features(
        config, utterances, warps=warps, njobs=args.njobs, log=log)

    # save them
    h5f_file = data_directory / 'features' / f'{args.corpus}_{config.stem}.h5f'
    if args.do_vtln:
        h5f_file = h5f_file.replace('.h5f', '_vtln.h5f')

    features.save(h5f_file)


if __name__ == '__main__':
    main()
