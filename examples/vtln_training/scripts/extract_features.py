#!/usr/bin/env python

import argparse
import pathlib
from shennong import pipeline, Utterances
from shennong.logger import get_logger


def main():
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_directory', type=pathlib.Path,
        help='input/output data directory')
    parser.add_argument(
        'conf', choices=['only', 'nocmvn', 'full'],
        help='pipeline configuration')
    parser.add_argument(
        'warps', type=pathlib.Path, help='VTLN warps to use')
    parser.add_argument(
        '-o', '--output-file', type=pathlib.Path, help='features file')
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

    config = data_directory / 'config' / f'mfcc_{args.conf}.yaml'
    if not config.is_file():
        raise ValueError(f'file not found: {config}')

    warps = None
    if args.warps.is_file():
        warps = {spk: float(warp) for spk, warp in (
            line.strip().split() for line in open(args.warps, 'r'))}
    else:
        # the case without VTLN
        assert str(args.warps) == 'off'

    log = get_logger('extraction', 'debug' if args.verbose else 'info')

    # load input utterances
    log.info('loading utterances...')
    utterances = Utterances.load(data_directory / 'english.utts')

    # extract the features
    features = pipeline.extract_features(
        config, utterances, warps=warps, njobs=args.njobs, log=log)

    # save them
    (args.output_file.parent).mkdir(exist_ok=True, parents=True)
    features.save(args.output_file)


if __name__ == '__main__':
    main()
