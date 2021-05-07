#!/usr/bin/env python
"""VTLN training for phone discrimination experiment"""

import argparse
import pathlib

from shennong import Utterances
from shennong.processor import VtlnProcessor


def main():
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_directory', type=pathlib.Path,
        help='input/output data directory')
    parser.add_argument(
        'corpus', choices=['english', 'xitsonga'],
        help='corpus to process')
    parser.add_argument(
        '-d', '--duration', default=10*60, type=float,
        help=(
            'speech duration per speaker to use for VTLN training '
            'in seconds, (default to %(default)s)'))
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

    output_warps = data_directory / f'{args.corpus}.warps'
    if output_warps.is_file():
        raise ValueError(f'file already exists: {output_warps}')

    # load input utterances truncated to 'duration' seconds per speaker
    print(f'loading utterances from {data_directory}/{args.corpus}.utts')
    utterances = Utterances.load(
        data_directory / f'{args.corpus}.utts').fit_to_duration(
            args.duration, truncate=True, shuffle=False)

    # train VTLN model
    proc = VtlnProcessor()
    proc.set_logger('debug' if args.verbose else 'info')
    warps = {}  proc.process(utterances, njobs=args.njobs, group_by='speaker')

    # write the VTLN warps
    open(output_warps, 'w').write(
        '\n'.join(f'{s} {w}' for s, w in warps.items()) + '\n')


if __name__ == '__main__':
    main()
