#!/usr/bin/env python

import argparse
import os

from shennong.processor import VtlnProcessor
from shennong.utils import duration_per_speaker


def main():
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', help='input/output data directory')
    parser.add_argument(
        'corpus', choices=['english', 'xitsonga'], help='corpus to process')
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
    if not os.path.isdir(data_directory):
        raise ValueError(f'directory not found: {data_directory}')
    output_warps = f'{data_directory}/{args.corpus}.warps'
    if os.path.isfile(output_warps):
        raise ValueError(f'file already exists: {output_warps}')

    # load input utterances
    utterances = [line.split(' ') for line in open(os.path.join(
        data_directory, f'{args.corpus}.utts'), 'r')]

    # build utterances for VTLN training
    utterances = duration_per_speaker(
        utterances, args.duration, truncate=False)

    # train VTLN model
    proc = VtlnProcessor()
    proc.set_logger('debug' if args.verbose else 'info')
    warps = proc.process(utterances, njobs=args.njobs, group_by='speaker')

    # write the VTLN warps
    open(output_warps, 'w').write(
        '\n'.join(f'{s} {w}' for s, w in warps.items()) + '\n')



if __name__ == '__main__':
    main()
