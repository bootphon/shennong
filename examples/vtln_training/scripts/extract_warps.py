#!/usr/bin/env python
"""Extract VTLN warps for various durations

Read data/segments/{file}.utt and write data/wraps/{file}.warp

"""

import argparse
import pathlib

from shennong import Utterances
from shennong.processor import VtlnProcessor
from shennong.utils import get_njobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'utterances_file', type=pathlib.Path, help='path to the utterances')
    parser.add_argument(
        '-j', '--njobs', type=int, default=get_njobs(),
        help='number of parallel jobs, default to %(default)s')
    args = parser.parse_args()

    output_file = str(args.utterances_file).replace(
        'segments', 'warps').replace('.utt', '.warp')
    pathlib.Path(output_file).parent.mkdir(exist_ok=True)

    utterances = Utterances.load(args.utterances_file)
    processor = VtlnProcessor()
    processor.set_logger('info')
    warps = processor.process(utterances, njobs=args.njobs, group_by='speaker')

    with open(output_file, 'w') as fout:
        for spk, warp in sorted(warps.items()):
            fout.write(f'{spk} {warp}\n')


if __name__ == '__main__':
    main()
