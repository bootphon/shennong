#!/usr/bin/env python
"""Generate jobs for MFCC extraction with various warps"""

import argparse
import pathlib
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('warps_directory', type=pathlib.Path)
    parser.add_argument('njobs', type=int)
    parser.add_argument('-o', '--output_file', type=pathlib.Path)
    args = parser.parse_args()

    # the 'off' line means no warps
    warps = list((
        str(w.resolve()) for w in args.warps_directory.glob('*.warp')))
    warps += ['off']
    random.shuffle(warps)

    size = int(len(warps) / (args.njobs - 1))
    jobs = [warps[pos:pos+size] for pos in range(0, len(warps), size)]
    with open(args.output_file, 'w') as fout:
        for i, job in enumerate(jobs, start=1):
            for line in sorted(job):
                fout.write(f'{i} {line}\n')


if __name__ == '__main__':
    main()
