#!/usr/bin/env python
"""This script illustrates the use of VTLN within shennong.

The pipeline is as follows:

* Setup the Buckeye corpus (40 speakers, 38h of speech in 254 files)
* Here for fastest computing the corpus is reduced to 10 speakers and 68 files
* Compute VTLN warps on 10m of speech for each speaker
* Extract warped MFCCs on the whole corpus and save them to file

"""

import argparse
import pathlib

from shennong import Audio, Utterances
from shennong.processor import MfccProcessor, VtlnProcessor
from shennong.utils import list_files_with_extension, get_njobs


def prepare_buckeye(directory):
    """Generates a list of utterances from the Buckeye corpus

    The utterances as (<utterance> <wav> <speaker>). The Buckeye directory is
    organized as `<speaker>/<utterance>/<utterance>.wav`

    """
    print('preparing Buckeye corpus')
    wavs = list_files_with_extension(directory, '.wav', abspath=True)

    # for dev, use a reduced corpus
    print('WARNING: reducing to 10 speakers for this example...')
    wavs = [w for w in wavs if 's1' in w]

    utterances = []
    for wav in wavs:
        name = pathlib.Path(wav).stem
        spk = name[:3]
        utterances.append((name, wav, spk))
    utterances = Utterances(utterances)

    print(
        f'found {len(utterances)} utterances from '
        f'{len(utterances.by_speaker())} speakers')
    return utterances


def main():
    """Train VTLN, extract warps and apply warped MFCC on Buckeye corpus"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'buckeye_corpus', type=pathlib.Path,
        help='path to the raw Buckeye Corpus')
    parser.add_argument(
        'output_file', type=pathlib.Path,
        help='where to save the computed MFCCs')
    parser.add_argument(
        '-j', '--njobs', type=int, default=get_njobs(),
        help='number of parallel jobs to use, default to %(default)s')
    parser.add_argument(
        '-d', '--duration', type=float, default=10*60,
        help=('speech duration per speaker for VTLN training, '
              'default to %(default)s'))
    parser.add_argument(
        '--warp-step', type=float, default=0.01,
        help='VTLN warp step, default to %(default)s')
    parser.add_argument(
        '--warp-min', type=float, default=0.85,
        help='VTLN min warp, default to %(default)s')
    parser.add_argument(
        '--warp-max', type=float, default=1.25,
        help='VTLN max warp, default to %(default)s')
    args = parser.parse_args()

    # check input parameters
    if args.output_file.exists():
        raise ValueError(f'{args.output_file} already exists')
    if not args.buckeye_corpus.is_dir():
        raise ValueError(f'{args.buckeye_corpus} is not a directory')

    # generates utterances from the Buckeye corpus
    utterances = prepare_buckeye(args.buckeye_corpus)

    # extract 10m of speech per speaker to train VTLN
    vtln_utterances = utterances.fit_to_duration(args.duration)

    # compute the VTLN warps coefficients
    print(
        f'training VTLN on {args.duration}s per speaker '
        f'({len(vtln_utterances)} utterances)')
    processor = VtlnProcessor(
        warp_step=args.warp_step,
        min_warp=args.warp_min,
        max_warp=args.warp_max)
    processor.set_logger('info')
    warps = processor.process(
        vtln_utterances, njobs=args.njobs, group_by='speaker')

    print('VTLN warps per speaker are:')
    for spk, warp in sorted(warps.items()):
        print(f'{spk}: {warp}')

    # convert warps from speaker to utterance in the whole corpus
    warps = {utt.name: warps[utt.speaker] for utt in utterances}

    print(f'computing MFCCs for {len(utterances)} uttterances')
    features = MfccProcessor().process_all(
        utterances, vtln_warp=warps, njobs=args.njobs)

    print(f'writing MFCCs to {args.output_file}')
    features.save(args.output_file)


if __name__ == '__main__':
    main()
