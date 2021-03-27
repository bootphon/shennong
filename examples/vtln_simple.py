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

from shennong import Audio
from shennong.processor import MfccProcessor, VtlnProcessor
from shennong.utils import list_files_with_extension, get_njobs


def prepare_buckeye(directory):
    """Generates a list of utterances from the Buckeye corpus

    The utterances are returned as list of (<utterance> <wav> <speaker>). The
    Buckeye directory is organized as `<speaker>/<utterance>/<utterance>.wav`

    """
    print('preparing Buckeye corpus')
    wavs = list_files_with_extension(directory, '.wav', abspath=True)

    # for dev, use a reduced corpus
    print('WARNING: reducing to 10 speakers for this example...')
    wavs = [w for w in wavs if 's1' in w]

    utts = [pathlib.Path(w).stem for w in wavs]
    spks = [u[:3] for u in utts]

    print(f'found {len(utts)} utterances from {len(set(spks))} speakers')
    return list(zip(utts, wavs, spks))


def prepare_vtln_utterances(utterances, duration):
    """Returns the first `duration`s of each speaker"""
    print(f'build VTLN utterances: {duration}s per speaker')
    vtln_utterances = []
    for speaker in set(utt[2] for utt in utterances):
        # for the utterances from that speaker
        remaining_duration = duration
        for name, wav, _ in (utt for utt in utterances if utt[2] == speaker):
            wav_duration = Audio.scan(wav).duration
            if wav_duration >= remaining_duration:
                vtln_utterances.append(
                    (name, wav, speaker, 0, remaining_duration))
                remaining_duration = 0
                break

            vtln_utterances.append(
                (name, wav, speaker, 0, wav_duration))
            remaining_duration -= wav_duration

        if remaining_duration > 0:
            raise ValueError(
                f'not enough audio for speaker {speaker} '
                f'{duration - remaining_duration}s < {duration}s') from None

    return vtln_utterances


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
        '--warp-step', type=float, default=0.01, help='VTLN warp step')
    parser.add_argument(
        '--warp-min', type=float, default=0.85, help='VTLN min warp')
    parser.add_argument(
        '--warp-max', type=float, default=1.25, help='VTLN max warp')
    args = parser.parse_args()

    # check input parameters
    if args.output_file.exists():
        raise ValueError(f'{args.output_file} already exists')
    if not args.buckeye_corpus.is_dir():
        raise ValueError(f'{args.buckeye_corpus} is not a directory')

    # generates utterances from the Buckeye corpus
    utterances = prepare_buckeye(args.buckeye_corpus)

    # extract 10m of speech per speaker to train VTLN
    vtln_utterances = prepare_vtln_utterances(utterances, 10*60)

    # compute the VTLN warps coefficients
    processor = VtlnProcessor(
        warp_step=args.warp_step,
        min_warp=args.warp_min,
        max_warp=args.warp_max)
    processor.set_logger('info')
    warps = processor.process(vtln_utterances, njobs=args.njobs)

    # at this point we have a dict (utterance_vtln -> warp), with an identical
    # warp for each utterance of a same speaker, but we need to retrieve a dict
    # (speaker -> warp)
    warps = {utt[:3]: warp for utt, warp in warps.items()}

    print('VTLN warps per speaker are:')
    for spk, warp in sorted(warps.items()):
        print(f'{spk}: {warp}')

    # convert warps from speaker to utterance in the whole corpus
    warps = {utt[0]: warps[utt[2]] for utt in utterances}

    print(f'computing MFCCs for {len(utterances)} uttterances')
    features = MfccProcessor().process_all(
        {name: Audio.load(audio) for name, audio, _ in utterances},
        vtln_warp=warps, njobs=args.njobs)

    print(f'writing MFCCs to {args.output_file}')
    features.save(args.output_file)


if __name__ == '__main__':
    main()
