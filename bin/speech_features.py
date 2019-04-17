#!/usr/bin/env python
"""Computes speech features on raw speech wav files

The general (and configurable) extraction pipeline is as follow:

          |--> features --> CMVN --> delta -->|
   wav -->|                                   |--> output
          |---------------> pitch ----------->|

Features extraction basically involves two steps: configuring an
extraction pipeline and extracting features given a configuration. See
the available commands below for more info.

"""

import argparse
import datetime
import os
import sys

import shennong.features.pipeline as pipeline
import shennong.utils as utils
from shennong import version_long
from shennong.audio import AudioData
from shennong.features.serializers import supported_extensions


binary_name = 'speech-features'
"""The name of executable as users see it"""

log = utils.null_logger()
"""A logger to display messages (configurable with -v/-q options)"""


def _parser_verbose(parser):
    """Initializes log messages options (-v/-q)"""
    # add verbose/quiet options to control log level
    group = parser.add_argument_group('log messages arguments')
    group = group.add_mutually_exclusive_group()
    group.add_argument(
        '-v', '--verbose', action='count', default=0, help='''
        increase the amount of logging on stderr (by default only
        warnings and errors are displayed, a single '-v' adds info
        messages and '-vv' adds debug messages, use '--quiet' to
        disable logging)''')
    group.add_argument(
        '-q', '--quiet', action='store_true',
        help='do not display any log message')


#
# speech-features config
#

def parser_config(subparsers, epilog):
    """Initialize options for 'speech-features config'"""
    parser = subparsers.add_parser(
        'config',
        description='Generate a configuration for features extraction',
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '-o', '--output', metavar='config-file', default=None,
        help='The YAML configuration file to write. '
        'If not specified, write to stdout')

    parser.add_argument(
        '--no-comments', action='store_true',
        help='Do not include comments in the output YAML configuration file. '
        'By default all parameters in the YAML are explained in comments.')

    group = parser.add_argument_group('pipeline arguments')
    group.add_argument(
        'features', type=str, choices=pipeline.valid_features(),
        help='Configure the pipeline to extract those features')

    group.add_argument(
        '--no-pitch', action='store_true',
        help='Configure without pitch extraction')

    group.add_argument(
        '--no-cmvn', action='store_true',
        help='Configure without CMVN normalization')

    group.add_argument(
        '--no-delta', action='store_true',
        help='Configure without deltas extraction')


def command_config(args):
    config = pipeline.get_default_config(
        args.features,
        to_yaml=True, yaml_commented=not args.no_comments,
        with_pitch=not args.no_pitch,
        with_cmvn=not args.no_cmvn,
        with_delta=not args.no_delta)

    output = sys.stdout if not args.output else open(args.output, 'w')
    output.write(config)


#
# speech-features extract
#

def parser_extract(subparsers, epilog):
    parser = subparsers.add_parser(
        'extract',
        description='Extract features from wav files given a configuration',
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '-j', '--njobs', type=int, default=1, metavar='<int>',
        help='number of parallel jobs to use, default to %(default)s')

    group = parser.add_argument_group('input/output arguments')
    group.add_argument(
        'config', metavar='<config-yaml>', type=str,
        help='the pipeline configuration as a YAML file')
    group.add_argument(
        'wavs', metavar='<wavs-index>', type=str,
        help='wav files to compute features on')
    group.add_argument(
        'output_file', help='file to save the computed features')


def command_extract(args):
    # setup the logger (level given by -q/-v arguments)
    if args.quiet:
        log = utils.null_logger()
    else:
        if args.verbose == 0:
            level = 'warning'
        elif args.verbose == 1:
            level = 'info'
        else:  # verbose >= 2
            level = 'debug'
        log = utils.get_logger(name='speech-features', level=level)
    # forward the initialized log to shennong
    utils._logger = log

    # make sure the output file is not already existing and have a
    # valid extension
    output_file = args.output_file
    if os.path.exists(output_file):
        log.error('output file already exist: %s', output_file)
        return
    output_ext = os.path.splitext(output_file)[1]
    if output_ext not in supported_extensions().keys():
        log.error(
            'output file has an unsupported extension "%s", must be in %s',
            output_ext, ", ".join(supported_extensions().keys()))
        return

    # the list of wav files on which to estimate speech
    # features. Check they are all correct (loadable). Check as well
    # for sample rate (warning if not homogeneous) and mono (fail if
    # one or more is not mono)
    if not args.wav:
        log.error('no wav files, exiting')
        sys.exit(-1)
    wavs_metadata = [AudioData.scan(w) for w in args.wav]
    log.info(
        'get %s wav files, total duration: %s', len(args.wav),
        datetime.timedelta(seconds=sum(w.duration for w in wavs_metadata)))
    if not all(w.nchannels == 1 for w in wavs_metadata):
        log.error('all wavs are not mono, exiting')
        sys.exit(-1)
    samplerates = set(w.sample_rate for w in wavs_metadata)
    if len(samplerates) > 1:
        log.warning(
            'several sample rates found in wav files: %s, features extraction '
            'will work but this may not be a good idea to work on '
            'heterogeneous data',
            ', '.join(str(s) + 'Hz' for s in samplerates))

    # TODO
    audios = {
        os.path.splitext(os.path.basename(w))[0]: AudioData.load(w)
        for w in args.wav}
    return

    # computes MFCC with default arguments and save them to disk
    processor = MfccProcessor(sample_rate=args.sample_rate)
    features = processor.process_all(audios, njobs=args.njobs)
    features.save(args.out_file)


@utils.CatchExceptions
def main():
    # a footer for help messages
    epilog = (
        'speech-features is part of the shennong library\n'
        'see full documentation at https://coml.lscp.ens.fr/shennong')

    parser = argparse.ArgumentParser(
        description=__doc__, epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--version', action='version', version=version_long(),
        help='display version and copyright information and exit')

    # use a disctinct subcommand for each features processor
    subparsers = parser.add_subparsers(
        title='speech-features commands',
        description='bla bla',
        dest='command')

    # add parser for each command
    parser_config(subparsers, epilog)
    parser_extract(subparsers, epilog)

    # parse the command line options
    args = parser.parse_args()

    if args.command == 'config':
        command_config(args)
    elif args.command == 'extract':
        command_extract(args)


if __name__ == '__main__':
    main()
