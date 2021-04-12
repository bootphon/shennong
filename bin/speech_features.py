#!/usr/bin/env python
"""Speech features extraction pipeline from raw audio files

The general extraction pipeline is as follow::

     <input-config>     |--> features --> CMVN --> delta -->|
         and         -->|     (VTLN)                        |--> <output-file>
  <input-utterances>    |---------------> pitch ----------->|


Simple exemple
~~~~~~~~~~~~~~

Features extraction basically involves three steps:

1. Configure an extraction pipeline. For exemple this defines a full pipeline
   for MFCCs extraction (with CMVN, but without delta, pitch nor VTLN) and
   writes it to the file ``config.yaml``::

     speech-features config mfcc --no-pitch --no-delta --no-vtln -o config.yaml

   You can then edit the file ``config.yaml`` to modify the
   parameters.

2. Define a list of utterances on which to extract features (along
   with optional speakers or timestamps specification), for exemple
   you can write a ``utterances.txt`` file with the following content
   (see below for details on the format)::

     utterance1 /path/to/audio1.wav speaker1
     utterance2 /path/to/audio2.wav speaker1
     utterance3 /path/to/audio3.wav speaker2

3. Apply the configured pipeline on the defined utterances. For
   exemple this computes the features using 4 parallel subprocesses
   and save them to a file in the numpy ``.npz`` format::

     speech-features extract --njobs 4 config.yaml utterances.txt features.npz


Definition of `<input-config>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``<input-config>`` is a configuration file in YAML format defining
all the parameters of the extraction pipeline, including main features
extraction (spectrogram, mfcc, plp, rastaplp, filterbank or bottleneck
features) and post-processing (CMVN, delta and pitch extraction).

You can generate a configuration template using ``speech-features
config``. It will write a YAML file with default parameters that you
can edit. See ``speech-features config --help`` for description of the
available options.


Definition of `<input-utterances>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``<input-utterances>`` is a text file indexing the utterances on
which to apply the extraction pipeline. Each line of the file defines
a single utterance (or sentence, or speech fragment), it can have one
of the following formats:

1. ``<utterance-id> <audio-file>``

    The simplest format. Give a name to each utterance, identifiers must be
    unique. Each entire audio file is considered a single utterance.

2. ``<utterance-id> <audio-file> <speaker-id>``

    Specify a speaker for each utterance. This is required if you are
    using CMVN or VTLN normalization per speaker.

3. ``<utterance-id> <audio-file> <tstart> <tstop>``

    Each audio file contains several utterances, the utterance boundaries are
    defined by the start and stop timestamps within the audio file (given in
    seconds).

4. ``<utterance-id> <audio-file> <speaker-id> <tstart> <tstop>``

    Combination of 2 and 3. Several utterances per audio file, with speakers
    identification.


Definition of `<output-file>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``<output-file>`` will store the extracted features. The underlying
format is a dictionnary of utterances. Each utterance's features are
stored as a matrix `[nframes * ndims]`, along with timestamps and
metadata.

Several file formats are supported, the format is guessed by the file
extension specified in command line:

  ===========  ========= ===========================================
  File format  Extension Use case
  ===========  ========= ===========================================
  pickle       .pkl      Very fast, standard Python format
  h5features   .h5f      Fast and efficient for very big datasets
  numpy        .npz      Standard numpy format
  matlab       .mat      Compatibility with Matlab
  kaldi        .ark      Compatibility with Kaldi
  CSV          <folder>  Very slow, raw text, one utterance per file
  ===========  ========= ===========================================

More info on file formats are available on the online documentation.

"""

import argparse
import os
import sys

import shennong.logger as logger
import shennong.pipeline as pipeline
import shennong.utils as utils
from shennong import url, version_long
from shennong.serializers import supported_extensions
from shennong.utterances import Utterances


#
# speech-features config
#

def parser_config(subparsers, epilog):
    """Initialize options for 'speech-features config'"""
    parser = subparsers.add_parser(
        'config',
        description='Generate a configuration for features extraction, '
        "have a 'speech-features --help' for more details",
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
        '--cmvn', action='store_true',
        help='Configure with CMVN normalization')

    group.add_argument(
        '--delta', action='store_true',
        help='Configure with deltas extraction')

    group.add_argument(
        '--pitch', choices=['kaldi', 'crepe'],
        help=(
            'Configure with Kaldi or CREPE pitch extraction, '
            'no pitch by default'))

    group.add_argument(
        '--vtln', choices=['simple', 'full'],
        help=(
            'Configure with VTLN normalization, no VTLN by default. '
            'When "full" exposes all arguments, when "simple" exposes '
            'a reduced set of arguments.'))


def command_config(args):
    """Execute the 'speech-features config' command"""
    config = pipeline.get_default_config(
        args.features,
        to_yaml=True, yaml_commented=not args.no_comments,
        with_pitch=args.pitch or False,
        with_cmvn=args.cmvn,
        with_delta=args.delta,
        with_vtln=args.vtln or False)

    output = sys.stdout if not args.output else open(args.output, 'w')
    output.write(config)


#
# speech-features extract
#

def parser_extract(subparsers, epilog):
    """Initialize options for 'speech-features extract'"""
    parser = subparsers.add_parser(
        'extract',
        description='Extract features from wav files given a configuration, '
        "have a 'speech-features --help' for more details",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '-j', '--njobs', type=int, default=1, metavar='<int>',
        help='number of parallel jobs to use, default to %(default)s')

    group = parser.add_argument_group('input/output arguments')
    group.add_argument(
        'config', metavar='<input-config>', type=str,
        help='pipeline configuration file in YAML format, as generated by '
        "the 'speech-features config' command")
    group.add_argument(
        'utts_index', metavar='<input-utterances>', type=str,
        help='utterances index file defining utterances on which to compute '
        "features on, see 'speech-features --help' for a description "
        "of the format")
    group.add_argument(
        'output_file', metavar='<output-file>',
        help='file to save the computed features (must not exist), '
        "see 'speech-features --help' for a list of supported output formats")

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


def command_extract(args):
    """Execute the 'speech-features extract' command"""
    # setup the logger (level given by -q/-v arguments)
    if args.quiet:
        log = utils.null_logger()
        level = 'error'
    else:
        if args.verbose == 0:
            level = 'warning'
        elif args.verbose == 1:
            level = 'info'
        else:  # verbose >= 2
            level = 'debug'
        log = logger.get_logger(name='speech-features', level=level)

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

    # make sure the input config and utterances exists
    for filename in (args.config, args.utts_index):
        if not os.path.exists(filename):
            log.error('input file not found: %s', filename)

    # read the utterances file
    utterances = Utterances.load(args.utts_index)

    # run the pipeline
    features = pipeline.extract_features(
        args.config, utterances, njobs=args.njobs, log=log)

    # save the features
    log.info('saving the features to %s', output_file)
    features.save(output_file)


@utils.CatchExceptions
def main():
    """Entry point of the 'speech-features' program"""
    # a footer for help messages
    epilog = (
        f'speech-features is part of the shennong library\n'
        f'see full documentation at {url()}')

    description = (
        __doc__.replace('::', ':').replace('``', '\'') +
        'Command line arguments\n' +
        '~~~~~~~~~~~~~~~~~~~~~~\n')

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '-V', '--version', action='version', version=version_long(),
        help='display version and copyright information and exit')

    # use a disctinct subcommand for each features processor
    subparsers = parser.add_subparsers(
        title='speech-features commands',
        description="use 'speech-features <command> --help' for more details",
        help="the 'config' command generates configuration templates, "
        "the 'extract' command extracts features given a configuration",
        dest='command')

    # add parser for each command
    parser_config(subparsers, epilog)
    parser_extract(subparsers, epilog)

    # parse the command line options
    args = parser.parse_args()

    # execute the requested command
    if args.command == 'config':
        command_config(args)
    elif args.command == 'extract':
        command_extract(args)


if __name__ == '__main__':
    main()
