#!/usr/bin/env python
"""Computes speech features on raw speech audio files

            +-----------------------------+
            |features +--> CMVN +         |
            |                   |         |
   wavs +-->|                   +--> delta+---> output
            |                   |         |
            |             pitch +         |
            +-----------------------------+

"""

import argparse
import datetime
import logging
import os
import sys

from shennong import version_long
from shennong.audio import AudioData
from shennong.features.processor import get_processor
from shennong.features.serializers import supported_extensions
from shennong.utils import list_files_with_extension, null_logger, get_logger


log = null_logger()


def list_wavs(data_dir):
    wavs = list_files_with_extension(data_dir, '.wav', abspath=True)
    return {os.path.splitext(os.path.basename(wav))[0]: wav for wav in wavs}


def init_parser(subparsers, command, epilog):
    processor_class = get_processor(command)
    processor_instance = processor_class()

    parser = subparsers.add_parser(
        command,
        description=processor_class.__doc__.split('\n')[0],
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '-V', '--version', action='version', version=version_long(),
        help='display version information and exit')

    # add verbose/quiet options to control log level
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-v', '--verbose', action='count', default=0, help='''
        increase the amount of logging on stderr (by default only
        warnings and errors are displayed, a single '-v' adds info
        messages and '-vv' adds debug messages, use '--quiet' to
        disable logging)''')

    group.add_argument(
        '-q', '--quiet', action='store_true',
        help='do not display any log message')

    parser.add_argument(
        '-j', '--njobs', type=int, default=1, metavar='<int>',
        help='number of parallel jobs to use, default to %(default)s')

    group = parser.add_argument_group('input/output arguments')
    group.add_argument(
        'wav', nargs='+', help='wav files to compute features on')
    group.add_argument(
        'output_file', help='file to save the computed features')

    # the sample rate is adapted per wav (and thus it is not fixed
    # globally), We also skip htk_compat because this too
    # low-level for a command-line tool.
    ignored_attributes = ['sample_rate', 'htk_compat']

    group = parser.add_argument_group(
        '{} features extraction parameters'.format(command))
    for param, default in processor_instance.get_params().items():
        if param not in ignored_attributes:
            # prepare the help message using the attribute docstring
            help = getattr(processor_class, param).__doc__.strip()
            if help[-1] == '.':
                help = help[:-1]
            help += '. Default is {}.'.format(
                '{:.5g}'.format(default) if isinstance(default, float)
                else default)

            # add the attribute to the parser
            group.add_argument(
                '--{}'.format(param.replace('_', '-')),
                type=type(default),
                metavar='<{}>'.format(type(default).__name__),
                default=default,
                help=help)


class GetConfAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        print(option_strings)
        print(dest)
        print(nargs)
        print(kwargs)
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)
        print(self.__dict__)
        print('-'*30)

    def __call__(self, parser, namespace, values, option_string=None):
        print(namespace)
        print(values)
        print(option_string)
        sys.exit(0)


def main():
    # a footer for help messages
    epilog = (
        'speech-features is part of the shennong library\n'
        'see full documentation at https://coml.lscp.ens.fr/shennong')

    # the possible features processors to be used
    processors = ['mfcc', 'filterbank', 'plp', 'bottleneck']

    parser = argparse.ArgumentParser(
        description=__doc__, epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '-V', '--version', action='version', version=version_long(),
        help='display version information and exit')

    group = parser.add_argument_group('pipeline configuration')
    group.add_argument(
        '-c', '--config-file',
        metavar='<config-file>', type=str, required=True,
        help='configuration file in YAML format, if -F/--fetch-conf is used '
        'the fetched configuration is wrote to this file, else it is read to '
        'initialize the features extraction pipeline')

    group.add_argument(
        '-F', '--fetch-conf', action=GetConfAction, choices=processors)

    # add verbose/quiet options to control log level
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-v', '--verbose', action='count', default=0, help='''
        increase the amount of logging on stderr (by default only
        warnings and errors are displayed, a single '-v' adds info
        messages and '-vv' adds debug messages, use '--quiet' to
        disable logging)''')

    group.add_argument(
        '-q', '--quiet', action='store_true',
        help='do not display any log message')

    parser.add_argument(
        '-j', '--njobs', type=int, default=1, metavar='<int>',
        help='number of parallel jobs to use, default to %(default)s')

    group = parser.add_argument_group('input/output arguments')
    group.add_argument(
        'wav', nargs='+', help='wav files to compute features on')
    group.add_argument(
        'output_file', metavar='<output-file>', type=str,
        help='file to save the computed features')

    # # use a disctinct subcommand for each features processor
    # subparsers = parser.add_subparsers(
    #     title='available features extraction commands',
    #     help='\n'.join('{} - {}'.format(
    #         c, get_processor(c).__doc__.split('\n')[0]) for c in commands))
    # for command in commands:
    #     init_parser(subparsers, command, epilog)

    # parse the command line options
    args = parser.parse_args()

    # setup the logger (level given by -q/-v arguments)
    if args.quiet:
        log = null_logger()
    else:
        if args.verbose == 0:
            level = logging.WARNING
        elif args.verbose == 1:
            level = logging.INFO
        else:  # verbose >= 2
            level = logging.DEBUG
        log = get_logger(name='speech-features', level=level)
    # forward the initialized log to shennong
    AudioData._log = log

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


if __name__ == '__main__':
    main()
