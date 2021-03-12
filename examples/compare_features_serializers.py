#!/usr/bin/env python
"""Compare the performances of features serializers

Comparison is on file size, writing and reading speed

"""

import argparse
import datetime
import os
import pathlib
import tempfile

import tabulate
from shennong import Audio, FeaturesCollection
from shennong.processor.mfcc import MfccProcessor
from shennong.serializers import supported_serializers
from shennong.utils import list_files_with_extension


# results obtained from a previous run on 1:03:00
RESULTS = {'duration': datetime.timedelta(hours=1, minutes=3),
           'data': {
               '.npz': (datetime.timedelta(seconds=3, microseconds=836073),
                        datetime.timedelta(microseconds=534519),
                        24020221),
               '.mat': (datetime.timedelta(seconds=1, microseconds=615583),
                        datetime.timedelta(microseconds=311637),
                        19699810),
               '.json': (datetime.timedelta(seconds=13, microseconds=267195),
                         datetime.timedelta(seconds=81, microseconds=209878),
                         275948480),
               '.pkl': (datetime.timedelta(microseconds=203584),
                        datetime.timedelta(microseconds=174826),
                        25728528),
               '.h5f': (datetime.timedelta(microseconds=538507),
                        datetime.timedelta(microseconds=235518),
                        23787596),
               '.ark': (datetime.timedelta(microseconds=239396),
                        datetime.timedelta(microseconds=236622),
                        39345566)}}


# from https://stackoverflow.com/questions/1094841
def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'K', 'M', 'G', 'T']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'P', suffix)


def get_size(path):
    """Return the total size of a file or folder"""
    if os.path.isfile(path):
        return os.path.getsize(path)

    path = pathlib.Path(path)
    return sum(os.path.getsize(f) for f in path.glob('**/*') if f.is_file())


def print_results(results):
    print('total duration: {}'.format(results['duration']))
    print(
        tabulate.tabulate(
            [[k, sizeof_fmt(v[2]),
              str(v[0]).split('.')[0], str(v[1]).split('.')[0]]
             for k, v in results['data'].items()],
            headers=['extension', 'size', 't write', 't read'],
            tablefmt='fancy_grid'))


def analyze_serializer(features, serializer, output_dir, with_properties=True):
    with tempfile.TemporaryDirectory(dir=output_dir) as tmpdir:
        filename = os.path.join(tmpdir, 'features_' + serializer)
        if serializer == 'kaldi':
            filename += '.ark'

        print('writing {}...'.format(filename))
        t1 = datetime.datetime.now()
        features.save(
            filename, serializer=serializer, with_properties=with_properties)
        t2 = datetime.datetime.now()
        t_write = t2 - t1
        print('took {}'.format(t_write))

        f_size = get_size(filename)
        print('filesize: {}'.format(sizeof_fmt(f_size)))

        print('reading {}...'.format(filename))
        t1 = datetime.datetime.now()
        features2 = FeaturesCollection.load(filename, serializer=serializer)
        t2 = datetime.datetime.now()
        t_read = t2 - t1
        print('took {}'.format(t_read))

        return (t_write, t_read, f_size)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.add_argument(
        'data_dir', help='input directory with wavs')
    parser.add_argument(
        'output_dir', default='/tmp', nargs='?',
        help='output directory (created files are deleted at exit)')
    parser.add_argument(
        '--no-properties', action='store_true',
        help='do not save features properties')
    parser.add_argument(
        '-j', '--njobs', type=int, default=1,
        help='njobs for MFCC computation')

    args = parser.parse_args()

    # load audio data and compute total duration
    audio_data = {
        os.path.basename(f): Audio.load(f)
        for f in list_files_with_extension(args.data_dir, '.wav')}
    total_duration = datetime.timedelta(
        seconds=int(sum(a.duration for a in audio_data.values())))
    print('found {} wav files, total duration of {}'
          .format(len(audio_data), str(total_duration)))

    # compute the features (default MFCC)
    print(f'computing MFCC features on {args.njobs} jobs...')
    processor = MfccProcessor()
    t1 = datetime.datetime.now()
    features = processor.process_all(audio_data, njobs=args.njobs)
    t2 = datetime.datetime.now()
    print('took {}'.format(t2 - t1))

    # save the features in all the supported formats
    data = {
        'duration': total_duration,
        'data': {
            serializer: analyze_serializer(
                features, serializer, args.output_dir,
                with_properties=not args.no_properties)
            for serializer in supported_serializers() if serializer != 'json'}}

    print_results(data)


if __name__ == '__main__':
    main()
    # print_results(RESULTS)
