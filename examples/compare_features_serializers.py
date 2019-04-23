#!/usr/bin/env python
"""Compare the performances of features serializers

Comparison is on file size, writing and reading speed

"""

import argparse
import datetime
import os
import tabulate
import tempfile

from shennong.audio import Audio
from shennong.utils import list_files_with_extension
from shennong.features import FeaturesCollection
from shennong.features.processor.mfcc import MfccProcessor
from shennong.features.serializers import supported_extensions


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


def print_results(results):
    print('total duration: {}'.format(results['duration']))
    print(
        tabulate.tabulate(
            [[k, sizeof_fmt(v[2]),
              str(v[0]).split('.')[0], str(v[1]).split('.')[0]]
             for k, v in results['data'].items()],
            headers=['extension', 'size', 't write', 't read'],
            tablefmt='fancy_grid'))


def analyze_serializer(features, ext, output_dir):
    with tempfile.TemporaryDirectory(dir=output_dir) as tmpdir:
        filename = os.path.join(tmpdir, 'features' + ext)

        print('writing {}...'.format(filename))
        t1 = datetime.datetime.now()
        features.save(filename)
        t2 = datetime.datetime.now()
        t_write = t2 - t1
        print('took {}'.format(t_write))

        f_size = os.path.getsize(filename)
        print('filesize: {}'.format(sizeof_fmt(f_size)))

        print('reading {}...'.format(filename))
        t1 = datetime.datetime.now()
        features2 = FeaturesCollection.load(filename)
        t2 = datetime.datetime.now()
        t_read = t2 - t1
        print('took {}'.format(t_read))
        print('rw equality: {}'.format(features2 == features))

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
    print('computing MFCC features...')
    t1 = datetime.datetime.now()
    processor = MfccProcessor()
    features = FeaturesCollection(
        **{k: processor.process(v) for k, v in audio_data.items()})
    t2 = datetime.datetime.now()
    print('took {}'.format(t2 - t1))

    # save the features in all the supported formats
    data = {'duration': total_duration,
            'data': {
                ext: analyze_serializer(features, ext, args.output_dir)
                for ext in supported_extensions().keys()}}

    print_results(data)


if __name__ == '__main__':
    main()
    # print_results(RESULTS)
