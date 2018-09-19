"""Handles time alignment of speech signals"""

import gzip
import os
import math
import numpy as np


class AlignmentCollection(dict):
    """A dictionary of time alignments indexed by items

    An AlignmentCollection is a usual Python dictionnary with some
    additional functions. Keys are strings and values are `Alignment`
    instances. Depending on the underlying data, a key can be a single
    utterance or the speech corresponding to an entire WAV file or
    speaker.

    Parameters
    ----------
    data : sequence of (item, tstart, tstop, phone)
       A list or a sequence of quadruplets `(item, tstart, tstop,
       phone)` representing a time aligned phone for a given
       `item`. `tstart` is the start timestamp of the pronunced phone,
       `tstop` is the end timestamp of the pronunciation and `phone`
       is a string representation of the phone. `tstart` and `tstop`
       are expressed in seconds.

    Attributes
    ----------
    phones_set : set
        The set of phones present in the alignment

    Raises
    ------
    ValueError
        If one element of `data` is not a quadruplet, or if one
        `tstart` or `tstop` cannot be casted to float

    """
    def __init__(self, data):
        self.phones_set = set()
        for i, entry in enumerate(data):
            if len(entry) != 4:
                raise ValueError(
                    'alignment must have 4 columns but line {} has {}'
                    .format(i+1, len(entry)))

            item = entry[0]
            self.phones_set.add(entry[3])

            # first init of the dict with lists of entries
            if item not in self.keys():
                self[item] = []
            self[item].append(entry[1:])

        # second init: from list to Alignment
        for item, data in self.items():
            try:
                self[item] = Alignment(data, phones_set=self.phones_set)
            except ValueError as err:
                raise ValueError('item {}: {}'.format(item, err))

    def save(self, filename, sort=False, compress=False):
        """Save the alignments to a `filename`

        The saved alignment is in text format (encoded in utf8), each
        line structured as::

            <item> <tstart> <tstop> <phone>

        Parameters
        ----------
        filename : str
            The text file to write (should have a `.txt` extension, or
            `.txt.gz` if `compress` is True, but this is not
            required). Must be a non existing file.
        sort : bool, optional
            When True, the items are sorted in lexicographical
            order. Default to False.
        compress : bool, optional
            When True the file is compressed using the gzip
            algorithm. Default to False.

        Raises
        ------
        ValueError
            If the `filename` already exists or is not writable.

        """
        # check this file does not exist
        if os.path.isfile(filename):
            raise ValueError('{} already exist'.format(filename))

        # prepare the items to write, optionally sorted
        items = self.keys()
        if sort is True:
            items = sorted(items)

        # write in raw text or gzip text format
        open_fun = gzip.open if compress is True else open

        try:
            with open_fun(filename, 'wt', encoding='utf8') as fh:
                # write the file item by item
                for item in items:
                    fh.write('\n'.join(self._list_str(item)) + '\n')
        except FileNotFoundError:
            raise ValueError('cannot write to {}'.format(filename))

    def _list_str(self, item):
        """Returns an alignment item as a list of strings"""
        return [item + ' ' + line for line in self[item]._list_str()]

    @staticmethod
    def load(alignment_file, compress=False):
        """Returns an AlignmentCollection loaded from the `alignment_file`

        The file, in text format and optionally compressed, is read as
        utf8. It must be composed of lines with the following 4
        fields:

            <item> <tstart> <tstop> <phone>

        Parameters
        ----------
        alignment_file : str
            The path to the alignment file to read, must be an
            existing file

        Returns
        -------
        alignment : AlignmentCollection
            The AlignmentCollection instance initialized from the
            `alignment_file`

        Raises
        ------
        ValueError
            If the `alignment_file` is not a valid alignment or if the
            AlignmentCollection cannot be instanciated.

        """
        if not os.path.isfile(alignment_file):
            raise ValueError('{}: file not found'.format(alignment_file))

        # read the input file compressed or not
        open_fun = gzip.open if compress is True else open

        data = [line.split() for line in
                open_fun(alignment_file, 'rt', encoding='utf8').readlines()]

        return AlignmentCollection(data)


class Alignment:
    """Time alignment of phones

    An Alignment handles a time alignment of phones, i.e. a suite of
    phones linked with their onset (tstart) and offset (tstop)
    timestamps.

    The timestamps in `data` must fill the following conditions to
    ensure temporal consistency:

    * `tstart` must be sorted in increasing order: `data` is a
       temporal sequence

    * `tstart` must be lesser than `tstop`: each phone in `data` has a
      positive duration

    * `tstop[n]` must be equal to `tstart[n+1]`: `data` has a temporal
      continuity.

    Parameters
    ----------
    data : sequence of (tstart, tstop, phone)
        A list or sequence of triplets `(tstart, tstop, phone)`
        representing a time aligned phone. `tstart` is the start
        timestamp of the pronunced phone, `tstop` is the end timestamp
        of the pronunciation and `phone` is a string representation of
        the phone. `tstart` and `tstop` are expressed in seconds.
    phones_set : set, optional
        The set of the different phones present in the alignment. If
        not specified the `phones_set` is built from the phones in
        `data`.

    Attributes
    ----------
    times : array of float32, shape = [nphones, 2]
        The array of (tstart, tstop) timestamps for each aligned phone
    phones : array of str, shape = [nphones, 1]
       The array of aligned phones
    phones_set : set
       The set of the distinct phones present in the alignement

    Raises
    ------
    ValueError
        If the timestamps in `data` are not temporally
        consistent. When `phones_set` is defined and a phone of the
        alignment is not deefiened in the set.

    """
    def __init__(self, data, phones_set=None):
        # load the timestamps as (tstart, tstop)
        self.times = np.array([entry[:2] for entry in data], dtype=np.float32)

        # check tstart < tstop for all timestamps
        for i in range(self.times.shape[0]):
            if self.times[i, 0] >= self.times[i, 1]:
                raise ValueError(
                    'tstart must be lesser than tstop')

        # check tstarts are sorted in increasing order and tstop[n]
        # matches tstart[n+1]
        for i in range(self.times.shape[0] - 1):
            if self.times[i, 0] > self.times[i+1, 0]:
                raise ValueError(
                    'timestamps must be sorted in increasing order')
            if self.times[i, 1] != self.times[i+1, 0]:
                raise ValueError(
                    'mismatch in tstop/tstart timestamps')

        self.phones = np.array([d[2] for d in data])

        # if the phone set is unspecified, take it from the alignment
        if phones_set is None:
            self.phones_set = set(self.phones)
        else:
            # phones_set is specified, make sure all phones in the
            # alignment are present in the phones_set
            phone_diff = set(self.phones).difference(set(phones_set))
            if len(phone_diff) != 0:
                raise ValueError(
                  'following phones are in alignment but not in phones_set: '
                  '{}'.format(', '.join(phone_diff)))

            self.phones_set = phones_set

    def __eq__(self, other):
        return (np.array_equal(self.times, other.times)
                and np.array_equal(self.phones, other.phones)
                and self.phones_set == self.phones_set)

    def __getitem__(self, time):
        """Returns data aligned in `time` slice

        Extracts a subpart of the alignment using slice notation. For
        example ``alignment[:2.0]`` will extract alignment for the
        first two seconds, or ``alignment[3.25:4.25]`` will extract
        one second in the middle of the data.

        Parameters
        ----------
        time : slice (start, stop), in seconds
            The time interval on which to extract the alignment is
            defined by `slice.start` and `slice.stop`, expressed in
            seconds. `slice.step` is not used.

        Returns
        -------
        times : array of float32
            The timestamps within the given `time` interval
        phones : array of string
            The phones within the given `time` interval

        Raises
        ------
        ValueError
            If `time` is not a slice

        """
        if not isinstance(time, slice):
            raise ValueError(
                'time must be a slice but is {}'.format(type(time)))

        # setup the start and stop timestamp from the `time` slice
        tstart = self.times[0, 0] if time.start is None else time.start
        tstop = self.times[-1, 1] if time.stop is None else time.stop

        # better to use np.searchsorted than np.where here,
        # considering the times are sorted
        istart = np.searchsorted(self.times[:, 0], tstart, side='left')
        istop = np.searchsorted(self.times[:, 1], tstop, side='right')
        return self.times[istart:istop], self.phones[istart:istop]

    def _list_str(self):
        """Returns the alignement as a list of strings"""
        return ['{:.5f} {:.5f} {}'.format(  # float precision to 5 decimals
            self.times[i, 0], self.times[i, 1], self.phones[i])
                for i in range(self.times.shape[0])]

    def at_sample_rate(self, sample_rate, tstart=None, tstop=None):
        """Yields phones read at the given `sample_rate`"""
        # the phones to read
        times, phones = self[tstart:tstop]

        for i, (tstart, tstop) in enumerate(times):
            # number of times the phone is read at the given
            # samplerate
            nread = math.floor((tstop - tstart) * sample_rate)
            for _ in range(nread):
                yield phones[i]

    def duration(self):
        """Returns the duration of the aligned speech, in seconds"""
        return self.times[-1, 1] - self.times[0, 0]

    def nphones(self):
        """Returns the number of distinct phones in the alignment"""
        return len(self.phones_set)
