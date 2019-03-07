"""Handles time alignments of speech signals

A speech signal is time-aligned when, for each pronunced phone in the
speech, their associated onset and offset times are provided. An
alignment can be obtained manually (by annotation), or automatically
(using Kaldi for example).

Alignment files supported by `shennong` are text files (optionnaly
compressed) in which each line is formatted as follow::

     <item> <onset> <offset> <phone>

The ``<item>`` can be the reference of an utterance, a speaker, or an
file. The ``<onset>`` and ``<offset>`` are begin and end timestamps of
the ``<phone>`` being pronunced. An exemple file is located in
``shennong/test/data/alignment.txt`` and has been produced by a Kaldi
forced-alignement.

This module provides two classes to operate on time alignments:

* :class:`AlignmentCollection` is a high-level class to load/save alignment
  files. It exposes a dictionnary of items mapped to alignments.

* :class:`Alignment` is the class representing a time-alignment for a
  single *item*.

Examples
--------

>>> from shennong.alignment import AlignmentCollection

Load a file with time alignment for 34 items

>>> alignments = AlignmentCollection.load('./test/data/alignment.txt')
>>> len(alignments.keys())
34

Get the alignment of one item, an item from an AlignmentCollection is
an instance of Alignment:

>>> ali1 = alignments['S01F1522_0033']
>>> type(ali1)
<class 'shennong.alignment.Alignment'>
>>> ali1.duration()
0.64
>>> print(ali1)
0.0125 0.0425 m
0.0425 0.1225 a
0.1225 0.1825 s
0.1825 0.2425 o
0.2425 0.3025 r
0.3025 0.3625 e
0.3625 0.4325 k
0.4325 0.4925 a
0.4925 0.5625 r
0.5625 0.6525 a

Extract a subpart of the alignment, as an Alignment instance as well

>>> ali2 = ali1[0.4325:0.6525]
>>> print(ali2)
0.4325 0.4925 a
0.4925 0.5625 r
0.5625 0.6525 a

"""

import gzip
import os
import numpy as np


class AlignmentCollection(dict):
    """A dictionary of :class:`.Alignment` indexed by items

    An `AlignmentCollection` is a usual Python dictionary with some
    additional functions. Keys are strings, values are `Alignment`
    instances.

    Parameters
    ----------
    data : sequence of quadruplets
       A list or a sequence of quadruplets `(item, onset, offset,
       phone)` representing a time aligned phone for a given
       `item`, where `onset` is the start timestamp of the pronunced phone,
       `offset` is the end timestamp of the pronunciation and `phone`
       is a string representation of the phone. `onset` and `offset`
       are expressed in seconds.

    Raises
    ------
    ValueError
        If one element of `data` is not a quadruplet, if the Alignment
        mapped to an `item` cannot be instanciated.

    """
    def __init__(self, data):
        for i, entry in enumerate(data):
            if len(entry) != 4:
                raise ValueError(
                    'alignment must have 4 columns but line {} has {}'
                    .format(i+1, len(entry)))

            item = entry[0]

            # first init of the dict with lists of entries
            if item not in self.keys():
                self[item] = []
            self[item].append(entry[1:])

        # second init: from list to Alignment
        for item, data in self.items():
            try:
                self[item] = Alignment.from_list(data, validate=True)
            except ValueError as err:
                raise ValueError('item {}: {}'.format(item, err))

    @staticmethod
    def load(filename, compress=False):
        """Returns an `AlignmentCollection` loaded from the `alignment_file`

        The text file, optionally compressed, is read as utf8. It must
        be composed of lines with 4 fields ``<item> <onset> <offset>
        <phone>``.

        Parameters
        ----------
        filename : str
            The path to the alignment file to read, must be an
            existing text file.

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
        if not os.path.isfile(filename):
            raise ValueError('{}: file not found'.format(filename))

        # read the input file compressed or not
        open_fun = gzip.open if compress is True else open

        data = [line.split() for line in
                open_fun(filename, 'rt', encoding='utf8').readlines()]

        return AlignmentCollection(data)

    def save(self, filename, sort=False, compress=False):
        """Save the alignments to a `filename`

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
        return [item + ' ' + '{} {} {}'.format(l[0], l[1], l[2])
                for l in self[item].to_list()]

    def get_phones_inventory(self):
        """Returns the different phones composing the collection

        Returns
        -------
        phones : set
            Unique phones present in the collection's alignments

        """
        return set.union(*(v.get_phones_inventory() for v in self.values()))


class Alignment:
    """Time alignment of phones

    An Alignment handles a time alignment of phones, i.e. a suite of
    phones linked with their onset and offset timestamps. See the
    :func:`validate` method for a list constraints applying to the
    `data`.

    Parameters
    ----------
    times : array of float, shape = [nphones, 2]
        The array of (onset, offset) timestamps for each aligned phone
    phones : array of str, shape = [nphones, 1]
        The array of aligned phones
    validate : bool, optional
        When True, checks the alignment is in a valid format, when
        False does not perform any verification, default is True

    Raises
    ------
    ValueError
        When :func:`validate` is True and the alignment data is not
        correctly formatted

    """
    def __init__(self, times, phones, validate=True):
        self._times = times
        self._phones = phones

        if validate is True:
            self.validate()

    @property
    def times(self):
        """The (start, stop) timestamps of the aligned phones in seconds"""
        return self._times

    @property
    def onsets(self):
        """The start timestamps of the aligned phones in seconds"""
        return self._times[:, 0]

    @property
    def offsets(self):
        """The stop timestamps of the aligned phones in seconds"""
        return self._times[:, 1]

    @property
    def phones(self):
        """The aligned phones associated with timestamps"""
        return self._phones

    @staticmethod
    def from_list(data, validate=True):
        """Build an Alignment from a list of (tstart, tsop, phone) triplets

        This method checks all elements in the `data` list have 3
        fields, convert them to `times` and `data` arrays, and
        instanciates an Alignment instance with them.

        Parameters
        ----------
        data : sequence of (tstart, tstop, phone)
            A list or sequence of triplets `(tstart, tstop, phone)`
            representing a time aligned phone. `tstart` and `tstop`
            are the onset and offset of the pronunciation (in
            seconds). `phone` is a string representation of the phone.

        """
        # check we have 3 fields in each data entry
        for i, entry in enumerate(data):
            if len(entry) != 3:
                raise ValueError(
                    'line {}: entry must have 3 fields but has {}'
                    .format(i, len(entry)))

        times = np.array([d[:2] for d in data], dtype=np.float)
        phones = np.array([d[2] for d in data])
        return Alignment(
            times, phones, validate=validate)

    def validate(self):
        """Raises a ValueError is the Alignment is not consistent

        The following conditions must apply for the alignment to be
        valid:

        * `onsets`, `offsets` and `phones` must have the same length

        * `onsets` and `offsets` must be sorted in increasing order:
           `data` is a temporal sequence

        * `onsets[n]` must be lesser than `offsets[n]`: each phone in
          `data` has a strictly positive duration

        * `offsets[n]` must be equal to `onsets[n+1]`: `data` has a
          temporal continuity.

        """
        # same length for timestamps and phones
        if not self._times.shape[0] == self.phones.shape[0]:
            raise ValueError('timestamps and phones must have the same length')

        # check tstart < tstop for all timestamps
        for i in range(self.phones.shape[0]):
            if self.onsets[i] >= self.offsets[i]:
                raise ValueError(
                    'phone {}: onset must be lesser than offset'.format(i))

        # check tstarts are sorted in increasing order and tstop[n]
        # matches tstart[n+1]
        for i in range(self.phones.shape[0] - 1):
            if self.onsets[i] > self.onsets[i+1]:
                raise ValueError(
                    'timestamps must be sorted in increasing order')
            if self.offsets[i] != self.onsets[i+1]:
                raise ValueError(
                    'mismatch in tstop/tstart timestamps')

    def is_valid(self):
        """Returns True if the Alignment is consistent, False otherwise"""
        try:
            self.validate()
        except ValueError:
            return False
        return True

    def __eq__(self, other):
        return (np.array_equal(self._times, other._times)
                and np.array_equal(self.phones, other.phones))

    def __getitem__(self, time):
        """Returns data aligned in `time` slice

        Extracts a subpart of the alignment using slice notation. For
        example ``alignment[:2.0]`` will extract alignment for the
        first two seconds, or ``alignment[3.25:4.25]`` will extract
        one second in the middle of the data.

        Parameters
        ----------
        time : slice (onset, offset), in seconds
            The time interval on which to extract the alignment is
            defined by `slice.start` and `slice.stop`, expressed in
            seconds. `slice.step` is not used.

        Returns
        -------
        alignment : Alignment
            The sub-alignment rextracted from the original one

        Raises
        ------
        ValueError
            If `time` is not a slice, or if `time.step` is defined.

        """
        if not isinstance(time, slice):
            raise ValueError(
                'time must be a slice but is {}'.format(type(time)))

        if time.step is not None:
            raise ValueError('time.step is defined but is useless')

        # setup the start and stop timestamp from the `time` slice,
        # bound them at start and stop timestamps of the alignment
        tmin = self.onsets[0]
        tstart = time.start
        if tstart is None or tstart < tmin:
            tstart = tmin

        tmax = self.offsets[-1]
        tstop = time.stop
        if tstop is None or tstop > tmax:
            tstop = tmax

        # deal with corner cases
        if tstart >= tstop or tstart >= tmax or tstop <= tmin:
            return Alignment(np.array([]), np.array([]), validate=False)

        if tstart == tmin and tstop == tmax:
            return self

        # now (tstart, tstop) are in boundaries
        assert tmin <= tstart < tstop <= tmax

        # TODO in the following lines we can optimize. This is useless
        # to do a np.where on the whole timestamps, can we restrict
        # the area of search using np.searchsorted for instance?

        # find the start index (last <= tstart)
        if tstart == tmin:
            istart = 0
        else:
            istart = np.where(self.onsets <= tstart)[0][-1]

        # find the stop index (first >= tstop)
        if tstop == tmax:
            istop = self.phones.shape[0] - 1
        else:
            istop = np.where(self.offsets >= tstop)[0][0]

        # we have a partial read of a single phone
        if istart == istop:
            phones = np.array(self.phones[istart:istart+1])
            times = np.array([tstart, tstop]).reshape(1, 2)
        else:  # build the computed subalignment
            phones = self.phones[istart:istop+1]
            times = np.copy(self._times[istart:istop+1, :])
            times[0, 0] = tstart
            times[-1, 1] = tstop

        return Alignment(times, phones, validate=False)

    def __repr__(self):
        return '\n'.join(
            '{} {} {}'.format(t0, t1, p)
            for (t0, t1, p) in self.to_list())

    def to_list(self):
        """Returns the alignment as a list of triplets (onset, offset, phone)

        This is the reverse operation of :func:`from_list`.

        """
        return [(self.onsets[i], self.offsets[i], self.phones[i])
                for i in range(self.phones.shape[0])]

    def at_sample_rate(self, sample_rate):
        """Returns an array of phones read at the given `sample_rate`"""
        # allocate the result data
        data = np.zeros(
            (int(self.duration() * sample_rate),),
            dtype=self.phones.dtype)

        # the sampled timestamps
        times = np.arange(data.shape[0]) / sample_rate + self.onsets[0]

        j = 0
        for i in range(data.shape[0]):
            while times[i] >= self.offsets[j]:
                j += 1
            data[i] = self.phones[j]

        return data

    def duration(self):
        """Returns the duration of the alignment in seconds"""
        if len(self.phones) == 0:
            return 0
        return self.offsets[-1] - self.onsets[0]

    def get_phones_inventory(self):
        """Returns the different phones composing the alignment

        Returns
        -------
        phones : set
            Unique phones present in the alignment

        """
        return set(self.phones)
