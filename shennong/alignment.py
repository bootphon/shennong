"""Handles time alignment of speech signals"""

import gzip
import os
import numpy as np


class AlignmentCollection(dict):
    """A dictionary of time alignments indexed by items

    An AlignmentCollection is a usual Python dictionary with some
    additional functions. Keys are strings, values are `Alignment`
    instances. Depending on the underlying data, keys can represents a
    single utterance, a single speaker or an entire WAV file.

    Parameters
    ----------
    data : sequence of (item, onset, offset, phone)
       A list or a sequence of quadruplets `(item, onset, offset,
       phone)` representing a time aligned phone for a given
       `item`. `onset` is the start timestamp of the pronunced phone,
       `offset` is the end timestamp of the pronunciation and `phone`
       is a string representation of the phone. `tstart` and `tstop`
       are expressed in seconds.

    Attributes
    ----------
    phones_set : set
        The set of phones present in the alignment

    Raises
    ------
    ValueError
        If one element of `data` is not a quadruplet, if one `tstart`
        or `tstop` cannot be casted to float

    See Also
    --------
    Alignment
        The values of the AlignmentCollection dictionary

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
                self[item] = Alignment.from_list(
                    data, phones_set=self.phones_set, validate=True)
            except ValueError as err:
                raise ValueError('item {}: {}'.format(item, err))

    def save(self, filename, sort=False, compress=False):
        """Save the alignments to a `filename`

        The saved alignment is in text format (encoded in utf8), each
        line structured as::

            <item> <onset> <offset> <phone>

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

    @staticmethod
    def load(alignment_file, compress=False):
        """Returns an AlignmentCollection loaded from the `alignment_file`

        The file, in text format and optionally compressed, is read as
        utf8. It must be composed of lines with the following 4
        fields:

            <item> <onset> <offset> <phone>

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
    phones linked with their onset and offset timestamps. See the
    `validate` method for a list constraints applying to the `data`.

    Attributes
    ----------
    times : array of float, shape = [nphones, 2]
        The array of (onset, offset) timestamps for each aligned phone
    phones : array of str, shape = [nphones, 1]
        The array of aligned phones
    phones_set : set
        The set of the distinct phones present in the alignement
    validate : bool, optional
        When True, checks the alignment is in a valid format, when
        False does not perform any verification, default is True

    Raises
    ------
    ValueError
        When `validate` is True and the alignment data is not
        correctly formatted

    """
    def __init__(self, times, phones, phones_set=None, validate=True):
        self._times = times
        self.phones = phones

        # if the phone set is unspecified, take it from the alignment
        if phones_set is None:
            self.phones_set = set(self.phones)
        else:
            self.phones_set = phones_set

        if validate is True:
            self.validate()

    @property
    def onsets(self):
        """The start timestamps of the aligned phones in seconds"""
        return self._times[:, 0]

    @property
    def offsets(self):
        """The stop timestamps of the aligned phones in seconds"""
        return self._times[:, 1]

    @staticmethod
    def from_list(data, phones_set=None, validate=True):
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
        phones_set : set,
            optional The set of the different phones present in the
            alignment. If not specified the `phones_set` is built from
            the phones in `data`.

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
            times, phones, phones_set=phones_set, validate=validate)

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

        # make sure all phones in alignment are present in phones_set
        diff = set(self.phones).difference(set(self.phones_set))
        if len(diff) != 0:
            raise ValueError(
                'following phones are in alignment but not in phones_set: '
                '{}'.format(', '.join(diff)))

    def is_valid(self):
        """Returns True if the Alignment is consistent, False otherwise"""
        try:
            self.validate()
        except ValueError:
            return False
        return True

    def __eq__(self, other):
        return (np.array_equal(self._times, other._times)
                and np.array_equal(self.phones, other.phones)
                and self.phones_set == other.phones_set)

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
            return Alignment(
                np.array([]), np.array([]),
                phones_set=self.phones_set,
                validate=False)

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

        return Alignment(
            times, phones,
            phones_set=self.phones_set,
            validate=False)

    def __repr__(self):
        return '\n'.join(
            '{} {} {}'.format(t0, t1, p)
            for (t0, t1, p) in self.to_list())

    def to_list(self):
        """Returns the alignment as a list of triplets (onset, offset, phone)

        See Also
        --------
        Alignment.from_list
            The reverse operation

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
