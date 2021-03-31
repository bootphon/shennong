"""Provides the :class:`Uttterance` and :class:`Utterances` classes

An utterance correspond to a sentence, or a speech segment, that is processed
individually by an extraction pipeline. An utterance is defined by one of the
following format:

* 2-uple: ``<utterance-id> <audio-file>``
* 3-uple: ``<utterance-id> <audio-file> <speaker-id>``
* 4-uple: ``<utterance-id> <audio-file> <tstart> <tstop>``
* 5-uple: ``<utterance-id> <audio-file> <speaker-id> <tstart> <tstop>``

.. note::

   Most of ``shennong`` components (processors and post processors) work
   directly on individual audio files. Utterances are used when training a
   :class:`~shennong.processor.vtln.VtlnProcessor` or extracting features from
   a :mod:`shennong.pipeline`.

"""

import collections
import os
import random
import warnings

from shennong import Audio


VALID_FORMATS = {
    1: '<utterance-id> <audio-file>',
    2: '<utterance-id> <audio-file> <speaker-id>',
    3: '<utterance-id> <audio-file> <tstart> <tstop>',
    4: '<utterance-id> <audio-file> <speaker-id> <tstart> <tstop>'}
"""The valid formats for an utterance, as detailed above"""


class Utterance:
    """Manage a single utterance

    The class :class:`Utterance` manages individual utterances and basically
    give access to their components: name, speaker, corresponding audio
    segment. The utterance must be defined by one of the formats defined above.

    Parameters
    ----------
    *args:
        The arguments must be 2, 3, 4 or 5. The number of arguments defines the
        utterance format and the signification of each positional argument (see
        :data:`VALID_FORMATS`)

    Raises
    ------
    ValueError
        If the arguments are not 2, 3, 4 or 5, or if the utterance
        cannot be created from them (for instance the audio file is not
        readable)

    """
    def __init__(self, *args):
        if len(args) < 2 or len(args) > 5:
            raise ValueError(f'invalid utterance format: {args}')

        # read the utterances from input fields
        self._format = len(args) - 1
        self._name = args[0]
        self._audio = args[1]
        self._speaker = None
        self._tstart = None
        self._tstop = None
        if len(args) == 3:
            self._speaker = args[2]
        elif len(args) == 4:
            self._tstart = args[2]
            self._tstop = args[3]
        elif len(args) == 5:
            self._speaker = args[2]
            self._tstart = args[3]
            self._tstop = args[4]

        # cast tstart and tstop as float
        if self._tstart is not None:
            try:
                self._tstart = float(self._tstart)
            except ValueError:
                raise ValueError(
                    f'cannot cast tstart as float: {self._tstart}') from None
        if self._tstop is not None:
            try:
                self._tstop = float(self._tstop)
            except ValueError:
                raise ValueError(
                    f'cannot cast tstop as float: {self._tstop}') from None

        # ensures tstart and tstop are valid
        if (
                (self._tstart is None and self._tstop is not None) or
                (self._tstop is None and self._tstart is not None)
        ):
            raise ValueError('both tstart and tstop must be defined or None')
        if (
                self._tstart is not None and
                (self._tstart < 0 or self._tstart >= self._tstop)
        ):
            raise ValueError('we must have 0 <= tstart < tstop')

        # compute the utterance duration, warns if tstop if beyond audio
        # boundaries. Scanning the audio file raises if the file is not found
        # nor valid.
        self._duration = Audio.scan(self._audio).duration
        if self._tstart is not None:
            if self._tstop > self._duration:
                warnings.warn(
                    f'{self._audio}: file duration is {self._duration} but '
                    f'asking interval ({self._tstart}, {self._tstop}), '
                    f'will be truncated')
                self._tstop = self._duration
            self._duration = self._tstop - self._tstart

    def __eq__(self, other):
        return str(self) == str(other)

    @property
    def format(self):
        """The utterance format code"""
        return self._format

    @property
    def name(self):
        """The utterance name, or <utterance-id>"""
        return self._name

    @property
    def audio_file(self):
        """The audio file attached to the utterance"""
        return self._audio

    @property
    def speaker(self):
        """The utterance speaker, or None if no speaker information"""
        return self._speaker

    @property
    def tstart(self):
        """The utterance onset time in the audio file, or None"""
        return self._tstart

    @property
    def tstop(self):
        """The utterance offset time in the audio file, or None"""
        return self._tstop

    @property
    def duration(self):
        """The utterance duration in seconds"""
        return self._duration

    def __str__(self):
        if self._format == 1:
            return f'{self.name} {self.audio_file}'
        if self._format == 2:
            return f'{self.name} {self.audio_file} {self.speaker}'
        if self._format == 3:
            return f'{self.name} {self.audio_file} {self.tstart} {self.tstop}'
        # format == 4
        return (
            f'{self.name} {self.audio_file} {self.speaker} '
            f'{self.tstart} {self.tstop}')

    def load_audio(self):
        """Returns the utterance's :class:`~shennong.audio.Audio` data"""
        data = Audio.load(self._audio)
        if self.tstart or self.tstop:
            data = data.segment([(self.tstart, self.tstop)])[0]
        return data


class Utterances:
    """Manages a collection of :class:`Utterance`.

    The :class:`Utterances` manages a collection of utterances and allows to
    iterate over the utterances by name or by speaker, as well as generating
    sub-utterances fit to a particular duration.

    The following conditions apply:

    * All utterances in the collection must have the same format
    * All utterances must have a unique name

    Parameters
    ----------
    utterances : list of :class:`Utterance` or list of tuples
        The utterances to be stored

    Raises
    ------
    ValueError
        If the utterances cannot be created because of the above conditions, or
        because one of the utterances if not valid

    """
    def __init__(self, utterances):
        # build it as a list of Utterance
        utterances = self._parse(utterances)
        if not utterances:
            raise ValueError('empty input utterances')

        # parse the utterances format
        self._format = set(utt.format for utt in utterances)
        if not len(self._format) == 1:
            raise ValueError('utterances format is not homogeneous')
        self._format = list(self._format)[0]

        # ensure utterances have unique names
        duplicates = [u for u, c in collections.Counter(
            u.name for u in utterances).items() if c > 1]
        if duplicates:
            raise ValueError(
                f'duplicates found in utterances: {", ".join(duplicates)}')

        # sort the utterances by audio, this is a minor
        # optimization to use the cache system of Audio.load(), ie this avoids
        # to reload several times the same wav when using tstart/tstop
        # segments.
        utterances = sorted(utterances, key=lambda u: (u.audio_file, u.name))

        # build the utterances collection indexed by name
        self._utterances = {u.name: u for u in utterances}

    @staticmethod
    def _parse(utterances):
        parsed = []
        for utt in utterances:
            if not isinstance(utt, Utterance):
                try:
                    utt = Utterance(*utt)
                except TypeError:
                    raise ValueError(
                        f'utterance must be an iterable, not {utt}') from None
            parsed.append(utt)
        return parsed

    def __len__(self):
        return len(self._utterances)

    def __iter__(self):
        return iter(self._utterances.values())

    def __getitem__(self, name):
        return self._utterances[name]

    def __eq__(self, other):
        return self._utterances == other._utterances

    @classmethod
    def load(cls, filename):
        """Returns utterances loaded from a file

        All the lines in the must conform to the same utterance format.

        Parameters
        ----------
        filename : str
            The file to load

        Raises
        ------
        ValueError
            If the ``filename`` is not found, if all the utterances do not have
            the same format, if all the <utterance-id> are not unique or if
            some defined utterances are not valid (audio file not found for
            instance).

        """
        if not os.path.isfile(filename):
            raise ValueError(f'{filename} not found')

        lines = (line.strip() for line in open(filename, 'r'))
        utterances = [line.split(' ') for line in lines if line]
        return cls(utterances)

    def save(self, filename):
        """Writes the utterances to file

        Parameters
        ----------
        filename: str
            The filename to write

        """
        open(filename, 'w').write('\n'.join(str(utt) for utt in self) + '\n')

    def format(self, type=int):
        """Returns the utterances format

        Parameters
        ----------
        type : optional, int or str
            When int return the format code, when str returns it's string
            representation

        Raises
        ------
        ValueError
            If ``type`` is not int or str

        """
        return VALID_FORMATS[self._format] if type is str else self._format

    def has_speakers(self):
        """Returns True if there is speaker information, False otherwise"""
        return self.format(type=int) in (2, 4)

    def by_speaker(self):
        """Returns a dictionary of utterances indexed by speaker

        The returned dictionary has speakers as keys and list of
        :class:`Utterance` as values.

        Raises
        ------
        ValueError
            If there is no speaker information

        """
        if not self.has_speakers():
            raise ValueError('utterances have no speaker information')

        by_speaker = collections.defaultdict(list)
        for utt in self:
            by_speaker[utt.speaker].append(utt)
        return by_speaker

    def by_name(self):
        """Returns a dictonary of utterances indexed by name

        The returned dictionary has utterance names as keys and
        :class:`Utterance` instances as values.

        """
        return self._utterances

    def duration(self):
        """Returns the total duration of the utterances in seconds"""
        return sum(utt.duration for utt in self)

    def fit_to_duration(self, duration, truncate=False, shuffle=False):
        """Returns a subset of utterances, keeping ``duration`` sec per speaker

        Parameters
        ----------
        duration : float
            The duration to keep per speaker, in seconds
        truncate : bool, optional
            When True, truncate the the total duration to the one available if
            there is not enough data. When False, raise an error if the
            duration cannot be returned for a speaker. Default to False.
        shuffle : bool, optional
            When True, shuffle the utterances before extracting segments. When
            False, take thme in order. Default to False.

        Returns
        -------
        utterances : :class:`Utterances`
            The utterances segments fitting the given ``duration`` for each
            speaker

        Raises
        ------
        ValueError
            When ``duration`` is not strictly positive or, when ``truncate`` is
            True, if a speaker has not enough data to build segments.

        """
        if duration <= 0:
            raise ValueError(
                f'duration must be a positive number, it is {duration}')

        segments = []
        for speaker, utterances in self.by_speaker().items():
            if shuffle:
                random.shuffle(utterances)

            remaining_duration = duration
            for utt in utterances:
                if utt.duration >= remaining_duration:
                    segments.append(Utterance(
                        utt.name, utt.audio_file,
                        utt.speaker, 0, remaining_duration))
                    remaining_duration = 0
                    break

                segments.append(utt)
                remaining_duration -= utt.duration

            if remaining_duration > 0:
                message = (
                    f'speaker {speaker}: only {duration - remaining_duration}s'
                    f' of audio available but {duration}s requested')
                if truncate:
                    warnings.warn(message)
                else:
                    raise ValueError(message)

        return Utterances(segments)
