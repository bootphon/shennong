""""""

import collections
import os
import random
import warnings

from shennong import Audio


valid_formats = {
    2: '<utterance-id> <audio-file>',
    3: '<utterance-id> <audio-file> <speaker-id>',
    4: '<utterance-id> <audio-file> <tstart> <tstop>',
    5: '<utterance-id> <audio-file> <speaker-id> <tstart> <tstop>'}


class Utterance:
    def __init__(self, name, audio, speaker=None, tstart=None, tstop=None):
        self._name = name
        self._audio = audio
        self._speaker = speaker

        if (
                (tstart is None and tstop is not None) or
                (tstop is None and tstart is not None)
        ):
            raise ValueError('both tstart and tstop must be defined or None')
        if tstart is not None and (tstart < 0 or tstart >= tstop):
            raise ValueError('we must have 0 <= tstart < tstop')

        if name is None:
            if not (speaker is None and tstart is None and tstop is None):
                raise ValueError('invalid format for utterance')
            self._format = 1
        elif speaker is None:
            self._format = 2 if tstart is None else 4
        elif tstart is None:
            self._format = 3
        else:
            self._format = 5

        self._duration = Audio.scan(audio).duration
        if tstart is not None:
            if tstop < self._duration:
                raise ValueError(
                    f'{audio}: file duration is {self._duration} but asking '
                    'interval ({tstart}, {tstop})')
            self._duration = tstop - tstart

        self._tstart = tstart
        self._tstop = tstop

    @property
    def name(self):
        return self._name

    @property
    def audio_file(self):
        return self._audio

    @property
    def audio_data(self):
        data = Audio.load(self._audio)
        if self.tstart or self.tstop:
            data = data.segment([(self.tstart, self.tstop)])[0]
        return data

    @property
    def speaker(self):
        return self._speaker

    @property
    def tstart(self):
        return self._tstart

    @property
    def tstop(self):
        return self._tstop

    @property
    def duration(self):
        return self._duration

    def format(self, as_string=False):
        return valid_formats[self._format] if as_string else self._format


class Utterances:
    def __init__(self, utterances):
        # parse the utterances format
        self._format = set(u.format for u in utterances)
        if not len(self._format) == 1:
            raise ValueError('the utterances format is not homogeneous')
        self._format = list(self._format)[0]

        # ensure utterances have unique names
        duplicates = [u for u, c in collections.Counter(
            u.audio if self._format == 1 else u.name
            for u in utterances).items() if c > 1]
        if duplicates:
            raise ValueError(
                f'duplicates found in utterances: {", ".join(duplicates)}')

        # sort the utterances by audio, this is a minor
        # optimization to use the cache system of Audio.load(), ie this avoids
        # to reload several times the same wav when using tstart/tstop
        # segments.
        utterances = sorted(utterances, key=lambda u: u.audio)

        # build the utterances collection indexed by id
        self._by_name = {u.name: u for u in utterances}

        self._by_speaker = {}
        if self.with_speakers():
            for utt in self.by_name.values():
                if utt.speaker not in self._by_speaker:
                    self._by_speaker[utt.speaker] = []
                self._by_speaker[utt.speaker].append(utt)

    @classmethod
    def load(cls, filename):
        """Returns utterances loaded from a file"""
        if not os.path.isfile(filename):
            raise ValueError(f'{filename} not found')

        utterances = []
        for line in (line.strip() for line in open(filename, 'r') if line):
            line = line.split(' ')
            if len(line) == 1:
                utt = Utterance(line[0])
            elif len(line) == 2:
                utt = Utterance(line[1], name=line[0])
            elif len(line) == 3:
                utt = Utterance(line[1], name=line[0], speaker=line[2])
            elif len(line) == 4:
                utt = Utterance(
                    line[1], name=line[0], tstart=line[2], tstop=line[3])
            elif len(line) == 5:
                utt = Utterance(
                    line[1], name=line[0], speaker=line[2],
                    tstart=line[4], tstop=line[4])
            else:
                raise ValueError(f'invalid utterance format: {line}')
            utterances.append(utt)

        return cls(utterances)

    def format(self, as_string=False):
        """The utterances format"""
        return valid_formats[self._format] if as_string else self._format

    def with_speakers(self):
        """Returns True if there is speaker information, False otherwise"""
        return self._format in (3, 5)

    @property
    def by_speaker(self):
        """A dict (speaker: list of utternances)

        Raises
        ------
        ValueError if there is no speaker information

        """
        if not self.with_speakers():
            raise ValueError(
                f'utterances have no speaker information: {self.format}')
        return self._by_speaker

    @property
    def by_name(self):
        """A dict (name: utterance)"""
        return self._by_name

    def as_list(self):
        """Returns utterances as a list of (<utterance-id> <...>)"""
        return [
            (index,) + tuple(info for info in utt if info)
            for index, utt in self.by_name.items()]

    def duration(self):
        """Returns the total duration os utterances"""
        return sum(utt.duration for utt in self.by_name.values)

    def duration_per_speaker(self, duration, truncate=False, shuffle=True):
        """Returns utterances segments, keeping ``duration`` sec per speaker

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
            False, take thme in order. Default to True.

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
        for speaker, utterances in self.by_speaker.items():
            if shuffle:
                random.shuffle(utterances)

            remaining_duration = duration
            for utt in utterances:
                if utt.duration >= remaining_duration:
                    segments.append(Utterance(
                        utt.name, utt.audio, speaker=utt.speaker,
                        tstart=0, tstop=remaining_duration))
                    remaining_duration = 0
                    break

                segments.append(utt)
                remaining_duration -= utt.duration

            if remaining_duration > 0:
                message = (
                    f'speaker {speaker}: only {duration - remaining_duration}s '
                    f'of audio available but {duration}s requested')
                if truncate:
                    warnings.warn(message)
                else:
                    raise ValueError(message)

        return Utterances(segments)
