"""Test of the module shennong.utterances"""

import numpy as np
import pytest
from shennong.utterances import Utterance, Utterances


def test_utterance_bad(wav_file):
    with pytest.raises(ValueError) as err:
        Utterance(())
    assert 'invalid utterance format' in str(err.value)

    with pytest.raises(ValueError) as err:
        Utterance(0)
    assert 'invalid utterance format' in str(err.value)

    with pytest.raises(ValueError) as err:
        Utterance(0, 0, 0, 0, 0, 0)
    assert 'invalid utterance format' in str(err.value)

    with pytest.raises(ValueError) as err:
        Utterance(0, 0)
    assert '0: file not found' in str(err.value)

    with pytest.raises(ValueError) as err:
        Utterance(0, wav_file, None, 1)
    assert 'both tstart and tstop must be defined or None' in str(err.value)

    with pytest.raises(ValueError) as err:
        Utterance(0, wav_file, 0, None)
    assert 'both tstart and tstop must be defined or None' in str(err.value)

    with pytest.raises(ValueError) as err:
        Utterance(0, wav_file, 'spk', 1, 0)
    assert 'we must have 0 <= tstart < tstop' in str(err.value)

    with pytest.raises(ValueError) as err:
        Utterance(0, wav_file, -1, 0)
    assert 'we must have 0 <= tstart < tstop' in str(err.value)

    with pytest.raises(ValueError) as err:
        Utterance(0, wav_file, 'abc', 0)
    assert 'cannot cast tstart as float' in str(err.value)

    with pytest.raises(ValueError) as err:
        Utterance(0, wav_file, 1, 'abc')
    assert 'cannot cast tstop as float' in str(err.value)


def test_utterance_f1(audio, wav_file):
    utt = Utterance('name', wav_file)
    assert utt.format == 1
    assert utt.name == 'name'
    assert utt.audio_file == wav_file
    assert utt.speaker is None
    assert utt.tstart is None
    assert utt.tstop is None
    assert utt.duration == pytest.approx(audio.duration)
    assert utt.load_audio() == audio
    assert str(utt) == f'name {wav_file}'


def test_utterance_f2(audio, wav_file):
    utt = Utterance('name', wav_file, 'spk')
    assert utt.format == 2
    assert utt.name == 'name'
    assert utt.audio_file == wav_file
    assert utt.speaker == 'spk'
    assert utt.tstart is None
    assert utt.tstop is None
    assert utt.duration == pytest.approx(audio.duration)
    assert utt.load_audio() == audio
    assert str(utt) == f'name {wav_file} spk'


def test_utterance_f3(audio, wav_file):
    utt = Utterance('name', wav_file, 0, 1)
    assert utt.format == 3
    assert utt.name == 'name'
    assert utt.audio_file == wav_file
    assert utt.speaker is None
    assert utt.tstart == 0
    assert utt.tstop == 1
    assert utt.duration == 1
    assert np.all(utt.load_audio().data == audio.data[:16000])
    assert str(utt) == f'name {wav_file} 0.0 1.0'


def test_utterance_f4(audio, wav_file):
    utt = Utterance('name', wav_file, 'spk', 0, 1)
    assert utt.format == 4
    assert utt.name == 'name'
    assert utt.audio_file == wav_file
    assert utt.speaker == 'spk'
    assert utt.tstart == 0
    assert utt.tstop == 1
    assert utt.duration == 1
    assert np.all(utt.load_audio().data == audio.data[:16000])
    assert str(utt) == f'name {wav_file} spk 0.0 1.0'


def test_utterance_truncate(wav_file, audio):
    with pytest.warns(UserWarning) as warn:
        utt = Utterance('name', wav_file, 'spk', 0, 10)
    assert 'asking interval (0.0, 10.0)' in warn[0].message.args[0]
    assert utt.duration == pytest.approx(audio.duration)
    assert utt.load_audio() == audio

    with pytest.warns(UserWarning) as warn:
        utt = Utterance('name', wav_file, 'spk', 1, 5)
    assert 'asking interval (1.0, 5.0)' in warn[0].message.args[0]
    assert utt.duration + 1 == pytest.approx(audio.duration)
    assert np.all(utt.load_audio().data == audio.data[16000:])


def test_utterances_bad(wav_file):
    with pytest.raises(ValueError) as err:
        Utterances([])
    assert 'empty input utterances' in str(err.value)

    with pytest.raises(ValueError) as err:
        Utterances([(0,)])
    assert 'invalid utterance format: (0,)' in str(err.value)

    with pytest.raises(ValueError) as err:
        Utterances([('utt1', wav_file), 0])
    assert 'utterance must be an iterable' in str(err.value)

    with pytest.raises(ValueError) as err:
        Utterances([('utt1', wav_file), ('utt2', wav_file, 'spk')])
    assert 'utterances format is not homogeneous' in str(err.value)

    with pytest.raises(ValueError) as err:
        Utterances([('utt1', wav_file), ('utt1', wav_file)])
    assert 'duplicates found' in str(err.value)


def test_utterances_spk(wav_file, wav_file_8k):
    utterances = Utterances([
        ['utt1', wav_file, 'spk1', 0, 1],  # try with list
        ('utt2', wav_file_8k, 'spk1', 0, 1.2),
        ('utt3', wav_file, 'spk2', 0, 1)])

    assert len(utterances) == 3
    # utterances sorted by filename
    assert [u.name for u in utterances] == ['utt2', 'utt1', 'utt3']
    assert utterances['utt1'].name == 'utt1'
    assert utterances['utt1'].tstop == 1

    assert utterances.format() == 4
    assert utterances.format(type=int) == 4
    assert utterances.format(type=str) == (
        '<utterance-id> <audio-file> <speaker-id> <tstart> <tstop>')

    assert utterances.has_speakers()
    assert len(utterances.by_speaker()['spk1']) == 2
    assert len(utterances.by_speaker()['spk2']) == 1
    assert sorted(utterances.by_speaker().keys()) == ['spk1', 'spk2']

    assert list(utterances.by_name().keys()) == ['utt2', 'utt1', 'utt3']
    assert sorted(utterances.by_name().keys()) == ['utt1', 'utt2', 'utt3']
    assert utterances.duration() == 3.2


def test_utterances_nospk(wav_file, wav_file_8k):
    utterances = Utterances([
        ('utt1', wav_file, 0, 1),
        ('utt2', wav_file_8k, 0, 1.2),
        ('utt3', wav_file, 0, 1)])

    assert len(utterances) == 3
    # utterances sorted by filename
    assert [u.name for u in utterances] == ['utt2', 'utt1', 'utt3']
    assert utterances['utt1'].name == 'utt1'
    assert utterances['utt1'].tstop == 1
    assert utterances.format() == 3
    assert utterances.format(type=int) == 3
    assert utterances.format(type=str) == (
        '<utterance-id> <audio-file> <tstart> <tstop>')

    assert not utterances.has_speakers()
    with pytest.raises(ValueError) as err:
        utterances.by_speaker()
    assert 'utterances have no speaker information' in str(err.value)

    assert list(utterances.by_name().keys()) == ['utt2', 'utt1', 'utt3']
    assert sorted(utterances.by_name().keys()) == ['utt1', 'utt2', 'utt3']
    assert utterances.duration() == 3.2


def test_save_load(wav_file, tmpdir):
    filename = str(tmpdir / 'utts')
    utts = Utterances([
        ('utt1', wav_file, 0, 1),
        ('utt2', wav_file, 0, 1.2),
        ('utt3', wav_file, 0, 1)])

    utts.save(filename)
    assert Utterances.load(filename) == utts

    with pytest.raises(ValueError):
        Utterances.load('/spam/spam/i/love/spam')


def test_fit_duration_bad(wav_file):
    utts = Utterances([
        ('utt1', wav_file, 0, 0.5),
        ('utt2', wav_file, 0, 1),
        ('utt3', wav_file, 0, 1.2)])

    with pytest.raises(ValueError) as err:
        utts.fit_to_duration(10)
    assert 'utterances have no speaker information' in str(err.value)

    utts = Utterances([('utt1', wav_file, 'spk', 0, 0.5)])
    with pytest.raises(ValueError) as err:
        utts.fit_to_duration(0)
    assert 'duration must be a positive number' in str(err.value)


@pytest.mark.parametrize('shuffle', (False, True))
def test_fit_duration(wav_file, shuffle):
    utts = Utterances([
        ('utt1', wav_file, 'spk1', 0, 0.5),
        ('utt2', wav_file, 'spk1', 0, 1),
        ('utt3', wav_file, 'spk2', 0, 1.2)])

    fit = utts.fit_to_duration(1, shuffle=shuffle)
    assert fit.duration() == 2
    if not shuffle:
        assert fit == Utterances([
            ('utt1', wav_file, 'spk1', 0, 0.5),
            ('utt2', wav_file, 'spk1', 0, 0.5),
            ('utt3', wav_file, 'spk2', 0, 1)])

    with pytest.raises(ValueError) as err:
        utts.fit_to_duration(1.5, shuffle=shuffle)
    assert (
        'speaker spk2: only 1.2s of audio available but 1.5s requested'
        in str(err.value))

    with pytest.warns(UserWarning) as warn:
        fit = utts.fit_to_duration(1.5, shuffle=shuffle, truncate=True)
    assert (
        'speaker spk2: only 1.2s of audio available but 1.5s requested'
        in warn[0].message.args[0])
    if not shuffle:
        assert fit == utts
