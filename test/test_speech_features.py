"""Test of the speech-features binary"""

import shlex
import shutil
import subprocess
import tempfile

import pytest
import yaml

from shennong import FeaturesCollection, Utterances


def get_config(args):
    with tempfile.NamedTemporaryFile() as tmp:
        command = f'speech-features config {args} -o {tmp.name}'
        subprocess.run(shlex.split(command), check=True)
        return yaml.safe_load(open(tmp.name, 'r'))


def get_features(args_conf, utterances):
    with tempfile.TemporaryDirectory() as tmpdir:
        command = f'speech-features config {args_conf} -o {tmpdir}/conf'
        subprocess.run(shlex.split(command), check=True)

        utterances.save(f'{tmpdir}/utts')

        command = (
            f'speech-features extract -j 3 '
            f'{tmpdir}/conf {tmpdir}/utts {tmpdir}/feats.pkl')
        subprocess.run(shlex.split(command), check=True)

        return FeaturesCollection.load(f'{tmpdir}/feats.pkl')


@pytest.mark.skipif(
    not shutil.which('speech-features'),
    reason='speech-features command not found')
def test_config():
    assert ['mfcc'] == list(get_config('mfcc --no-comments').keys())
    assert {'mfcc', 'cmvn', 'delta', 'pitch'} == set(
        get_config('mfcc --delta --cmvn --pitch crepe --no-comments').keys())


@pytest.mark.skipif(
    not shutil.which('speech-features'),
    reason='speech-features command not found')
def test_extract(wav_file):
    utts = Utterances([
        ('utt1', wav_file, 'spk1', 0, 0.5),
        ('utt2', wav_file, 'spk1', 0, 1),
        ('utt3', wav_file, 'spk2', 0, 1.2)])

    feats = get_features('mfcc --delta', utts)
    assert feats.keys() == utts.by_name().keys()
    assert feats['utt1'].ndims == 39
