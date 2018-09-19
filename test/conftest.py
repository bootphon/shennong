"""Defines project-wide fixtures for unit testing"""

import os
import pytest

from shennong.audio import AudioData
from shennong.features.mfcc import MfccProcessor
from shennong.alignment import AlignmentCollection


@pytest.fixture(scope='session')
def alignment_file():
    return os.path.join(os.path.dirname(__file__), 'data', 'alignment.txt')


@pytest.fixture(scope='session')
def alignments(alignment_file):
    return AlignmentCollection.load(alignment_file)


@pytest.fixture(scope='session')
def wav_file():
    return os.path.join(os.path.dirname(__file__), 'data', 'test.wav')


@pytest.fixture(scope='session')
def audio(wav_file):
    return AudioData.load(wav_file)


@pytest.fixture(scope='session')
def mfcc(audio):
    return MfccProcessor().process(audio)
