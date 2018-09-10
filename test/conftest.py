import os
import pytest

from shennong.features.audio import AudioData


@pytest.fixture(scope='session')
def wav_file():
    return os.path.join(os.path.dirname(__file__), 'data', 'test.wav')


@pytest.fixture(scope='session')
def audio(wav_file):
    return AudioData.load(wav_file)
