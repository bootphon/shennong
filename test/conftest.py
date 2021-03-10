"""Defines project-wide fixtures for unit testing"""

import os
import struct
import numpy as np
import pytest

from shennong import Audio, Features, FeaturesCollection
from shennong.alignment import AlignmentCollection
from shennong.processor.mfcc import MfccProcessor


@pytest.fixture(scope='session')
def data_path():
    return os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture(scope='session')
def alignment_file(data_path):
    return os.path.join(data_path, 'alignment.txt')


@pytest.fixture(scope='session')
def alignments(alignment_file):
    return AlignmentCollection.load(alignment_file)


@pytest.fixture(scope='session')
def wav_file(data_path):
    return os.path.join(data_path, 'test.wav')


@pytest.fixture(scope='session')
def audio(wav_file):
    return Audio.load(wav_file)


@pytest.fixture(scope='session')
def wav_file_8k(data_path):
    return os.path.join(data_path, 'test.8k.wav')


@pytest.fixture(scope='session')
def wav_file_float32(data_path):
    return os.path.join(data_path, 'test.float32.wav')


@pytest.fixture(scope='session')
def audio_8k(wav_file_8k):
    return Audio.load(wav_file_8k)


@pytest.fixture(scope='session')
def audio_tiny(audio):
    return Audio(audio.data[:10], audio.sample_rate)


@pytest.fixture(scope='session')
def mfcc(audio):
    return MfccProcessor().process(audio)


@pytest.fixture(scope='session')
def bottleneck_original(data_path):
    fea_file = os.path.join(data_path, 'test.bottleneck.fea')

    # this is BottleneckFeaturesExtraction.utils.read_htk(file)
    try:
        fh = open(fea_file, 'rb')
    except TypeError:
        fh = fea_file
    try:
        nSamples, sampPeriod, sampSize, parmKind = struct.unpack(
            ">IIHH", fh.read(12))
        m = np.frombuffer(fh.read(nSamples * sampSize), 'i1')
        m = m.view('>f').reshape(nSamples, int(sampSize/4))
    finally:
        if fh is not fea_file:
            fh.close()
    return m


@pytest.fixture(scope='session')
def features_collection():
    # build a collection of 3 random features of same ndims, various
    # nframes
    dim = 10
    feats = FeaturesCollection()
    for n in range(3):
        nframes = np.random.randint(5, 15)
        feats[str(n)] = Features(
            np.random.random((nframes, dim)),
            np.arange(0, nframes))
    return feats
