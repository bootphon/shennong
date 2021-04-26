#!/usr/bin/env python
"""Comparison of pitch algorithms on various noise conditions"""

import argparse
import io
import pathlib
import pickle
import urllib.request
import warnings
import zipfile

import amfm_decompy.basic_tools
import amfm_decompy.pYAAPT
import matplotlib.pyplot as plt
import numpy as np
import parselmouth
import tqdm

from shennong import Audio, Features, FeaturesCollection
from shennong.frames import Frames
from shennong.processor import CrepePitchProcessor, KaldiPitchProcessor
from shennong.processor.pitch_crepe import predict_voicing, _nccf_to_pov


NOISE_LIST = ['gauss', 'babble']
"""Type of noises"""

SNR_LIST = [-15, -10, -5, 0, 5, 10, 15, 'inf']
"""SNR to apply noise to speech in dB"""

KEELE_URL = 'https://zenodo.org/record/3920591/files/KEELE.zip'
"""URL to download the KEELE dataset"""

BABBLE_NOISE_URL = (
    'https://docbox.etsi.org/stq/Open/EG%20202%20396-1%20Background%20noise'
    '%20database/Binaural_Signals/Pub_Noise_Binaural_V2.wav')
"""URL to download the babble noise"""


def download_keele_dataset(data_directory):
    """Downloads and unzips the KEELE dataset"""
    if (data_directory / 'raw' / 'KEELE').is_dir():
        return

    print(f'downloading dataset {data_directory}/raw/KEELE...')
    (data_directory / 'raw').mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(KEELE_URL).read())) \
           .extractall(data_directory / 'raw')


def download_babble_noise(data_directory):
    """Download the babble noise file"""
    filename = data_directory / 'raw' / 'babble_noise.wav'
    if filename.is_file():
        return

    print(f'downloading file {filename}...')
    urllib.request.urlretrieve(BABBLE_NOISE_URL, filename)


def snr(signal, noise, ratio):
    """Returns the signal to noise ratio (in dB)"""
    if ratio == 'inf':
        return signal

    # fill the noise with repeated copies of himself if shorter than signal
    noise = np.resize(noise, len(signal))
    return (
        signal + 10**(-ratio / 20) * np.linalg.norm(signal) /
        np.linalg.norm(noise) * noise).astype(np.int16)


def prepare_wavs(data_directory):
    """Adds Gaussian and babble noise at various SNRs on Keele dataset"""
    # for each KEELE file, generate babble and gaussian noisy versions at
    # various SNRs
    output_directory = data_directory / 'wavs'
    if output_directory.is_dir():
        return

    print(f'generating noisy dataset in {data_directory}/wavs...')
    output_directory.mkdir(parents=True)

    # retrieve wav files
    wavs = list(data_directory.glob('raw/KEELE/**/signal.wav'))
    assert len(wavs) == 10  # expect 10 wav files
    assert set(  # expect sampling rate at 20kHz
        Audio.scan(w).sample_rate for w in wavs) == set((20000,))

    # load the first channel of babble noise at 20kHz
    sample_rate = 20000
    babble_noise = Audio.load(data_directory / 'raw' / 'babble_noise.wav') \
                        .channel(0).resample(sample_rate)

    for wav in wavs:
        audio = Audio.load(wav)
        speaker = wav.parent.stem[:2]

        # various SNRs
        for ratio in SNR_LIST:
            filename = output_directory / f'{speaker}_babble_{ratio}.wav'
            if not filename.is_file():
                signal = snr(audio.data, babble_noise.data, ratio)
                Audio(signal, sample_rate).save(filename)

            filename = output_directory / f'{speaker}_gauss_{ratio}.wav'
            if not filename.is_file():
                signal = snr(
                    audio.data, np.random.normal(size=len(audio.data)), ratio)
                Audio(signal, sample_rate).save(filename)


def prepare_ground_truth(data_directory):
    """Retrieves pitch ground truth for evaluations"""
    output_file = data_directory / 'pitch' / 'ground_truth.h5f'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.is_file():
        return

    print('retrieving pitch ground truth...')
    truth = FeaturesCollection()
    for pitch in data_directory.glob('raw/KEELE/**/pitch.npy'):
        data = np.load(pitch)
        truth[pitch.parent.stem[:2]] = Features(
            np.atleast_2d(data['pitch']).T,
            # from https://lost-contact.mit.edu/afs/nada.kth.se/dept/tmh/
            # corpora/KeelePitchDB/Speech/keele_pitch_database.htm we have
            # pitch computed on 10ms steps over 25.6ms windows. Here we shift
            # time from frame beginning to middle time.
            data['time'] + 0.0128)
    truth.save(output_file)


class ComputePitchKaldi:
    """Kaldi pitch extractor"""
    # 20kHz is audio rate of KEELE, frame shift of 25.6ms to be online with
    # pitch ground truth from KEELE.
    name = 'kaldi'
    processor = KaldiPitchProcessor(
        sample_rate=20000, frame_shift=0.01, frame_length=0.0256)

    def todo(self, data_directory):
        """Return (noise, snr, filename) of pitches to compute"""
        jobs = []
        noise_ratio = [(n, r) for n in NOISE_LIST for r in SNR_LIST]
        for noise, ratio in noise_ratio:
            filename = (
                data_directory / 'pitch' / f'{self.name}_{noise}_{ratio}.h5f')
            if not filename.is_file():
                jobs.append((noise, ratio, filename))
        return jobs

    def process_one(self, wav):
        """Process a single wav"""
        return self.processor.process(Audio.load(wav))

    def process_all(self, data_directory):
        """Process the whole dataset"""
        jobs = self.todo(data_directory)
        if not jobs:
            return

        print(f'extracting {self.name} pitch...')
        output_directory = data_directory / 'pitch'
        output_directory.mkdir(parents=True, exist_ok=True)

        for noise, ratio, filename in tqdm.tqdm(jobs):
            pitch = FeaturesCollection()
            for wav in data_directory.glob(f'wavs/*_{noise}_{ratio}.wav'):
                pitch[wav.stem[:2]] = self.process_one(wav)
            pitch.save(filename)


class ComputePitchCrepe(ComputePitchKaldi):
    """CREPE pitch extractor"""
    # frame shift of 25.6ms to be online with pitch ground truth from KEELE.
    name = 'crepe'
    processor = CrepePitchProcessor(frame_shift=0.01, frame_length=0.0256)


class ComputePitchPraat(ComputePitchKaldi):
    """Praat pitch extractor"""
    # 20kHz is audio rate of KEELE, frame shift of 25.6ms to be online with
    # pitch ground truth from KEELE.
    name = 'praat'
    processor = None
    frames = Frames(
        sample_rate=20000, frame_shift=0.01, frame_length=0.0256)

    def process_one(self, wav):
        audio = Audio.load(wav)
        raw = parselmouth.Sound(
            audio.data, sampling_frequency=audio.sample_rate).to_pitch()
        times = self.frames.times(audio.nsamples)

        # linear interpolation of Praat pitch to be on same timestamps
        # thant ground truth and other models
        pitch = np.atleast_2d(np.nan_to_num(np.asarray(
            [raw.get_value_at_time(t) for t in times.mean(axis=1)]))).T
        return Features(pitch, times)


class ComputePitchYaapt(ComputePitchKaldi):
    """YAAPT pitch extractor"""
    name = 'yaapt'
    processor = None

    def process_one(self, wav):
        # frame shift of 25.6ms to be online with ground truth
        audio = amfm_decompy.basic_tools.SignalObj(wav)
        pitch = amfm_decompy.pYAAPT.yaapt(
            audio, frame_length=25.6, frame_space=10)
        return Features(
            np.atleast_2d(pitch.samp_values).T,
            pitch.frames_pos / audio.fs)


def compute_pitch(data_directory, model):
    """Compute pitch with given model"""
    return {
        'kaldi': ComputePitchKaldi,
        'crepe': ComputePitchCrepe,
        'praat': ComputePitchPraat,
        'yaapt': ComputePitchYaapt}[model]().process_all(data_directory)


def load_masked_data(data_directory):
    """Load computed pitches with unvoiced frames masked"""
    print('loading pitch data...')
    # load ground truth pitch values
    data = {'truth': FeaturesCollection.load(
        data_directory / 'pitch' / 'ground_truth.h5f')}

    # load computed pitch estimations
    for name in ('crepe', 'kaldi', 'praat', 'yaapt'):
        data[name] = {}
        for noise in NOISE_LIST:
            data[name][noise] = {
                ratio: FeaturesCollection.load(
                    data_directory / 'pitch' / f'{name}_{noise}_{ratio}.h5f')
                for ratio in SNR_LIST}

    nmasked = {}
    for speaker in data['truth'].keys():
        # get frames with valid pitch
        masks = {}
        masks['crepe'] = predict_voicing(
            data['crepe']['gauss']['inf'][speaker].data[:, 0])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            masks['kaldi'] = predict_voicing(_nccf_to_pov(
                data['kaldi']['gauss']['inf'][speaker].data[:, 0]))
        masks['praat'] = data['praat']['gauss']['inf'][speaker].data[:, 0] > 0
        masks['yaapt'] = data['yaapt']['gauss']['inf'][speaker].data[:, 0] > 0
        masks['truth'] = data['truth'][speaker].data[:, 0] > 0

        # get intersection of pitched frames for all models
        size = min(len(m) for m in masks.values())
        mask = np.logical_and.reduce([m[:size] for m in masks.values()])

        nmasked[speaker] = {k: (v.sum(), len(v)) for k, v in masks.items()}
        nmasked[speaker]['total'] = (mask.sum(), len(mask))

        # put at 0 all frames that are not in the intersection and resize
        # features to the minimal size
        data['truth'][speaker] = data['truth'][speaker].data[:size]
        data['truth'][speaker][mask == 0] = 0
        noise_ratio = ((n, r) for n in NOISE_LIST for r in SNR_LIST)
        for noise, ratio in noise_ratio:
            for name in ('crepe', 'kaldi', 'praat', 'yaapt'):
                feats = data[name][noise][ratio][speaker]
                feats = feats.data[:size]
                feats[mask == 0] = 0
                data[name][noise][ratio][speaker] = feats

    # sum up the proportion of masked frames per model
    print('proportion of voiced frames:')
    for model in ('truth', 'kaldi', 'crepe', 'praat', 'yaapt', 'total'):
        prop = (
            sum(mask[model][0] for mask in nmasked.values()) /
            sum(mask[model][1] for mask in nmasked.values()))
        print('   {}: {:.1f} %'.format(model, 100 * prop))

    return data


def mean_absolute_error(pitch, truth):
    """Returns the pitch MAE in Hz"""
    assert pitch.shape == truth.shape
    return np.mean(np.abs(pitch - truth))


def gross_error_rate(pitch, truth, rate=0.05):
    """Returns the pitch GER in % for a given `rate`"""
    assert pitch.shape == truth.shape
    return 100 * (np.abs(pitch - truth) > rate * truth).mean()


def compute_error_single(func, pitch, truth):
    """Compute pitch estimation error according to `func`"""
    speakers = truth.keys()
    errors = {
        'gauss': np.zeros((len(SNR_LIST), len(speakers))),
        'babble': np.zeros((len(SNR_LIST), len(speakers)))}
    for noise in NOISE_LIST:
        for i, ratio in enumerate(SNR_LIST):
            for j, k in enumerate(speakers):
                errors[noise][i, j] = func(
                    pitch[noise][ratio][k][:, -1],
                    truth[k][:, 0])
    return errors


def compute_error(data_directory):
    """Computes the whole estimation errors"""
    print('computing estimation errors...')
    data = load_masked_data(data_directory)
    error = {
        'mae': {model: compute_error_single(
            mean_absolute_error, data[model], data['truth'])
                for model in ('kaldi', 'crepe', 'praat', 'yaapt')},
        'ger': {model: compute_error_single(
            lambda x, y: gross_error_rate(x, y, rate=0.05),
            data[model], data['truth'])
                for model in ('kaldi', 'crepe', 'praat', 'yaapt')}}

    with open(data_directory / 'pitch' / 'error.pkl', 'wb') as handler:
        pickle.dump(error, handler)


def plot_error(data_directory, show=False):
    """Plot MAE / GER estimation errors, show the plots if `show` is True"""
    print('plotting estimation errors...')
    with open(data_directory / 'pitch' / 'error.pkl', 'rb') as handler:
        error = pickle.load(handler)
    (data_directory / 'plots').mkdir(exist_ok=True)

    xticks = np.array(SNR_LIST[:-1]+[20])
    for noise in NOISE_LIST:
        for metric, ylabel in (('mae', 'MAE (Hz)'), ('ger', r'GER (\%)')):
            filename = data_directory / 'plots' / f'{metric}_{noise}.pdf'
            plt.style.use('./plot.style')
            plt.figure(figsize=(6, 3))
            plt.grid(axis='both')

            models = (
                ('kaldi', 'Kaldi', 'o'),
                ('crepe', 'CREPE', 's'),
                ('praat', 'Praat', '^'),
                ('yaapt', 'YAAPT', 'v'))
            for model, label, marker in models:
                mean = error[metric][model][noise].mean(1)
                hstd = error[metric][model][noise].std(1) / 2
                plt.plot(xticks, mean, marker=marker, label=label)
                plt.fill_between(xticks, mean - hstd, mean + hstd, alpha=0.15)

            plt.xticks(xticks, SNR_LIST[:-1] + [r"$\infty$"])
            plt.xlabel('SNR (dB)')
            plt.ylabel(ylabel)
            plt.legend()

            # metrics share the same y-axis for easy comparison
            if metric == 'mae':
                plt.ylim(top=81)
            else:
                plt.ylim(top=41)

            plt.savefig(filename)
            print(f'... {filename}')
            if show:
                plt.show()
            else:
                plt.close()


def main():
    """Script entry point from command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_directory', type=pathlib.Path,
        help='where to write results and intermediate files')
    data_directory = parser.parse_args().data_directory

    # preparing data
    download_keele_dataset(data_directory)
    download_babble_noise(data_directory)
    prepare_wavs(data_directory)
    prepare_ground_truth(data_directory)

    # extracting pitch
    compute_pitch(data_directory, 'kaldi')
    compute_pitch(data_directory, 'praat')
    compute_pitch(data_directory, 'crepe')
    compute_pitch(data_directory, 'yaapt')

    # pitch estimation errors
    compute_error(data_directory)
    plot_error(data_directory, show=False)


if __name__ == '__main__':
    main()
